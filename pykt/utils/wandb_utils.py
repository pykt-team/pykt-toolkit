import os
import wandb
import numpy as np
import pandas as pd
import yaml
from wandb.apis.public import gql
import json
from multiprocessing.pool import ThreadPool # 线程池

def get_runs_result(runs):
    result_list = []
    for run in runs:
        result = {}
        result.update(run.summary._json_dict)
        model_config = {k: v for k, v in run.config.items()
                        if not k.startswith('_') and type(v) not in [list, dict]}
        result.update(model_config)
        result['name'] = run.name
        result['path_id'] = run.path[-1]
        result['state'] = run.state
        result_list.append(result)
    runs_df = pd.DataFrame(result_list)
    runs_df['create_time'] = runs_df['_timestamp']
    model_config_keys = list(model_config.keys())
    return runs_df,model_config_keys


class WandbUtils:
    """wandb utils

    wandb_api = WandbUtils(user='tabchen', project_name='pykt_iekt_pred')
    >self.sweep_dict is {'mx2tvwfy': ['mx2tvwfy']}
    
    """
    def __init__(self,user,project_name) -> None:
        self.user = user
        self.project_name = project_name
        self._init_wandb()
        

    def _init_wandb(self):
        self.api = wandb.Api(timeout=180)
        self.project = self.api.project(name=self.project_name)
        self.sweep_dict = self.get_sweep_dict()
        self.invert_sweep_dict = dict(zip(list(self.sweep_dict.values()),list(self.sweep_dict.keys())))
        print(f"self.sweep_dict is {self.sweep_dict}")
        self.sweep_keys = list(self.sweep_dict.keys())
        self.sweep_keys.sort()
    

    def get_sweep_dict(self):
        '''Get sweep dict'''
        sweep_dict = {}
        for sweep in self.project.sweeps():
            if sweep.name not in sweep_dict:
                sweep_dict[sweep.name] = []
            sweep_dict[sweep.name].append(sweep.id)
               
        for name in list(sweep_dict.keys()):
            if len(sweep_dict[name]) > 1:
                print(f"Error!! we can not process the same sweep name {name}, we will not return those sweeps:{sweep_dict[name]}")
                del sweep_dict[name]
            else:
                sweep_dict[name] = sweep_dict[name][0]
        return sweep_dict

    def _get_sweep_id(self,id,input_type):
        if input_type == "sweep_name":
            sweep_id = self.sweep_dict[id]
        else:
            sweep_id = id
        return sweep_id

    def get_df(self,id,input_type="sweep_name", drop_duplicate=False, drop_na=True, only_finish=True):
        """Get one sweep result

        Args:
            id (str): the sweep name or sweep id.
            input_type (str, optional): the type of id. Defaults to sweep_name.

        Returns:
            pd.Data: _description_
        """
        sweep_id = self._get_sweep_id(id,input_type)
        sweep = self.api.sweep(f"{self.user}/{self.project_name}/{sweep_id}")
        df,model_config_keys = get_runs_result(sweep.runs)
        
        if drop_na:
            df = df.dropna()
            df['create_time'] = df['_timestamp'].apply(int)
        if only_finish:
            df = df[df['state'] == 'finished'].copy()

        if drop_duplicate:
            df.drop_duplicates(model_config_keys)
        df = df.sort_values("create_time")
        df["run_index"] = range(len(df))
        df.index = range(len(df))
        return df

    def get_multi_df(self,id_list=[],input_type="sweep_name",drop_duplicate=False, drop_na=True, only_finish=True):
        """Get multi sweep result

        Args:
            id_list (list): the list of sweep name or sweep id.
            input_type (str, optional): the type of id. Defaults to sweep_name.

        Returns:
            _type_: _description_
        """
        df_list = []
        for id in id_list:
            df = self.get_df(id,input_type=input_type,drop_duplicate=drop_duplicate,drop_na=drop_na,only_finish=only_finish)
            df[input_type] = id
            df_list.append(df)
        return df_list

    def get_sweep_status(self,id,input_type="sweep_name"):
        """Get sweep run status

        Args:
            id (str): the sweep name or sweep id.
            input_type (str, optional): the type of id. Defaults to sweep_name.

        Returns:
            str: the state of sweep. 'RUNNING', 'CANCELED' or 'FINISHED'
        """
        query = gql(
            """query Sweep($project: String, $entity: String, $name: String!) {
                project(name: $project, entityName: $entity) {
                    sweep(sweepName: $name) {
                        id
                        name
                        bestLoss
                        config
                        state
                    }
                },
            }
            """)
        sweep_id = self._get_sweep_id(id,input_type)
        variables = {
                "entity": self.user,
                "project": self.project_name,
                "name": sweep_id}
        status = self.project.client.execute(query,variable_values=variables)['project']['sweep']['state']
        return status

    def get_sweep_run_num(self,id,input_type="sweep_name"):
        """Get sweep run num

        Args:
            id (str): the sweep name or sweep id.
            input_type (str, optional): the type of id. Defaults to sweep_name.

        Returns:
            int: the num of sweep run
        """
        sweep_id = self._get_sweep_id(id,input_type)
        sweep = self.api.sweep(f"{self.user}/{self.project_name}/{sweep_id}")
        return len(sweep.runs)


    def check_sweep_early_stop(self,id,input_type="sweep_name",metric="validauc",metric_type="max",min_run_num=200,patience=50,force_check_df=False):
        """Check sweep early stop

        Args:
            id (str): the sweep name or sweep id.
            input_type (str, optional): the type of id. Defaults to sweep_name.
            metric (str, optional): the metric to check. Defaults to validauc.
            metric_type (str, optional): the type of metric max or min. Defaults to max. metric_type=='min' todo
            min_run_num (int, optional): the min run num to check. Defaults to 200.
            patience (int, optional): the patience to stop. Defaults to 50.
            force_check_df: always check df, defalut is false.

        Returns:
            dict: {"state":state,'df':df,"num_run":num_run}, state is 'RUNNING', 'CANCELED' or 'FINISHED',df is the df of the sweep, num_run is the num of sweep run, -1 mean the sweep is finished to save time we will not check it again.
        """
        print(f"Start check {id}")
        sweep_id = self._get_sweep_id(id,input_type)
        sweep_status = self.get_sweep_status(sweep_id,input_type="sweep_id")
        
        report = {"stop_cmd":"","id":sweep_id,"sweep_name":self.invert_sweep_dict[sweep_id]}
        if force_check_df:
            df = self.get_df(sweep_id,input_type="sweep_id",only_finish=True)#get sweep result
            report['df'] = df

        if sweep_status in ['CANCELED','FINISHED']:
            report['state'] = True
            if 'df' in report:
                report['num_run'] = len(df)
            else:
                report['num_run'] = -1
        else:
            num_run = self.get_sweep_run_num(sweep_id,input_type="sweep_id")#get sweep run num
            report['num_run'] = num_run
            if num_run<min_run_num:
                report['state'] = False
            else:
                #
                if 'df' not in report:
                    df = self.get_df(sweep_id,input_type="sweep_id",only_finish=True)#get sweep result
                    report['df'] = df
                df[f'{metric}_precsion3'] = df[metric].apply(lambda x:round(x,3))#忽略 1e-3 级别的提升
                #find stop point
                finish = False
                for i in range(min_run_num,len(df)):
                    best_value = df[:i][f'{metric}_precsion3'].max()#get best value
                    first_best_index = df[df[f'{metric}_precsion3']==best_value]['run_index'].min()
                    not_improve_num = len(df[df['run_index'] >= first_best_index])
                    if not_improve_num > patience:#如果连续 patience 次没有提高，则停止
                        finish = True 
                        break
                if finish:
                    df = df[:i].copy()#only keep before stop point
                    report['not_improve_num'] = not_improve_num
                    stop_cmd = f"wandb sweep {self.user}/{self.project_name}/{sweep_id} --cancel"
                    print(f"    Run `{stop_cmd}` to stop the sweep.")
                    report['state'] = True
                    report['stop_cmd'] = stop_cmd
                    report['first_best_index'] = first_best_index
                else:
                    report['state'] = False
        print(f"    details: {id} state is {report['state']},num of runs is {report['num_run']}")
        print("-"*60+'\n')
        return report

    def stop_sweep(self,cmd):
        # os.system(cmd)
        print(f"We will stop the sweep, by {cmd}")

    
    def check_sweep_list(self, sweep_key_list, metric="validauc", metric_type="max", min_run_num=200, patience=50, force_check_df=False, stop=False,n_jobs=5):
        check_result_list = []

        def check_help(sweep_name, input_type='sweep_name',
                    metric=metric, metric_type=metric_type, min_run_num=min_run_num, patience=patience, force_check_df=force_check_df):
            check_result = self.check_sweep_early_stop(sweep_name, input_type=input_type,
                                                    metric=metric, metric_type=metric_type, min_run_num=min_run_num, patience=patience, force_check_df=force_check_df)
            return check_result
        p = ThreadPool(n_jobs)
        check_result_list = p.map(check_help, sweep_key_list)
        p.close()
        if stop:  # stop sweep
            for result in check_result_list:
                if result['State'] and result['stop_cmd'] != 0:
                    self.stop_sweep(result['stop_cmd'])
        return check_result_list

    def check_sweep_by_pattern(self,sweep_pattern,metric="validauc",metric_type="max",min_run_num=200,patience=50,force_check_df=False,stop=False,n_jobs=5):
        """Check sweeps by pattern
        
        Args:
            sweep_pattern (str): check the sweeps which sweep names start with sweep_pattern
            metric (str, optional): the metric to check. Defaults to validauc.
            metric_type (str, optional): the type of metric max or min. Defaults to max.
            min_run_num (int, optional): the min run num to check. Defaults to 200.
            patience (int, optional): the patience to stop. Defaults to 50.
            force_check_df: always check df, defalut is false.
            
        Returns:
            list: the list of dict, each dict is {"id":id,"state":state,'df':df,"num_run":num_run}, state is 'RUNNING', 'CANCELED' or 'FINISHED',df is the df of the sweep, num_run is the num of sweep run, -1 mean the sweep is finished to save time we will not check it again.
        """
        sweep_key_list = []
        for sweep_name in self.sweep_keys:
            if sweep_name.startswith(sweep_pattern) or sweep_pattern=='all':
                sweep_key_list.append(sweep_name)
        check_result_list = self.check_sweep_list(sweep_key_list,metric=metric,metric_type=metric_type,min_run_num=min_run_num,patience=patience,force_check_df=force_check_df,stop=stop,n_jobs=n_jobs)
        
        return check_result_list

    def get_all_fold_name(self,dataset_name,model_name,emb_type="qid"):
        sweep_key_list = [f"{dataset_name}_{model_name}_{emb_type}_{fold}" for fold in range(5)]
        sweep_key_list = [x for x in sweep_key_list if x in self.sweep_keys]#filter error
        return sweep_key_list

    def check_sweep_by_model_dataset_name(self,dataset_name,model_name,emb_type="qid",metric="validauc",metric_type="max",min_run_num=200,patience=50,force_check_df=False,stop=False,n_jobs=5):
        sweep_key_list = self.get_all_fold_name(dataset_name,model_name,emb_type)
        if len(sweep_key_list)!=5:
            print("Input error, please check")
            return 
        check_result_list = self.check_sweep_list(sweep_key_list,metric=metric,metric_type=metric_type,min_run_num=min_run_num,patience=patience,force_check_df=force_check_df,stop=stop,n_jobs=n_jobs)
        return check_result_list

    def get_best_run(self,dataset_name,model_name,emb_type="qid",metric="validauc",metric_type="max",min_run_num=200,patience=50,save_dir="results/wandb_result",n_jobs=5,force_reget=False):
        os.makedirs(save_dir,exist_ok=True)        
        best_path = os.path.join(save_dir,f"{dataset_name}_{model_name}_{emb_type}_best.csv")
        if os.path.exists(best_path) and not force_reget:
            df = pd.read_csv(best_path)
            print(f"Load from {best_path}")
        else:
            check_result_list = self.check_sweep_by_model_dataset_name(dataset_name,model_name,emb_type,metric=metric,metric_type=metric_type,min_run_num=min_run_num,patience=patience,force_check_df=True,n_jobs=n_jobs)
            row_list = []
            for result in check_result_list:
                df = result['df']
                df.to_csv(os.path.join(save_dir,result['sweep_name']+'.csv'),index=False)
                df = df.sort_values(metric,ascending=False)
                row_list.append(df.iloc[0])
            df = pd.DataFrame(row_list)
            df.to_csv(best_path,index=False)
        return df

    def get_model_run_time(self,dataset_name,model_name,emb_type="qid",metric="validauc",metric_type="max",min_run_num=200,patience=50,save_dir="results/wandb_result",n_jobs=5):
        """Get the average run second in one sweep
        """
        check_result_list = self.check_sweep_by_model_dataset_name(dataset_name,model_name,emb_type,metric=metric,metric_type=metric_type,min_run_num=min_run_num,patience=patience,force_check_df=True,n_jobs=n_jobs)
        df_merge = pd.concat([x['df'] for x in check_result_list])
        run_time_list = df_merge['_runtime'].tolist()
        avg_run_time = int(np.mean(run_time_list))
        std_run_time = int(np.std(run_time_list))
        return avg_run_time,std_run_time

    

    #修改wandb配置文件
    def generate_wandb(self, dataset_name, model_name, emb_type, fpath, ftarget, model_path):
        with open(fpath,"r") as fin,\
            open(ftarget,"w") as fout:
            data = yaml.load(fin, Loader=yaml.FullLoader)
            name = ftarget.split('_')
            data['name'] = '_'.join([dataset_name, model_name, emb_type, 'prediction'])
            data['parameters']['save_dir']['values'] = model_path
            data['parameters']['save_dir']['values'] = model_path
            yaml.dump(data, fout)

    def write_config(self, dataset_name, dconfig, CONFIG_FILE):
        with open(CONFIG_FILE) as fin:
            data_config = json.load(fin)
            data_config[dataset_name] = dconfig
        with open(CONFIG_FILE, "w") as fout:
            data = json.dumps(data_config, ensure_ascii=False, indent=4)
            fout.write(data)

    # # 生成启动sweep的脚本
    def generate_sweep(self, wandb_key, pred_dir, sweep_shell, ftarget, generate_all):
        # with open(wandb_path) as fin:
        #     wandb_config = json.load(fin)
        pre = "WANDB_API_KEY=" + wandb_key + " wandb sweep "
        with open(sweep_shell,"w") as fallsh:
            if generate_all:
                files = os.listdir(pred_dir)
                files = sorted(files)
                for f in files:
                    fpath = os.path.join(pred_dir, f)
                    fallsh.write(pre + fpath + " -p {}".format(self.project_name) + "\n")
            else:
                fallsh.write(pre + ftarget + " -p {}".format(self.project_name) + "\n")

    def extract_best_models(self, df, dataset_name, model_name, emb_type="qid", eval_test=True, fpath="./seedwandb/predict.yaml", CONFIG_FILE="../configs/best_model.json", wandb_key="", pred_dir="pred_wandbs", launch_file="start_predict.sh", generate_all=False):
        """extracting the best models which performance best performance on the validation data for testing 
        
        Args:
            df: dataframe of best results in each fold
            dataset_name: dataset_name
            model_name: model_name
            emb_type: embedding_type, default:qid
            eval_test: evaluating on testing set, default:True
            fpath: the yaml template for prediction in wandb, default: "./seedwandb/predict.yaml"
            config_file: the config template of generating prediction file, default: "../configs/best_model.json"
            wandb_key: the key of wandb account
            pred_wandbs: the directory of prediction yaml files, default: "pred_wandbs"
            launch_file: the launch file of starting the wandb prediction, default: "start_predict.sh"
            generate_all: starting all the files on the pred_wandbs directory or not, default:False
            
        Returns:
            the launch file (e.g., "start_predict.sh") for wandb prediction of the best models in each fold
        """
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        model_path_fold_first = []
        dconfig = dict()
        for i, row in df.iterrows():
            fold, model_path = row["fold"], row["model_save_path"]
            model_path = model_path.rstrip(f"{emb_type}_model.ckpt")
            print(f">>> The best model of {dataset_name}_{model_name}_{fold}:{model_path}")
            model_path_fold_first.append(model_path)
        ftarget = os.path.join(pred_dir, "{}_{}_{}_fold_first_predict.yaml".format(dataset_name, model_name, emb_type))
        if eval_test:
            self.generate_wandb(dataset_name, model_name, emb_type, fpath, ftarget, model_path_fold_first)
            dconfig["model_path_fold_first"] = model_path_fold_first
            self.write_config(dataset_name, dconfig, CONFIG_FILE)
            self.generate_sweep(wandb_key, pred_dir, launch_file, ftarget, generate_all)

    def extract_prediction_results(self, dataset_name, model_name, emb_type="qid", print_std=True):

        """calculating the results on the testing data in the best model in validation set.
        
        Args:
            dataset_name: dataset_name
            model_name: model_name
            emb_type: embedding_type, default:qid
            print_std: print the standard deviation results or not, default:True

        Returns:
            the average results of auc, acc in 5-folds and the corresponding standard deviation results
        """
        all_res = self.get_df('_'.join([dataset_name, model_name, emb_type, 'prediction']), input_type="sweep_name")
        all_res = all_res.drop_duplicates(["save_dir"])
        if len(all_res) < 5:
            print("Failure running exists, please check!!!")
            return
        repeated_aucs = np.unique(all_res["testauc"].values)
        repeated_accs = np.unique(all_res["testacc"].values)
        repeated_window_aucs = np.unique(all_res["window_testauc"].values)
        repeated_window_accs = np.unique(all_res["window_testacc"].values)
        repeated_auc_mean, repeated_auc_std = np.mean(repeated_aucs), np.std(repeated_aucs, ddof=0)
        repeated_acc_mean, repeated_acc_std = np.mean(repeated_accs), np.std(repeated_accs, ddof=0)
        repeated_winauc_mean, repeated_winauc_std = np.mean(repeated_window_aucs), np.std(repeated_window_aucs, ddof=0)
        repeated_winacc_mean, repeated_winacc_std = np.mean(repeated_window_accs), np.std(repeated_window_accs, ddof=0)
        key = dataset_name + "_" + model_name
        if print_std:
            print(key + "_repeated:", "%.4f"%repeated_auc_mean + "±" + "%.4f"%repeated_auc_std + "," + "%.4f"%repeated_acc_mean + "±" + "%.4f"%repeated_acc_std + "," + "%.4f"%repeated_winauc_mean + "±" + "%.4f"%repeated_winauc_std + "," + "%.4f"%repeated_winacc_mean + "±" + "%.4f"%repeated_winacc_std) 
        else:
            print(key + "_repeated:", "%.4f"%repeated_auc_mean + "," + "%.4f"%repeated_acc_mean + "," + "%.4f"%repeated_winauc_mean + "," + "%.4f"%repeated_winacc_mean)
        try:       
            question_aucs = np.unique(all_res["oriaucconcepts"].values)
            question_accs = np.unique(all_res["oriaccconcepts"].values)
            question_window_aucs = np.unique(all_res["windowaucconcepts"].values)
            question_window_accs = np.unique(all_res["windowaccconcepts"].values)
            question_auc_mean, question_auc_std = np.mean(question_aucs), np.std(question_aucs, ddof=0)
            question_acc_mean, question_acc_std = np.mean(question_accs), np.std(question_accs, ddof=0)
            question_winauc_mean, question_winauc_std = np.mean(question_window_aucs), np.std(question_window_aucs, ddof=0)
            question_winacc_mean, question_winacc_std = np.mean(question_window_accs), np.std(question_window_accs, ddof=0)
            key = dataset_name + "_" + model_name
            if print_std:
                print(key + "_concepts:", "%.4f"%question_auc_mean + "±" + "%.4f"%question_auc_std + "," + "%.4f"%question_acc_mean + "±" + "%.4f"%question_acc_std + "," + "%.4f"%question_winauc_mean + "±" + "%.4f"%question_winauc_std + "," + "%.4f"%question_winacc_mean + "±" + "%.4f"%question_winacc_std) 
            else:
                print(key + "_concepts:", "%.4f"%question_auc_mean + "," + "%.4f"%question_acc_mean + "," + "%.4f"%question_winauc_mean + "," + "%.4f"%question_winacc_mean) 
        except:
            print(f"{model_name} don't have question tag!!!")
            return

        try:
            early_aucs = np.unique(all_res["oriaucearly_preds"].values)
            early_accs = np.unique(all_res["oriaccearly_preds"].values)
            early_window_aucs = np.unique(all_res["windowaucearly_preds"].values)
            early_window_accs = np.unique(all_res["windowaccearly_preds"].values)
            early_auc_mean, early_auc_std = np.mean(early_aucs), np.std(early_aucs, ddof=0)
            early_acc_mean, early_acc_std = np.mean(early_accs), np.std(early_accs, ddof=0)
            early_winauc_mean, early_winauc_std = np.mean(early_window_aucs), np.std(early_window_aucs, ddof=0)
            early_winacc_mean, early_winacc_std = np.mean(early_window_accs), np.std(early_window_accs, ddof=0)
            key = dataset_name + "_" + model_name
            if print_std:
                print(key + "_early:", "%.4f"%early_auc_mean + "±" + "%.4f"%early_auc_std + "," + "%.4f"%early_acc_mean + "±" + "%.4f"%early_acc_std + "," + "%.4f"%early_winauc_mean + "±" + "%.4f"%early_winauc_std + "," + "%.4f"%early_winacc_mean + "±" + "%.4f"%early_winacc_std)
            else:
                print(key + "_early:", "%.4f"%early_auc_mean + "," + "%.4f"%early_acc_mean + "," + "%.4f"%early_winauc_mean + "," + "%.4f"%early_winacc_mean)         
        except:
            print(f"{model_name} don't have early fusion!!!")

        late_mean_aucs = np.unique(all_res["oriauclate_mean"].values)
        late_mean_accs = np.unique(all_res["oriacclate_mean"].values)
        late_mean_window_aucs = np.unique(all_res["windowauclate_mean"].values)
        late_mean_window_accs = np.unique(all_res["windowacclate_mean"].values)
        latemean_auc_mean, latemean_auc_std = np.mean(late_mean_aucs), np.std(late_mean_aucs, ddof=0)
        latemean_acc_mean, latemean_acc_std = np.mean(late_mean_accs), np.std(late_mean_accs, ddof=0)
        latemean_winauc_mean, latemean_winauc_std = np.mean(late_mean_window_aucs), np.std(late_mean_window_aucs, ddof=0)
        latemean_winacc_mean, latemean_winacc_std = np.mean(late_mean_window_accs), np.std(late_mean_window_accs, ddof=0)
        key = dataset_name + "_" + model_name
        if print_std:
            print(key + "_latemean:", "%.4f"%latemean_auc_mean + "±" + "%.4f"%latemean_auc_std + "," + "%.4f"%latemean_acc_mean + "±" + "%.4f"%latemean_acc_std + "," + "%.4f"%latemean_winauc_mean + "±" + "%.4f"%latemean_winauc_std + "," + "%.4f"%latemean_winacc_mean + "±" + "%.4f"%latemean_winacc_std)
        else:
            print(key + "_latemean:", "%.4f"%latemean_auc_mean + "," + "%.4f"%latemean_acc_mean + "," + "%.4f"%latemean_winauc_mean + "," + "%.4f"%latemean_winacc_mean)

        late_vote_aucs = np.unique(all_res["oriauclate_vote"].values)
        late_vote_accs = np.unique(all_res["oriacclate_vote"].values)
        late_vote_window_aucs = np.unique(all_res["windowauclate_vote"].values)
        late_vote_window_accs = np.unique(all_res["windowacclate_vote"].values)
        latevote_auc_mean, latevote_auc_std = np.mean(late_vote_aucs), np.std(late_vote_aucs, ddof=0)
        latevote_acc_mean, latevote_acc_std = np.mean(late_vote_accs), np.std(late_vote_accs, ddof=0)
        latevote_winauc_mean, latevote_winauc_std = np.mean(late_vote_window_aucs), np.std(late_vote_window_aucs, ddof=0)
        latevote_winacc_mean, latevote_winacc_std = np.mean(late_vote_window_accs), np.std(late_vote_window_accs, ddof=0)
        key = dataset_name + "_" + model_name
        if print_std:
            print(key + "_latevote:", "%.4f"%latevote_auc_mean + "±" + "%.4f"%latevote_auc_std + "," + "%.4f"%latevote_acc_mean + "±" + "%.4f"%latevote_acc_std + "," + "%.4f"%latevote_winauc_mean + "±" + "%.4f"%latevote_winauc_std + "," + "%.4f"%latevote_winacc_mean + "±" + "%.4f"%latevote_winacc_std)
        else:
            print(key + "_latevote:", "%.4f"%latevote_auc_mean + "," + "%.4f"%latevote_acc_mean + "," + "%.4f"%latevote_winauc_mean + "," + "%.4f"%latevote_winacc_mean)

        late_all_aucs = np.unique(all_res["oriauclate_all"].values)
        late_all_accs = np.unique(all_res["oriacclate_all"].values)
        late_all_window_aucs = np.unique(all_res["windowauclate_all"].values)
        late_all_window_accs = np.unique(all_res["windowacclate_all"].values)
        lateall_auc_mean, lateall_auc_std = np.mean(late_all_aucs), np.std(late_all_aucs, ddof=0)
        lateall_acc_mean, lateall_acc_std = np.mean(late_all_accs), np.std(late_all_accs, ddof=0)
        lateall_winauc_mean, lateall_winauc_std = np.mean(late_all_window_aucs), np.std(late_all_window_aucs, ddof=0)
        lateall_winacc_mean, lateall_winacc_std = np.mean(late_all_window_accs), np.std(late_all_window_accs, ddof=0)
        key = dataset_name + "_" + model_name
        if print_std:
            print(key + "_lateall:", "%.4f"%lateall_auc_mean + "±" + "%.4f"%lateall_auc_std + "," + "%.4f"%lateall_acc_mean + "±" + "%.4f"%lateall_acc_std + "," + "%.4f"%lateall_winauc_mean + "±" + "%.4f"%lateall_winauc_std + "," + "%.4f"%lateall_winacc_mean + "±" + "%.4f"%lateall_winacc_std)
        else:
            print(key + "_lateall:", "%.4f"%lateall_auc_mean + "," + "%.4f"%lateall_acc_mean + "," + "%.4f"%lateall_winauc_mean + "," + "%.4f"%lateall_winacc_mean)