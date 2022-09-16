import os
import wandb
import pandas as pd
from wandb.apis.public import gql


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
        self.api = wandb.Api()
        self.project = self.api.project(name=self.project_name)
        self.sweep_dict = self.get_sweep_dict()
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
               
        for name in sweep_dict:
            if len(sweep_dict[name]) > 1:
                del sweep_dict[name]
                print(f"Error!! we can not process the same sweep name {name}, we will not return those sweeps:{sweep_dict[name]}")
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

    def get_multi_df(self,id_list=[],input_type="sweep_name"):
        """Get multi sweep result

        Args:
            id_list (list): the list of sweep name or sweep id.
            input_type (str, optional): the type of id. Defaults to sweep_name.

        Returns:
            _type_: _description_
        """
        df_list = []
        for id in id_list:
            df = self.get_df(id,input_type=input_type)
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
        df = None
        report = {"stop_cmd":""}
        if sweep_status in ['CANCELED','FINISHED'] and not force_check_df:
            report['state'] = True
            report['num_run'] = -1
        else:
            num_run = self.get_sweep_run_num(sweep_id,input_type="sweep_id")#get sweep run num
            report['num_run'] = num_run
            if num_run<min_run_num:
                report['state'] = False
            else:
                #
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
                    df = df[:i].copy()
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

    
    def check_sweep_list(self,sweep_key_list,metric="validauc",metric_type="max",min_run_num=200,patience=50,force_check_df=False,stop=False):
        check_result_list = []
        for sweep_name in sweep_key_list:
            check_result = self.check_sweep_early_stop(sweep_name,input_type='sweep_name',
                    metric=metric,metric_type=metric_type,min_run_num=min_run_num,patience=patience,force_check_df=force_check_df)
            check_result['sweep_name'] = sweep_name
            check_result_list.append(check_result)

        if stop:#stop sweep
            for result in check_result_list:
                if result['State'] and result['stop_cmd']!=0:
                    self.stop_sweep(result['stop_cmd'])
        return check_result_list

    def check_sweep_by_pattern(self,sweep_pattern,metric="validauc",metric_type="max",min_run_num=200,patience=50,force_check_df=False,stop=False):
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
        check_result_list = self.check_sweep_list(sweep_key_list,metric=metric,metric_type=metric_type,min_run_num=min_run_num,patience=patience,force_check_df=force_check_df,stop=stop)
        
        return check_result_list

    def get_all_fold_name(self,model_name,dataset_name,emb_type="qid"):
        sweep_key_list = [f"{dataset_name}_{model_name}_{emb_type}_{fold}" for fold in range(5)]
        sweep_key_list = [x for x in sweep_key_list if x in self.sweep_keys]#filter error
        return sweep_key_list

    def check_sweep_by_model_dataset_name(self,model_name,dataset_name,emb_type="qid",metric="validauc",metric_type="max",min_run_num=200,patience=50,force_check_df=False,stop=False):
        sweep_key_list = self.get_all_fold_name(model_name,dataset_name,emb_type)
        if len(sweep_key_list)!=5:
            print("Input error, please check")
            return 
        check_result_list = self.check_sweep_list(sweep_key_list,metric=metric,metric_type=metric_type,min_run_num=min_run_num,patience=patience,force_check_df=force_check_df,stop=stop)
        return check_result_list

    def get_best_run(self,model_name,dataset_name,emb_type="qid",metric="validauc",metric_type="max",min_run_num=200,patience=50):
        # sweep_key_list = self.get_all_fold_name(model_name,dataset_name,emb_type)
        check_result_list = self.check_sweep_by_model_dataset_name(model_name,dataset_name,emb_type,metric=metric,metric_type=metric_type,min_run_num=min_run_num,patience=patience,force_check_df=True)
        row_list = []
        for result in check_result_list:
            df = result['df']
            row_list.append(df.iloc[result['first_best_index']])
        df = pd.DataFrame(row_list)
        return df
