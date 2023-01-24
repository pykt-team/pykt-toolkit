import os
import json
from pickle import NONE
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm_notebook
import argparse
from pykt.config import que_type_models
from extract_quelevel_raw_result import get_one_result_help as get_quelevel_one_result_help
import traceback


cut = True

def eval_line(line):
    line0 = [int(x) for x in line.split("],")[0].strip(" [").split(",")]
    line1 = [int(x) for x in line.split("],")[1].strip(" [").split(",")]
    line2 = [int(x) for x in line.split("],")[2].strip(" [").split(",")]
    line3 = [int(x) for x in line.split("],")[3].strip(" [").split(",")]
    line4 = [float(x) for x in line.split("],")[4].strip(" [").split(",")]
    return [line0, line1, line2, line3, line4]


def parse_raw_file(file_path):
    return [eval_line(x) for x in open(file_path).readlines()]


def paser_raw_data_core(pred_list, df,stu_inter_num_dict):
    assert len(pred_list) == df.shape[0]
    # 汇总原始预测结果，并给统计结果上增加uid
    y_pred = []
    y_true = []
    uid_list = []
    cidx_list = []
    c_list = []
    
    for line, uid, cisx,c in zip(pred_list, df['uid'], df['cidxs'],df['concepts']):
        y_pred.extend(line[4])
        y_true.extend(line[3])
        cidx = list(filter(lambda x: x != '-1', cisx.split(",")[1:]))
        c = list(filter(lambda x: x != '-1', c.split(",")[1:]))
        if len(line[1])==199 and len(line[3])==1:
            cidx_list.extend(cidx[-1:])
            c_list.extend(c[-1:])
        else:
            cidx_list.extend(cidx)
            c_list.extend(c)
        uid_list.extend([uid]*len(line[3]))
#     print(len(y_pred), len(y_true), len(cidx_list), len(uid_list))
    df_result = pd.DataFrame(
        {"y_pred": y_pred, 'y_true': y_true, 'uid': uid_list, "cidx": cidx_list,"concepts":c_list})
    df_result['inter_num'] = df_result['uid'].map(stu_inter_num_dict)
    return df_result

def paser_raw_data(pred_list, df, stu_inter_num_dict, min_len=200):
    df_result = paser_raw_data_core(pred_list, df,stu_inter_num_dict)
    df_short = df_result[df_result['inter_num'] <= min_len].copy()
    df_long = df_result[df_result['inter_num'] > min_len].copy()
    return df_long, df_short

def save_df(df,name,save_dir):
    if not df is None:
        print(f"save {name} to {save_dir}")
        df.to_csv(os.path.join(save_dir,f"{name}_new.csv"),index=False)
    else:
        print(f"skip {name} to {save_dir}")
        
def get_metrics(df, y_pred_col='y_pred', y_true_col='y_true', name="test", cut=False,save_dir=None):
    """获取原始指标"""
    if not save_dir is None:
        save_df(df,name,save_dir)
    if len(df)==0:
        return {}
    print(f"get_metrics,y_pred_col={y_pred_col},name={name}")
    # 针对concept_preds
    if y_pred_col == "concept_preds":
        x_pred = []
        x_true = []
        for pred, true in zip(df[y_pred_col], df[y_true_col]):
            pred_split = pred.split(',')
            x_pred.extend(pred_split)
            x_true.extend([true]*len(pred_split))
        y_pred = np.array(x_pred, dtype=np.float)
        y_true = np.array(x_true)
    else:
        y_pred = df[y_pred_col]
        y_true = df[y_true_col]
    acc = accuracy_score(y_true, y_pred > 0.5)
    auc = roc_auc_score(y_true, y_pred)
    if cut:
        acc = round(acc, 4)
        auc = round(auc, 4)
    report = {f"{name}_acc": acc, f"{name}_auc": auc}
    return report

# load data


def get_stu_inter_map(data_dir):
    df_raw_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    # 计算每个学生的交互数量
    stu_inter_num_dict = {}
    for _, row in df_raw_test.iterrows():
        if 'is_repeat' not in df_raw_test.columns:
            num = len(row['concepts'].split(","))
        else:
            num = row['is_repeat'].count("0")
        stu_inter_num_dict[row['uid']] = num
    return stu_inter_num_dict


def load_all_raw_df(data_dir):
    df_raw_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    try:
        df_qid_test = pd.read_csv(os.path.join(data_dir, "test_sequences.csv"))
    except:
        df_qid_test = None
    try:
        df_qid_test_win = pd.read_csv(os.path.join(
        data_dir, "test_window_sequences.csv"))
    except:
        df_qid_test_win = None
    que_path = os.path.join(data_dir, "test_question_sequences.csv")
    if os.path.exists(que_path):
        df_que_test = pd.read_csv(que_path)
        df_que_win_test = pd.read_csv(os.path.join(
            data_dir, "test_question_window_sequences.csv"))
    else:
        df_que_test = None
        df_que_win_test = None
    return {"df_raw_test": df_raw_test, "df_qid_test": df_qid_test, "df_qid_test_win": df_qid_test_win,
            "df_que_test": df_que_test, "df_que_win_test": df_que_win_test}


# concept

def add_concepts(save_dir, data_dir, report, stu_inter_num_dict, cut, data_dict):
    config = json.load(open(os.path.join(save_dir,"config.json")))
    emb_type = config['params']['emb_type']
    # 非window
    qid_test = parse_raw_file(os.path.join(save_dir, f"{emb_type}_test_predictions.txt"))
    df_qid_test = data_dict['df_qid_test']
    # split
    df_qid_long, df_qid_short = paser_raw_data(
        qid_test, df_qid_test, stu_inter_num_dict=stu_inter_num_dict)
    # get report
    if len(df_qid_long)!=0:
        base_long_report = get_metrics(df_qid_long, cut=cut, name='long',save_dir=save_dir)
        report.update(base_long_report)
    
    if len(df_qid_short)!=0:
        base_short_report = get_metrics(df_qid_short, cut=cut, name='short',save_dir=save_dir)
        report.update(base_short_report)
   
    
    df_qid = pd.concat([df_qid_long, df_qid_short])
    base_report = get_metrics(df_qid, cut=cut, name='test',save_dir=save_dir)
    report.update(base_report)

    
    return df_qid,df_qid_long,df_qid_short


def add_concept_win(save_dir, data_dir, report, stu_inter_num_dict, cut, data_dict):
    config = json.load(open(os.path.join(save_dir,"config.json")))
    emb_type = config['params']['emb_type']

    qid_test_win = parse_raw_file(os.path.join(save_dir, f"{emb_type}_test_window_predictions.txt"))
    df_qid_test_win = data_dict['df_qid_test_win']
    df_qid_win_long, df_qid_win_short = paser_raw_data(qid_test_win, df_qid_test_win, stu_inter_num_dict)

    if len(df_qid_win_long)!=0:
        base_win_long_report = get_metrics(df_qid_win_long, cut=cut, name='win_long',save_dir=save_dir)
        report.update(base_win_long_report)

    
    if len(df_qid_win_short)!=0:
        base_win_short_report = get_metrics(df_qid_win_short, cut=cut, name='win_short',save_dir=save_dir)
        report.update(base_win_short_report)

    
    df_qid_win = pd.concat([df_qid_win_long, df_qid_win_short])
    base_report = get_metrics(df_qid_win, cut=cut, name='test_win',save_dir=save_dir)
    report.update(base_report)

    
    
    
    return df_qid_win,df_qid_win_long,df_qid_win_short

def concept_update_l2(save_dir, data_dir, report, stu_inter_num_dict, cut, data_dict):
    config = json.load(open(os.path.join(save_dir,"config.json")))
    emb_type = config['params']['emb_type']

    # 非window
    qid_test = parse_raw_file(os.path.join(save_dir, f"{emb_type}_test_predictions.txt"))
    df_qid_test = data_dict['df_qid_test']
    # window
    qid_test_win = parse_raw_file(os.path.join(save_dir, f"{emb_type}_test_window_predictions.txt"))
    df_qid_test_win = data_dict['df_qid_test_win']
    # 获取原始概率
    df_result = paser_raw_data_core(qid_test, df_qid_test,stu_inter_num_dict)
    df_result_win = paser_raw_data_core(qid_test_win, df_qid_test_win,stu_inter_num_dict)
    
    # 获取保留的题目
    keep_cidx = []
    for uid, group in df_result.groupby("uid"):
        inter_num = group.iloc[0]['inter_num']
        if inter_num < 200:
            continue
        keep_cidx.extend(group[200:]['cidx'].tolist())
        
    # 筛选需要计算的指标
    df_result_long = df_result[df_result['cidx'].isin(keep_cidx)].copy()
    if len(df_result_long)!=0:
        report.update(get_metrics(df_result_long, cut=cut, name='b200',save_dir=save_dir))
    
    df_result_win_long = df_result_win[df_result_win['cidx'].isin(keep_cidx)].copy()
    if len(df_result_win_long)!=0:
        report.update(get_metrics(df_result_win_long, cut=cut, name='win_b200',save_dir=save_dir))
    
    
    # 小于200的指标
    df_result_short = df_result[~df_result['cidx'].isin(keep_cidx)].copy()
    if len(df_result_short)!=0:
        report.update(get_metrics(df_result_short, cut=cut, name='s200',save_dir=save_dir))
        
    df_result_win_short = df_result_win[~df_result_win['cidx'].isin(keep_cidx)].copy()
    if len(df_result_short)!=0:
        report.update(get_metrics(df_result_win_short, cut=cut, name='win_s200',save_dir=save_dir))
    

# question

def update_question_df(df):
    df = df.copy()
    have_early = "early_preds" in df.columns

    t_label, vp_preds, ap_preds, ep_preds, lmp_preds = [], [], [], [], []
    line = 0
    for i, row in df.iterrows():
        cpreds = np.array([float(p) for p in row["concept_preds"].split(",")])
        # vp, ap = float(row["late_vote"]), float(row["late_all"])
        high, low = [], []
        for pred in cpreds:
            if pred >= 0.5:
                high.append(pred)
            else:
                low.append(pred)
        correctnum = list(cpreds>=0.5).count(True)

        vp_probility = np.mean(high) if correctnum / len(cpreds) >= 0.5 else np.mean(low)
        ap_probility = np.mean(high) if correctnum == len(cpreds) else np.mean(low)
        if have_early:
            ep_probility = float(row["early_preds"])
        else:
            ep_probility = 0
        lmp_probility = float(row["late_mean"])

        t_label.append(int(row["late_trues"]))

        vp_preds.append(vp_probility)#late vote
        ap_preds.append(ap_probility)#late all
        lmp_preds.append(lmp_probility)#late_mean
        ep_preds.append(ep_probility)#early 

    late_vote, late_all = np.array(vp_preds), np.array(ap_preds)  # ,vote,all
    early_preds, late_mean = np.array(
        ep_preds), np.array(lmp_preds)  # ,early,lmean
#     print(late_vote,late_all,early_preds,late_mean)

    for col in ['late_mean', 'late_vote', 'late_all', 'early_preds']:
        if col not in df.columns:
            continue
        else:
            df[col] = eval(col)
    return df


def que_update_ls_report(que_test, que_win_test, report,save_dir):
    # split
    if len(que_test)!=0:
        que_short = que_test[que_test['inter_num'] <= 200].reset_index()
        que_long = que_test[que_test['inter_num'] > 200].reset_index()
    else:
        que_short = pd.DataFrame()
        que_long = pd.DataFrame()
   
    if len(que_win_test)!=0:
        que_win_short = que_win_test[que_win_test['inter_num']
                                    <= 200].reset_index()
        que_win_long = que_win_test[que_win_test['inter_num'] > 200].reset_index()
    else:
        que_win_short = pd.DataFrame()
        que_win_long = pd.DataFrame()

    
    # update
    for y_pred_col in ['concept_preds', 'late_mean', 'late_vote', 'late_all', 'early_preds']:
        print(f"que_update_ls_report start {y_pred_col}")
        if len(que_test)!=0 and y_pred_col not in que_test.columns:
            print(f"skip {y_pred_col}")
            continue
        # long
        if len(que_long)!=0:
            report_long = get_metrics(que_long, y_true_col="late_trues",
                                    y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_long',save_dir=save_dir)
            report.update(report_long)

        # short
        if len(que_short)!=0:
            report_short = get_metrics(que_short, y_true_col="late_trues",
                                    y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_short',save_dir=save_dir)
            report.update(report_short)

        # long + short
        report_col = get_metrics(que_test, y_true_col="late_trues",
                                 y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}',save_dir=save_dir)
        report.update(report_col)

        # win long
        if len(que_win_long)!=0:
            report_win_long = get_metrics(que_win_long, y_true_col="late_trues",
                                        y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_win_long',save_dir=save_dir)
            report.update(report_win_long)

        # win short
        if len(que_win_short)!=0:
            report_win_short = get_metrics(que_win_short, y_true_col="late_trues",
                                        y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_win_short',save_dir=save_dir)
            report.update(report_win_short)
        
        # win long + win short
        report_win = get_metrics(que_win_test, y_true_col="late_trues",
                                 y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_win',save_dir=save_dir)
        report.update(report_win)
    return que_short,que_long,que_win_short,que_win_long




def que_update_l2(que_test, que_win_test, report,save_dir):
    keep_qidx = []
    for uid, group in que_test.groupby("uid"):
        inter_num = group.iloc[0]['inter_num']
        if inter_num < 200:
            continue
        keep_qidx.extend(group[200:]['qidx'].tolist())

    que_test_b_200 = que_test[que_test['qidx'].isin(keep_qidx)].copy()
    que_win_test_b_200 = que_win_test[que_win_test['qidx'].isin(keep_qidx)].copy()

    que_test_s_200 = que_test[~que_test['qidx'].isin(keep_qidx)].copy()
    que_win_test_s_200 = que_win_test[~que_win_test['qidx'].isin(keep_qidx)].copy()

    # update
    for y_pred_col in ['concept_preds', 'late_mean', 'late_vote', 'late_all', 'early_preds']:
        if y_pred_col not in que_test.columns:
            print(f"skip {y_pred_col}")
            continue
        # ori(>200)
        report.update(get_metrics(que_test_b_200, y_true_col="late_trues",
                                 y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_b200',save_dir=save_dir))
        # win(>200)
        report.update(get_metrics(que_win_test_b_200, y_true_col="late_trues",
                                 y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_win_b200',save_dir=save_dir))

        # ori(<200)
        report.update(get_metrics(que_test_s_200, y_true_col="late_trues",
                                 y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_s200',save_dir=save_dir))
        # win(<200)
        report.update(get_metrics(que_win_test_s_200, y_true_col="late_trues",
                                 y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_win_s200',save_dir=save_dir))
    return que_test_b_200,que_win_test_b_200,que_test_s_200,que_win_test_s_200



def add_question_report(save_dir, data_dir, report, stu_inter_num_dict, cut, data_dict):
    config = json.load(open(os.path.join(save_dir,"config.json")))
    emb_type = config['params']['emb_type']
    
    df_que_test = data_dict['df_que_test']
    # 映射学生
    orirow_2_uid = {}
    for _, row in df_que_test.iterrows():
        orirow_2_uid[int(row['orirow'].split(',')[0])] = row['uid']
   
    try:
        que_test = pd.read_csv(os.path.join(
            save_dir, f"{emb_type}_test_question_predictions.txt"), sep='\t')
        que_test = update_question_df(que_test)
        # map
        que_test['uid'] = que_test['orirow'].map(orirow_2_uid)
        que_test['inter_num'] = que_test['uid'].map(stu_inter_num_dict)
        save_df(que_test,'que_test',save_dir)
    except:
        que_test = pd.DataFrame()

   
    try:
        que_win_test = pd.read_csv(os.path.join(
            save_dir, f"{emb_type}_test_question_window_predictions.txt"), sep='\t')
        que_win_test = update_question_df(que_win_test)
        df_que_win_test = data_dict['df_que_win_test']

        que_win_test['uid'] = que_win_test['orirow'].map(orirow_2_uid)
        que_win_test['inter_num'] = que_win_test['uid'].map(stu_inter_num_dict)
        save_df(que_win_test,'que_win_test',save_dir)
    except:
        que_win_test = pd.DataFrame()

    # print("Start 基于题目的长短序列")
    try:
        print("Start 基于题目的长短序列")
        que_update_ls_report(que_test, que_win_test, report,save_dir=save_dir)  # short long 结果
    except:
        print("Fail 基于题目的长短序列")

    # print("Start 基于题目的>200部分")
    
    try:
        print("Start 基于题目的>200部分")
        que_update_l2(que_test, que_win_test, report,save_dir=save_dir)  # 大于200部分的结果
    except:
        print("Start 基于题目的>200部分")
         
   
    que_test = None
    return que_test,que_win_test



def get_one_result(root_save_dir, stu_inter_num_dict, data_dict, cut, skip=False, dataset="", model_name="", data_dir=""):
    # 存在则不跑了知识点长短序列
    report_path = os.path.join(root_save_dir, 'report.csv')
    if skip and os.path.exists(report_path):
        df_report = pd.read_csv(report_path)
        if df_report.shape[1] < 3:
            print(f"skip all {dataset}-{model_name},details: 已经跑过了")
    else:
        # 开始跑
        report_list = []
        for fold_name in os.listdir(root_save_dir):
            if "report" in fold_name or fold_name[0] == '.' or '_bak' in fold_name or "nohup.out" in fold_name:
                continue
            save_dir = os.path.join(root_save_dir, fold_name)
            report = {"save_dir": save_dir, "data_dir": data_dir, "cut": cut}

            #基于知识点
            try:
                print("Start 知识点非win")
                add_concepts(save_dir, data_dir, report,stu_inter_num_dict, cut, data_dict)
            except:
                print(f"Fail 知识点非win,details is {traceback.format_exc()}")

            #知识点win
            try:
                print("Start 知识点win")
                add_concept_win(save_dir, data_dir, report,stu_inter_num_dict, cut, data_dict)
            except:
                print(f"Fail 知识点win,details is {traceback.format_exc()}")

            #长短序列
            try:
                print("Start 知识点长短序列")
                concept_update_l2(save_dir, data_dir, report, stu_inter_num_dict, cut, data_dict)
            except:
                print(f"Fail 知识点长短序列,details is {traceback.format_exc()}")

            
            if not data_dict['df_que_test'] is None:
                #基于题目的,这个函数内部有更多保存
                print("Start 基于题目的")
                add_question_report(save_dir, data_dir, report,stu_inter_num_dict, cut, data_dict)
            print(f"report is {report}")
            report_list.append(report)
        df_report = pd.DataFrame(report_list)
        df_report.to_csv(report_path, index=False)
    return df_report


def run_all():
    dataset_list = ['assist2015', 'assist2009', 'algebra2005',
                    'bridge2algebra2006', 'statics2011', 'poj', 'nips_task34']
    model_list = ['dkt', 'dkvmn', 'dkt+', 'kqn',
                  'sakt', 'dkt_forget', 'saint', 'akt', 'atkt']
    for dataset in tqdm_notebook(dataset_list):
        print(f"start {dataset}")
        data_dir = os.path.join(data_root_dir, dataset)
        stu_inter_num_dict = get_stu_inter_map(data_dir)
        data_dict = load_all_raw_df(data_dir)  # 加载数据集切分时的数据
        for model_name in model_list:
            root_save_dir = os.path.join(model_root_dir, dataset, model_name)
            # 检查
            if not os.path.exists(root_save_dir):
                print(f"skip {dataset}-{model_name},details: 文件夹不存在")
                continue
            get_one_result(root_save_dir, stu_inter_num_dict,
                           data_dict, cut, skip=False, dataset=dataset, model_name=model_name, data_dir=data_dir)
            break
        break


def get_one_result_help(dataset, model_name,model_root_dir,data_root_dir):
    data_dir = os.path.join(data_root_dir, dataset)
    
    stu_inter_num_dict = get_stu_inter_map(data_dir)
    print("Start 载入原始数据")
    data_dict = load_all_raw_df(data_dir)  # 加载数据集切分时的数据
    root_save_dir = os.path.join(model_root_dir, dataset, model_name)
    # 检查
    if not os.path.exists(root_save_dir):
        print(f"skip {dataset}-{model_name},details: 文件夹不存在")
    else:
        get_one_result(root_save_dir, stu_inter_num_dict,
                       data_dict, cut, skip=False, dataset=dataset, model_name=model_name, data_dir=data_dir)


if __name__ == "__main__":
    # model_root_dir = "/root/autodl-nas/liuqiongqiong/bakt/pykt-toolkit/examples/best_model_path"
    # model_root_dir = "/root/autodl-nas/project/pykt_nips2022/examples/best_model_path"
    model_root_dir = "/root/autodl-nas/project/full_result_pykt/best_model_path"
    # model_root_dir = "/root/autodl-nas/project/pykt_qikt/examples/best_model_path"
    data_root_dir = '/root/autodl-nas/project/pykt_nips2022/data'
    

    import wandb
    wandb.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="akt")
    args = parser.parse_args()
    if args.model_name in que_type_models:
        get_quelevel_one_result_help(args.dataset, args.model_name,model_root_dir,data_root_dir)#for question level
    else:
        get_one_result_help(args.dataset, args.model_name,model_root_dir,data_root_dir)
    wandb.log({"dataset": args.dataset, "model_name": args.model_name})
    # python extract_raw_result.py --dataset {dataset} --model_name {model_name}
    #wandb sweep seedwandb/extract_raw.yaml
