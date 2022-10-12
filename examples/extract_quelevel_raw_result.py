import os
from pickle import NONE
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm_notebook
import argparse

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
    for line, uid in zip(pred_list, df['uid']):
        y_pred.extend(line[4])
        y_true.extend(line[3])
        # cidx = list(filter(lambda x: x != '-1', cisx.split(",")[1:]))
        # if len(line[1])==199 and len(line[3])==1:
            # cidx_list.extend(cidx[-1:])
        # else:
            # cidx_list.extend(cidx)
        uid_list.extend([uid]*len(line[3]))
#     print(len(y_pred), len(y_true), len(cidx_list), len(uid_list))
    df_result = pd.DataFrame(
        {"y_pred": y_pred, 'y_true': y_true, 'uid': uid_list})
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
    que_path = os.path.join(data_dir, "test_sequences_quelevel.csv")
   
    df_que_test = pd.read_csv(que_path)
    df_que_win_test = pd.read_csv(os.path.join(
        data_dir, "test_window_sequences_quelevel.csv"))
  
    return {"df_raw_test": df_raw_test,
            "df_que_test": df_que_test, "df_que_win_test": df_que_win_test}


def update_question_df(df):
    df = df.copy()
    have_early = "early_preds" in df.columns

    t_label, vp_preds, ap_preds, ep_preds, lmp_preds = [], [], [], [], []
    line = 0
    for i, row in df.iterrows():
        cpreds = [float(p) for p in row["concept_preds"].split(",")]
        vp, ap = float(row["late_vote"]), float(row["late_all"])
        lowl, highl = [], []
        for p in cpreds:
            if p >= 0.5:
                highl.append(p)
            else:
                lowl.append(p)
        low = np.mean(lowl) if len(lowl) > 0 else 0.4999

        vp_probility = np.mean(highl) if int(vp) == 1 else low
        ap_probility = np.mean(highl) if int(ap) == 1 else low
        if have_early:
            ep_probility = float(row["early_preds"])
        else:
            ep_probility = 0
        lmp_probility = float(row["late_mean"])
        cvp = 1 if vp_probility >= 0.5 else 0
        if cvp != int(vp):
            print(f"line: {line} cvp: {cvp}, vp: {vp}")
        t_label.append(int(row["late_trues"]))

        vp_preds.append(vp_probility)
        ap_preds.append(ap_probility)
        ep_preds.append(ep_probility)
        lmp_preds.append(lmp_probility)

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
    que_short = que_test[que_test['inter_num'] <= 200].reset_index()
    que_long = que_test[que_test['inter_num'] > 200].reset_index()

    que_win_short = que_win_test[que_win_test['inter_num']
                                 <= 200].reset_index()
    que_win_long = que_win_test[que_win_test['inter_num'] > 200].reset_index()

    
    # update
    for y_pred_col in ['concept_preds', 'late_mean', 'late_vote', 'late_all', 'early_preds']:
        print(f"que_update_ls_report start {y_pred_col}")
        if y_pred_col not in que_test.columns:
            print(f"skip {y_pred_col}")
            continue
        # long
        report_long = get_metrics(que_long, y_true_col="late_trues",
                                  y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_long',save_dir=save_dir)
        report.update(report_long)

        # short
        report_short = get_metrics(que_short, y_true_col="late_trues",
                                   y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_short',save_dir=save_dir)
        report.update(report_short)

        # long + short
        report_col = get_metrics(que_test, y_true_col="late_trues",
                                 y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}',save_dir=save_dir)
        report.update(report_col)

        # win long
        report_win_long = get_metrics(que_win_long, y_true_col="late_trues",
                                      y_pred_col=y_pred_col, cut=cut, name=f'{y_pred_col}_win_long',save_dir=save_dir)
        report.update(report_win_long)

        # win short
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
    que_test = pd.read_csv(os.path.join(
        save_dir, "qid_test_predictions.txt"), sep='\t')
    que_test = update_question_df(que_test)

    df_que_test = data_dict['df_que_test']

    que_win_test = pd.read_csv(os.path.join(
        save_dir, "qid_test_window_predictions.txt"), sep='\t')
    que_win_test = update_question_df(que_win_test)

    # df_que_win_test = data_dict['df_que_win_test']

    # 映射学生
    orirow_2_uid = {}
    for _, row in df_que_test.iterrows():
        orirow_2_uid[int(row['orirow'].split(',')[0])] = row['uid']

    # map
    que_test['uid'] = que_test['orirow'].map(orirow_2_uid)
    que_test['inter_num'] = que_test['uid'].map(stu_inter_num_dict)

    que_win_test['uid'] = que_win_test['orirow'].map(orirow_2_uid)
    que_win_test['inter_num'] = que_win_test['uid'].map(stu_inter_num_dict)

    # print("Start 基于题目的长短序列")
    # que_update_ls_report(que_test, que_win_test, report,save_dir=save_dir)  # short long 结果

    # print("Start 基于题目的>200部分")
    # que_update_l2(que_test, que_win_test, report,save_dir=save_dir)  # 大于200部分的结果
    save_df(que_test,'que_test',save_dir)
    save_df(que_win_test,'que_win_test',save_dir)
    return que_test,que_win_test



def get_one_result(root_save_dir, stu_inter_num_dict, data_dict, cut, skip=False, dataset="", model_name="", data_dir=""):
    # 存在则不跑了
    report_path = os.path.join(root_save_dir, 'report.csv')
    if skip and os.path.exists(report_path):
        df_report = pd.read_csv(report_path)
        if df_report.shape[1] < 3:
            print(f"skip all {dataset}-{model_name},details: 已经跑过了")
    else:
        # 开始跑
        report_list = []
        for fold_name in os.listdir(root_save_dir):
            if "report" in fold_name or fold_name[0] == '.' or 'bak' in fold_name:
                continue
            save_dir = os.path.join(root_save_dir, fold_name)
            report = {"save_dir": save_dir, "data_dir": data_dir, "cut": cut}
            
            if not data_dict['df_que_test'] is None:
                #基于题目的,这个函数内部有更多保存
                print("Start 基于题目的")
                add_question_report(save_dir, data_dir, report,stu_inter_num_dict, cut, data_dict)
            print(f"report is {report}")
            report_list.append(report)
        df_report = pd.DataFrame(report_list)
        df_report.to_csv(report_path, index=False)
    return df_report


def get_one_result_help(dataset, model_name):
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
    model_root_dir = "/root/autodl-nas/project/pykt_nips2022/examples/best_model_path_1002"
    data_root_dir = '/root/autodl-nas/project/pykt_nips2022/data'
    

    import wandb
    wandb.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="iekt")
    args = parser.parse_args()
    get_one_result_help(args.dataset, args.model_name)
    wandb.log({"dataset": args.dataset, "model_name": args.model_name})
    # python extract_raw_result.py --dataset {dataset} --model_name {model_name}
