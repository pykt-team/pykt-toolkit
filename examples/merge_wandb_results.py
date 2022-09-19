#!/usr/bin/env python
# coding=utf-8

from logging.config import dictConfig
import wandb
import pandas as pd
import numpy as np
# from tqdm import tqdm_notebook

import wandb 
import os, sys
import json
import socket
import yaml
import argparse

with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)

uid = wandb_config["uid"]
os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
CONFIG_FILE = "../configs/best_model.json"

def str2bool(str):
    return True if str.lower() == "true" else False

def get_runs_result(runs):
    result_list = []
    for run in runs:#tqdm_notebook(runs):
        result = {}
        result.update(run.summary._json_dict)
        model_config = {k: v for k, v in run.config.items()
                    if not k.startswith('_') and type(v) not in [list,dict]}
        result.update(model_config)
        result['Name'] = run.name
        result_list.append(result)
    runs_df = pd.DataFrame(result_list)#.drop_duplicates(list(model_config.keys()))
    return runs_df

def get_df(sweep_name,sweep_dict, project_name):
    df_list = []
    for sweep_id in sweep_dict[sweep_name]:
        sweep = api.sweep("{}/{}/{}".format(uid, project_name,sweep_id))
        df = get_runs_result(sweep.runs)
        df_list.append(df)
    df = pd.concat(df_list)
#     print(df.shape)
    return df

def get_sweep_dict(project):
    '''获取sweep的id'''
    sweep_dict = {}
    for sweep in project.sweeps():
        if sweep.name not in sweep_dict:
            sweep_dict[sweep.name] = []
        sweep_dict[sweep.name].append(sweep.id)
    return sweep_dict

def downloads(project_name, sweep_dict, dataset_name="", model_name="", emb_type=""):
    # print(f"project_name:{project_name}, dataset_name:{dataset_name}, model_name:{model_name}")
    all_res = dict()
    for key in sweep_dict:
        # emb_type = key.split("_")[-2]     
        if dataset_name != "" and model_name != "" and (key.find(dataset_name+"_") == -1 or key.find("_"+model_name+"_"+emb_type) == -1):
            continue
        try:
            df = get_df(key,sweep_dict,project_name)
            df = df.dropna(subset=["validauc"])
            print(f"key: {key}, df dropna: {df.shape}")
            tmps = key.split("_")
            key = "_".join(tmps[0:-1])
            all_res.setdefault(key, dict())
            fold = tmps[-1]
            print(f"key: {key}, fold: {fold}, df: {df.shape}")
            all_res[key][fold] = df
        except:
            print(f"error: {key}")
            continue
    return all_res

def read_cur_res(dfs, params_dir, key):
    print(f"dfs: {len(dfs)}")
    dfold = dict()
    all_df = pd.concat(dfs, axis=0)
    for fold in dfs: # all fold
        df = dfs[fold]
        df["Name"] = df["Name"].apply(lambda a: int(a.split("-")[-1]))
        df = df.sort_values(by=["Name"])
        for i, row in df.iterrows():
            fold, model_path = row["fold"], row["model_save_path"]
            model_path = model_path.rstrip("/qid_model.ckpt")
            dfold.setdefault(fold, dict())
            dfold[fold].setdefault(model_path, [])
            values = [row["validauc"], row["validacc"]]
            dfold[fold][model_path] = values
        # print(f"all_df: {all_df.shape}, df: {df.shape}, len(pams): {len(pams)}")
    return dfold

def get_results(d):
    aucs, accs = [], []
    num = 0
    
    best_model = []
    for fold in d:
        best_params = ""
        best_seed = ""
        best_name = -1
        auc, acc = 0.0, 0.0
        for model_path in d[fold]:
            num += 1
            curauc, curacc = d[fold][model_path]
            if curauc > auc:
                auc, acc, best_name = curauc, curacc, model_path

        aucs.append(auc)
        accs.append(acc)
        best_model.append(best_name)
                
    auc_mean, auc_std = round(np.mean(aucs), 4), round(np.std(aucs, ddof=0), 4)
    acc_mean, acc_std = round(np.mean(accs), 4), round(np.std(accs, ddof=0), 4)

    return auc_mean, auc_std, acc_mean, acc_std, best_model

def merge_results(all_res, params_dir):
    diffs = []
    # dataset_name = "all_wandbs"
    for key in all_res:  
        dfs = all_res[key]
        dfold = read_cur_res(dfs, params_dir, key)
        auc_mean, auc_std, acc_mean, acc_std, best_model = get_results(dfold)
        # else:
        #     auc_mean, auc_std, acc_mean, acc_std, winauc_mean, winauc_std, winacc_mean, winacc_std = get_results_pamfirst(dparams)
                    
#         print(f"\nparams: {params}, auc_mean: {auc_mean}, acc_mean: {acc_mean}, winauc_mean: {winauc_mean}, winacc_mean: {winacc_mean}")
#         print(f"    auc_std: {auc_std}, acc_std: {acc_std}, winauc_std: {winauc_std}, winacc_std: {winacc_std}")
        
#         print("="*20)
                
        # print(key, auc_mean, auc_std, acc_mean, acc_std, winauc_mean, winauc_std, winacc_mean, winacc_std)
        print(key + "," + str(auc_mean) + "±" + str(auc_std) + "," + str(acc_mean) + "±" + str(acc_std))
    
    return best_model

def cal_res(wandb_config, project, sweep_dict, dconfig, dataset_name, model_names, emb_types, update, extract_best_model="", abs_dir="", pred_dir="", launch_file="", generate_all=False, save_dir=""):
    model_names = model_names.split(",")
    emb_types = emb_types.split(",")
    for model_name in model_names:
        for emb_type in emb_types:
            dconfig.setdefault(model_name, dict())
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fname = os.path.join(save_dir, dataset_name + "_" + model_name+ "_" + emb_type + ".pkl")
            if update or not os.path.exists(fname):
                print("extracting the results from wandb")
                all_res = downloads(project, sweep_dict, dataset_name, model_name, emb_type)
                pd.to_pickle([all_res], fname)
            else:
                print("reading the results from pkl files")
                res = pd.read_pickle(fname)
                all_res = res[0]
            # print("all_res", all_res)
            best_model_fold_first = merge_results(all_res, "all_wandbs")
            # print("="*20)
            # best_model_params_first = merge_results(all_res, "all_wandbs", False)

            if extract_best_model:
                print("extracting the best model of {} in {}".format(model_names, dataset_name))
                model_path_fold_first = []
                for model_path in best_model_fold_first:
                    model_path_fold_first.append(abs_dir + "/" + model_path)
                dconfig[model_name]["model_path_fold_first"] = model_path_fold_first
                ftarget = os.path.join(pred_dir, "{}_{}_{}_fold_first_predict.yaml".format(dataset_name, model_name, emb_type))
                generate_wandb(fpath, ftarget, model_path_fold_first)
                write_config(dataset_name, dconfig)
                # wandb_path = "./configs/wandb.json"
                # sweep_shell = "start_predict.sh"
                generate_sweep(wandb_config, pred_dir, launch_file, ftarget, generate_all)

def write_config(dataset_name, dconfig):
    with open(CONFIG_FILE) as fin:
        data_config = json.load(fin)
        data_config[dataset_name] = dconfig
    with open(CONFIG_FILE, "w") as fout:
        data = json.dumps(data_config, ensure_ascii=False, indent=4)
        fout.write(data)

#修改wandb配置文件
def generate_wandb(fpath, ftarget, model_path):
    with open(fpath,"r") as fin,\
        open(ftarget,"w") as fout:
        data = yaml.load(fin, Loader=yaml.FullLoader)
        name = ftarget.split('_')
        data['name'] = '_'.join(name[:4])
        data['parameters']['save_dir']['values'] = model_path
        data['parameters']['save_dir']['values'] = model_path
        yaml.dump(data, fout)

# # 生成启动sweep的脚本
def generate_sweep(wandb_config, pred_dir, sweep_shell, ftarget, generate_all):
    # with open(wandb_path) as fin:
    #     wandb_config = json.load(fin)
    pre = "WANDB_API_KEY=" + wandb_config["api_key"] + " wandb sweep "
    with open(sweep_shell,"w") as fallsh:
        if generate_all:
            files = os.listdir(pred_dir)
            files = sorted(files)
            for f in files:
                fpath = os.path.join(pred_dir, f)
                fallsh.write(pre + fpath + "\n")
        else:
            fallsh.write(pre + ftarget + "\n")

def main(params):
    project_name, dataset_name, model_names, emb_types, update, extract_best_model, abs_dir, pred_dir, launch_file, generate_all, save_dir = params["project_name"], params["dataset_name"], \
    params["model_names"], params["emb_types"], params["update"], params["extract_best_model"], params["abs_dir"], params["pred_dir"], params["launch_file"], params["generate_all"], params["save_dir"]
    project = api.project(name=project_name)
    print("="*20)
    print(f"Reading the results from {project} of {uid}")
    sweep_dict = get_sweep_dict(project)
    # print(f"sweep_dict: {sweep_dict}")
    print("="*20)
    dconfig = dict()
    cal_res(wandb_config, project_name, sweep_dict, dconfig, dataset_name, model_names, emb_types, update, extract_best_model, abs_dir, pred_dir, launch_file, generate_all, save_dir)

if __name__ == "__main__":
    api = wandb.Api()
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="./seedwandb/")
    parser.add_argument("--project_name", type=str, default="kt_toolkits")
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_names", type=str, default="dkt")
    parser.add_argument("--emb_types", type=str, default="qid")
    parser.add_argument("--update", type=str2bool, default="True")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--extract_best_model", type=str2bool, default="False")
    parser.add_argument("--abs_dir", type=str, default="")
    parser.add_argument("--pred_dir", type=str, default="pred_wandbs")
    parser.add_argument("--launch_file", type=str, default="start_predict.sh")
    parser.add_argument("--generate_all", type=str2bool, default="False")

    args = parser.parse_args()

    fpath = "./{}/predict.yaml".format(args.src_dir)
    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)

    params = vars(args)
    print(params)
    main(params)
