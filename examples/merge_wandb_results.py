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

def downloads(project_name, sweep_dict, dataset_name="", model_name=""):
    # print(f"project_name:{project_name}, dataset_name:{dataset_name}, model_name:{model_name}")
    all_res = dict()
    na_res = dict()
    for key in sweep_dict:
        emb_type = key.split("_")[-2]     
        if dataset_name != "" and model_name != "" and (key.find(dataset_name+"_") == -1 or key.find("_"+model_name+"_"+emb_type) == -1):
            continue
        try:
            df = get_df(key,sweep_dict,project_name)
            curna = df[df["testauc"].isna()]
            df = df.dropna(subset=["testauc"])
            print(f"key: {key}, df dropna: {df.shape}")
            tmps = key.split("_")
            key = "_".join(tmps[0:-1])
            all_res.setdefault(key, dict())
            fold = tmps[-1]
            print(f"key: {key}, fold: {fold}, df: {df.shape}, na: {curna.shape}")
            all_res[key][fold] = df
            na_res.setdefault(key, dict())
            na_res[key][fold] = curna
        except:
            print(f"error: {key}")
            continue
    return all_res, na_res

def get_params(paramsf):
    lines = open(paramsf).read().split("\n")
    params, params_hy = [], []
    for i in range(0, len(lines)):
        cur = lines[i]
        pre = ""
        if i > 0:
            pre = lines[i-1]
        if pre == "":
            continue
        idx = cur.find("values")
        if idx != -1:
            key = pre.strip().replace(":", "")
            params.append(key)
            
            if key not in ["fold", "seed"]:
                params_hy.append(key)
    return params, params_hy

def read_cur_res(dfs, params_dir, key):
    print(f"dfs: {len(dfs)}")
    dfold, dparams, pams = dict(), dict(), dict()
    all_df = pd.concat(dfs, axis=0)
    for fold in dfs: # all fold
        df = dfs[fold]
        df["Name"] = df["Name"].apply(lambda a: int(a.split("-")[-1]))
        df = df.sort_values(by=["Name"])
        paramsf = os.path.join(params_dir, key+"_"+fold+".yaml")
        params, params_hy = get_params(paramsf)
        print(f"params: {params}, params_hy: {params_hy}")
        for i, row in df.iterrows():
            fold, seed = row["fold"], row["seed"]
            params_hystr = "_".join([str(s) for s in row[params_hy].values])
            # if i == 0:
            #     print(f"params_hystr:{params_hystr} fold: {fold} seed: {seed}")
            dfold.setdefault(fold, dict())
            dfold[fold].setdefault(params_hystr, dict())
            dfold[fold][params_hystr].setdefault(seed, [])

            dparams.setdefault(params_hystr, dict())
            dparams[params_hystr].setdefault(fold, dict())
            dparams[params_hystr][fold].setdefault(seed, [])
            
            params_str = "_".join([str(s) for s in row[params].values])
            pams[params_str] = [row["testauc"], row["Name"]]
            savepath = row["model_save_path"] if "model_save_path" in row else ""
            values = [row["testauc"], row["testacc"], row["window_testauc"], row["window_testacc"], row["Name"], savepath]
            dfold[fold][params_hystr][seed] = values
            dparams[params_hystr][fold][seed] = values
        # print(f"all_df: {all_df.shape}, df: {df.shape}, len(pams): {len(pams)}")
    return dfold, dparams

def read_na(dfs, params_dir, key):
    dna = dict()
    for fold in dfs:
        # key fold: df
        df = dfs[fold]
        df["Name"] = df["Name"].apply(lambda a: int(a.split("-")[-1]))
        paramsf = os.path.join(params_dir, key+"_"+fold+".yaml")
        # params, params_hy = get_params(paramsf)
        if df.shape[0] > 0:
            params, params_hy = get_params(paramsf)
            print(f"read_na! key: {key}, fold: {fold}, na: {df.shape}")
        for i, row in df.iterrows():
            fold, seed = row["fold"], row["seed"]
            params_hystr = "_".join([str(s) for s in row[params_hy].values])
            dna.setdefault(fold, dict())
            dna[fold].setdefault(params_hystr, dict())
            dna[fold][params_hystr].setdefault(seed, [])
            dna[fold][params_hystr][seed].append(row["Name"])
    return dna

def get_results(d, dna):
    aucs, accs, winaucs, winaccs = [], [], [], []
    num = 0
    
    best_model = []
    for fold in d:
        best_params = ""
        best_savepath = ""
        best_seed = ""
        best_name = -1
        auc, acc, winauc, winacc = 0.0, 0.0, 0.0, 0.0
        for params in d[fold]:
            for seed in d[fold][params]:
                num += 1
                curauc, curacc, curwinauc, curwinacc, name, cursavepath = d[fold][params][seed]
#                 print(f"params: {params}, fold: {fold}, seed: {seed}, len: {len(d[params][fold][seed])}")
                if curauc > auc:
                    auc, acc, winauc, winacc = curauc, curacc, curwinauc, curwinacc
                    best_params = params
                    best_seed = seed
                    best_name = name
                    best_savepath = cursavepath

#             print(f"params: {params}, fold: {fold}, len: {len(d[params][fold])}, auc: {auc}, acc: {acc}, winauc: {winauc}, winacc: {winacc}")
        # check best is in na or not!!!
        if fold in dna and best_params in dna[fold] and best_seed in dna[fold][best_params]:
            names = sorted(dna[fold][best_params][best_seed])
            if names[-1] > best_name:
                print(f"fold: {fold} best params: {best_params} has na, need retrain!! na name: {names}, best_name: {best_name}")
        print(f"fold: {fold}, best_name: {best_name}, auc: {auc}, acc: {acc}, windowauc: {winauc}, windowacc: {winacc}")
        aucs.append(auc)
        accs.append(acc)
        winaucs.append(winauc)
        winaccs.append(winacc)

        params = best_params.split("_")
        params[0], params[1] = params[1], params[0]
        params.extend([str(best_seed), str(fold)])
        # print("params", params)
        best_model.append(["_".join(params), best_savepath])
                
    auc_mean, auc_std = round(np.mean(aucs), 4), round(np.std(aucs, ddof=0), 4)
    acc_mean, acc_std = round(np.mean(accs), 4), round(np.std(accs, ddof=0), 4)
    winauc_mean, winauc_std = round(np.mean(winaucs), 4), round(np.std(winaucs, ddof=0), 4)
    winacc_mean, winacc_std = round(np.mean(winaccs), 4), round(np.std(winaccs, ddof=0), 4)

    return auc_mean, auc_std, acc_mean, acc_std, winauc_mean, winauc_std, winacc_mean, winacc_std, best_model

def get_results_pamfirst(d):
    maxres = [0.0] * 8
    best_params = ""
    for params in d:
        # if len(d[params]) > 1:
        #     print(f"params: {params}, len(d[params]): {len(d[params])}")
        aucs, accs, winaucs, winaccs = [], [], [], []
        for fold in d[params]:
            auc, acc, winauc, winacc = 0.0, 0.0, 0.0, 0.0
            # print(f"len(d[params]): {len(d[params])}, len(d[params][fold]): {len(d[params][fold])}")
            for seed in d[params][fold]:
                curauc, curacc, curwinauc, curwinacc = d[params][fold][seed]
#                 print(f"params: {params}, fold: {fold}, seed: {seed}, len: {len(d[params][fold][seed])}")
                if curauc > auc:
                    auc, acc, winauc, winacc = curauc, curacc, curwinauc, curwinacc
#             print(f"params: {params}, fold: {fold}, len: {len(d[params][fold])}, auc: {auc}, acc: {acc}, winauc: {winauc}, winacc: {winacc}")
            aucs.append(auc)
            accs.append(acc)
            winaucs.append(winauc)
            winaccs.append(winacc)
            
        auc_mean, auc_std = round(np.mean(aucs), 4), round(np.std(aucs, ddof=0), 4)
        acc_mean, acc_std = round(np.mean(accs), 4), round(np.std(accs, ddof=0), 4)
        winauc_mean, winauc_std = round(np.mean(winaucs), 4), round(np.std(winaucs, ddof=0), 4)
        winacc_mean, winacc_std = round(np.mean(winaccs), 4), round(np.std(winaccs, ddof=0), 4)         
#         print(f"\nparams: {params}, auc_mean: {auc_mean}, acc_mean: {acc_mean}, winauc_mean: {winauc_mean}, winacc_mean: {winacc_mean}")
#         print(f"    auc_std: {auc_std}, acc_std: {acc_std}, winauc_std: {winauc_std}, winacc_std: {winacc_std}")
        
#         print("="*20)
        
        if auc_mean > maxres[0]:
            maxres = [auc_mean, auc_std, acc_mean, acc_std, winauc_mean, winauc_std, winacc_mean, winacc_std]
            best_params = params
    print(f"best_params: {best_params}, len: {len(d[best_params])}")
    return maxres

def merge_results(all_res, na_res, params_dir, fold_first=True):
    for key in all_res:  
        
        dfs = all_res[key]
        dfold, dparams = read_cur_res(dfs, params_dir, key)
        dna = read_na(na_res[key], params_dir, key)
        if fold_first:
            auc_mean, auc_std, acc_mean, acc_std, winauc_mean, winauc_std, winacc_mean, winacc_std, best_model = get_results(dfold, dna)
        #else:
            #auc_mean, auc_std, acc_mean, acc_std, winauc_mean, winauc_std, winacc_mean, winacc_std = get_results_pamfirst(dparams)
                    
#         print(f"\nparams: {params}, auc_mean: {auc_mean}, acc_mean: {acc_mean}, winauc_mean: {winauc_mean}, winacc_mean: {winacc_mean}")
#         print(f"    auc_std: {auc_std}, acc_std: {acc_std}, winauc_std: {winauc_std}, winacc_std: {winacc_std}")
#         print("="*20)
        # print(key, auc_mean, auc_std, acc_mean, acc_std, winauc_mean, winauc_std, winacc_mean, winacc_std)
        print(key + "," + str(auc_mean) + "±" + str(auc_std) + "," + str(acc_mean) + "±" + str(acc_std) + "," + str(winauc_mean) + "±" + str(winauc_std) + "," + str(winacc_mean) + "±" + str(winacc_std))
    
    return best_model

def cal_res(wandb_config, project, sweep_dict, dconfig, dataset_name, model_names, 
        update, extract_best_model="", pred_dir="", launch_file="", generate_all=False, save_dir="", model_dir=""):
    model_names = model_names.split(",")
    with open(launch_file,"w") as fallsh:
        for model_name in model_names:
            dconfig.setdefault(model_name, dict())
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fname = os.path.join(save_dir, dataset_name + "_" + model_name + ".pkl")
            if update or not os.path.exists(fname):
                print("extracting the results from wandb")
                all_res, na_res = downloads(project, sweep_dict, dataset_name, model_name)
                pd.to_pickle([all_res,na_res], fname)
            else:
                print("reading the results from pkl files")
                res = pd.read_pickle(fname)
                all_res, na_res = res[0], res[1]
            # print("all_res", all_res)
            best_model_fold_first = merge_results(all_res, na_res, "all_wandbs", True)

            if extract_best_model:
                print("extracting the best model of {} in {}".format(model_names, dataset_name))
                model_path_fold_first = []
                for best_params, best_savepath in best_model_fold_first:
                    # model_path = "/".join(best_savepath.split("/")[0:-1])   
                    model_path = model_dir + "/" + best_savepath.split("/")[-2]
                    
                    best_model_dir = "./all_bestmodel/{}/{}".format(dataset_name, model_name)
                    if not os.path.exists(model_path):
                        print(f"model_path: {model_path} not found!")
                    else: 
                        abs_best_dir = os.path.abspath(best_model_dir)
                        if not os.path.exists(best_model_dir):
                            os.makedirs(best_model_dir)
                        abs_best_dir = os.path.abspath(best_model_dir)#.replace("/data/", "/hw/share/")
                        cmd = "cp -r " + model_path + " " + best_model_dir + "/"
                        # print("please copy the best model to the best model save path by the following commands!!!")
                        print(cmd)
                        os.system(cmd)
                            
                        if best_savepath == "":
                            model_path_fold_first.append(os.path.join(abs_best_dir, best_params))
                        else:
                            model_path_fold_first.append(os.path.join(abs_best_dir, model_path.split("/")[-1]))

                dconfig[model_name]["model_path_fold_first"] = model_path_fold_first
                ftarget = os.path.join(pred_dir, "{}_{}_fold_first_predict.yaml".format(dataset_name, model_name))
                generate_wandb(fpath, ftarget, model_path_fold_first)
                write_config(dataset_name, dconfig)
                # wandb_path = "./configs/wandb.json"
                # sweep_shell = "start_predict.sh"
                generate_sweep(wandb_config, project, pred_dir, fallsh, ftarget, generate_all)
        fallsh.write("fi"+ "\n")

def write_config(dataset_name, dconfig):
    with open(CONFIG_FILE) as fin:
        if fin.read().strip() == "":
            data_config = dict()
        else:
            data_config = json.load(open(CONFIG_FILE))
        data_config[dataset_name] = dconfig
    with open(CONFIG_FILE, "w") as fout:
        data = json.dumps(data_config, ensure_ascii=False, indent=4)
        fout.write(data)

def generate_wandb(fpath, ftarget, model_path):
    with open(fpath,"r") as fin,\
        open(ftarget,"w") as fout:
        data = yaml.load(fin, Loader=yaml.FullLoader)
        name = ftarget.split('_')
        data['name'] = '_'.join(name[:4])
        data['parameters']['save_dir']['values'] = model_path
        data['parameters']['save_dir']['values'] = model_path
        yaml.dump(data, fout)

def generate_sweep(wandb_config, project_name, pred_dir, fallsh, ftarget, generate_all):
    # with open(wandb_path) as fin:
    #     wandb_config = json.load(fin)
    pre = "WANDB_API_KEY=" + wandb_config["api_key"] + " wandb sweep "
    if generate_all:
        files = os.listdir(pred_dir)
        files = sorted(files)
        for f in files:
            fpath = os.path.join(pred_dir, f)
            fallsh.write(pre + fpath + " -p {}".format(project_name)  + "\n")
    else:
        fallsh.write(pre + ftarget + " -p {}".format(project_name) + "\n")

def main(params):
    project_name, dataset_name, model_names, update, extract_best_model, pred_dir, launch_file, generate_all, save_dir = params["project_name"], params["dataset_name"], \
    params["model_names"], params["update"], params["extract_best_model"], params["pred_dir"], params["launch_file"], params["generate_all"], params["save_dir"]
    project = api.project(name=project_name)
    print("="*20)
    print(f"Reading the results from {project} of {uid}")
    sweep_dict = get_sweep_dict(project)
    # print(f"sweep_dict: {sweep_dict}")
    print("="*20)
    dconfig = dict()
    model_dir = params["model_dir"]
    cal_res(wandb_config, project_name, sweep_dict, dconfig, dataset_name, model_names, 
            update, extract_best_model, pred_dir, launch_file, generate_all, save_dir, model_dir)

if __name__ == "__main__":
    api = wandb.Api()
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="../seedwandb/")
    parser.add_argument("--project_name", type=str, default="kt_toolkits")
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_names", type=str, default="dkt")
    parser.add_argument("--update", type=str2bool, default="True")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--model_dir", type=str, default="./saved_model")
    parser.add_argument("--extract_best_model", type=str2bool, default="False")
    parser.add_argument("--pred_dir", type=str, default="pred_wandbs")
    parser.add_argument("--launch_file", type=str, default="start_predict.sh")
    parser.add_argument("--generate_all", type=str2bool, default="False")

    args = parser.parse_args()

    fpath = "{}/predict.yaml".format(args.src_dir)
    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)

    params = vars(args)
    print(params)
    main(params)
