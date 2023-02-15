import os
import sys
import json
import yaml 
import numpy as np
import collections
import shutil

dataset_names=sys.argv[1]
model_names=sys.argv[2]
emb_types=sys.argv[3]

# new_emb_type = emb_type.rstrip("_attn")
 
# # 设定文件路径
# path=f'/root/autodl-nas/huangshuyan/attention/pykt-toolkit/examples/best_model_path/{dataset_name}/{model_name}/{emb_type}'

# ori_path=f'/root/autodl-nas/huangshuyan/attention/pykt-toolkit/examples/best_model_path/{dataset_name}/{model_name}/{emb_type}'
# new_path = f'/root/autodl-nas/huangshuyan/attention/pykt-toolkit/examples/best_model_path/{dataset_name}/{model_name}/{emb_type}'
# fileList=os.listdir(path)
# # 获取该目录下所有文件，存入列表中

# print(fileList)


# # -----------------------修改新跑的最优模型文件名----------------------
# # fileList=os.listdir(ori_path)
# # for i in fileList:
# #     if "csv" not in i:
# #         # 设置旧文件名（就是路径+文件名）
# #         oldname = ori_path + os.sep + i  # os.sep添加系统分隔符
# #         # # 设置新文件名
# #         tmp_name = i.split("_")
# #         newname = new_path + os.sep + model_name + "_tiaocan_" + dataset_name +"_" +"_".join(tmp_name[3:])
# #         os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名


# # # -----------------------修改新跑的最优模型文件名----------------------
# fileList=os.listdir(path)
# for i in fileList:
#     if "csv" not in i:
#         # 设置旧文件名（就是路径+文件名）
#         oldname = path + os.sep + i + os.sep + "qid_model.ckpt"  # os.sep添加系统分隔符
    
#         # # 设置新文件名
#         newname = path + os.sep + i + os.sep + f"{emb_type}_model.ckpt"

#         os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    
#         print(oldname, '======>', newname)

#         filename = path + os.sep + i + os.sep + "config.json"
#         with open(filename, "r") as load_f:
#             load_json = json.load(load_f)
#         load_json["params"]["emb_type"] = emb_type

#         with open(filename, "w") as dump_f:
#             json.dump(load_json,dump_f)

dataset_names = dataset_names.split(",")
emb_types = emb_types.split(",")
model_names = model_names.split(",")

for dataset_name in dataset_names:
    for model_name in model_names:
        for emb_type in emb_types:
            pred_dir="pred_wandbs"
            fpath="./seedwandb/predict.yaml"
            ftarget = os.path.join(pred_dir, "{}_{}_{}_predict.yaml".format(dataset_name, model_name, emb_type))
            wandb_key = "07ec58f8e8315aa083a202292092d7e2ee9d43b4"

            pre = "WANDB_API_KEY=" + wandb_key + " wandb sweep "
            sweep_shell = "start_predict.sh"
            project_name = "sparse_attn_predicitons"

            path=f'/root/autodl-nas/huangshuyan/attention/pykt-toolkit/examples/best_model_path/{dataset_name}/{model_name}/{emb_type}'
            fileList=os.listdir(path)

            new_fileList = []
            for i in fileList:
                if ".csv" not in i:
                    filename = path + os.sep + i
                    new_fileList.append(filename)

            with open(sweep_shell,"a+") as fallsh,\
                open(fpath,"r") as fin,\
                open(ftarget,"w") as fout:
                data = yaml.load(fin, Loader=yaml.FullLoader)
                name = ftarget.split('_')
                data['name'] = '_'.join([dataset_name, model_name, emb_type, 'prediction'])
                data['parameters']['save_dir']['values'] = new_fileList
                yaml.dump(data, fout)
                fallsh.write(pre + ftarget + " -p {}".format(project_name) + "\n")



# -----------------------rerun best_model----------------------
# for i,file in enumerate(fileList):
#     if "csv" not in file:
#         filename = path + os.sep + file + os.sep + "config.json"
#         with open(filename, "r") as load_f:
#             load_json = json.load(load_f)
#         seed = load_json["params"]["seed"]
#         fold = load_json["params"]["fold"]
#         dropout = load_json["params"]["dropout"]
        
#         final_fc_dim = load_json["params"]["final_fc_dim"]
#         final_fc_dim2 = load_json["params"]["final_fc_dim2"]
#         num_layers = load_json["params"]["num_layers"]
#         nheads = load_json["params"]["nheads"]
#         loss1 = load_json["params"]["loss1"]
#         loss2 = load_json["params"]["loss2"]
#         loss3 = load_json["params"]["loss3"]
#         start = load_json["params"]["start"]
#         d_model = load_json["params"]["d_model"]
#         d_ff = load_json["params"]["d_ff"]
#         num_attn_heads = load_json["params"]["num_attn_heads"]
#         n_blocks = load_json["params"]["n_blocks"]
#         learning_rate = load_json["params"]["learning_rate"]
        
#         print(f"WANDB_API_KEY=07ec58f8e8315aa083a202292092d7e2ee9d43b4 CUDA_VISIBLE_DEVICES={i} nohup python wandb_{model_name}_train.py --dataset={dataset_name} --dropout={dropout} --final_fc_dim={final_fc_dim} --final_fc_dim2={final_fc_dim2} --num_layers={num_layers} --nheads={nheads} --loss1={loss1} --loss2={loss2} --loss3={loss3} --start={start}  --d_model={d_model} --d_ff={d_ff} --num_attn_heads={num_attn_heads} --n_blocks={n_blocks} --learning_rate={learning_rate} --seed={seed} --emb_type={emb_type} --save_dir=rerun_model --fold={fold} &")


# with open("./2006_sub_scores.txt","r") as f:
#     all_scores = []
#     for line in f:
#         line = eval(line)
#         all_scores.extend(line)

# print(f"avg:{np.mean(all_scores)}, max:{np.max(all_scores)}, min:{np.min(all_scores)}")
# total_scores = [round(x,1) for x in all_scores]
# res = collections.Counter(total_scores)
# print(f"res:{res}")


# with open("./2009_sub_scores_new.txt","r") as f:
#     all_scores = []
#     for line in f:
#         line = eval(line)
#         all_scores.extend(line)

# print(f"2009: avg:{np.mean(all_scores)}, max:{np.max(all_scores)}, min:{np.min(all_scores)}")
# total_scores = [round(x,1) for x in all_scores]
# res = collections.Counter(total_scores)
# print(f"2009: res:{res}")

# with open("./nips34_sub_scores.txt","r") as f:
#     all_scores = []
#     for line in f:
#         line = eval(line)
#         all_scores.extend(line)

# print(f"nips34: avg:{np.mean(all_scores)}, max:{np.max(all_scores)}, min:{np.min(all_scores)}")
# total_scores = [round(x,1) for x in all_scores]
# res = collections.Counter(total_scores)
# print(f"nips34: res:{res}")


# with open("./2015_sub_scores_new.txt","r") as f:
#     all_scores = []
#     for line in f:
#         # try:
#         #     line = eval(line)
#         # except:
#         line = line.rstrip("\n")
#         line = line.replace("]","")
#         line = line.replace("[","")
#         line = line.split(",")
#         line = [float(x) for x in line]
#         all_scores.extend(line)

# print(f"2015: avg:{np.mean(all_scores)}, max:{np.max(all_scores)}, min:{np.min(all_scores)}")
# total_scores = [round(x,1) for x in all_scores]
# res = collections.Counter(total_scores)
# print(f"2015: res:{res}")


# with open("./2015_sub_scores_final.txt","r") as f:
#     all_scores = []
#     for line in f:
#         line = eval(line)
#         all_scores.extend(line)

# print(f"2015: avg:{np.mean(all_scores)}, max:{np.max(all_scores)}, min:{np.min(all_scores)}")
# total_scores = [round(x,1) for x in all_scores]
# res = collections.Counter(total_scores)
# print(f"2015: res:{res}")


# with open("./poj_sub_scores_final.txt","r") as f:
#     all_scores = []
#     for line in f:
#         line = eval(line)
#     print(f"lens: {len(all_scores)}")
# print(f"poj: avg:{np.mean(all_scores)}, max:{np.max(all_scores)}, min:{np.min(all_scores)}")
# total_scores = [round(x,1) for x in all_scores]
# res = collections.Counter(total_scores)
# print(f"poj: res:{res}")

#---------------------重新训练模型----------------

# import pandas as pd
# from pykt.utils.wandb_utils import WandbUtils

# dataset_names = dataset_names.split(",")
# emb_types = emb_types.split(",")
# model_names = model_names.split(",")

# wandb_key = "07ec58f8e8315aa083a202292092d7e2ee9d43b4"
# os.environ['WANDB_API_KEY'] = wandb_key

# wandb_api = WandbUtils(user='shyann', project_name=f'simplekt_attention')

# new_df = dict()
# new_df["model_name"] = []
# new_df["dataset_name"] = []
# new_df["emb_type"] = []
# new_df["fold"] = []
# new_df["model_save_path"] = []
# new_df["validauc"] = []

# with open(f"rerun_best_model_final.txt","w") as f:
# for dataset_name in dataset_names:
#     for model_name in model_names:
#         for emb_type in emb_types:
#             df = wandb_api.get_best_run(dataset_name, model_name, emb_type)
#             i = 0
#             for j,row in df.iterrows():
#                 if not os.path.exists(row['model_save_path']):
#                     save_dir = row["save_dir"]
#                     d_model = row["d_model"]
#                     d_ff = row["d_ff"]
#                     final_fc_dim = row["final_fc_dim"]
#                     final_fc_dim2 = row["final_fc_dim2"]
#                     dropout = row["dropout"]
#                     learning_rate = row["learning_rate"]
#                     num_attn_heads = row["num_attn_heads"]
#                     n_blocks = row["n_blocks"]
#                     sparse_ratio = row["sparse_ratio"]
#                     k_index = row["k_index"]
#                     stride = row["stride"]
#                     seed = row["seed"]
#                     fold = row["fold"]
#                     command = f"WANDB_API_KEY=07ec58f8e8315aa083a202292092d7e2ee9d43b4 CUDA_VISIBLE_DEVICES={i} nohup python wandb_{model_name}_train.py --dataset={dataset_name} --dropout={dropout} --final_fc_dim={final_fc_dim} --final_fc_dim2={final_fc_dim2} --d_model={d_model} --d_ff={d_ff} --num_attn_heads={num_attn_heads} --n_blocks={n_blocks} --learning_rate={learning_rate} --seed={seed} --emb_type={emb_type} --save_dir={save_dir} --fold={fold} --sparse_ratio={sparse_ratio} --k_index={k_index} --stride={stride} &"
#                     f.write(command + "\n")
#                     i += 1

#                     new_df["model_name"].append(model_name)
#                     new_df["dataset_name"].append(dataset_name)
#                     new_df["emb_type"].append(emb_type)
#                     new_df["fold"].append(fold)
#                     new_df["validauc"].append(row['validauc'])
#                     new_df["model_save_path"].append(row['model_save_path'])
#                 else:
#                     model_path = row['model_save_path'].rstrip(f"{emb_type}_model.ckpt")
#                     tmp_model_path = model_path.split("/")[-2]
#                     target_path = f"./best_model_path/{dataset_name}/{model_name}/{emb_type}/{tmp_model_path}"
#                     shutil.copytree(model_path, target_path)
#                     print(f"copy {model_path} to {target_path} done")

# my_df = pd.DataFrame(new_df)
# my_df.to_csv('dataframe_final.csv')

# df = pd.read_csv("./dataframe_final.csv")
# for j,row in df.iterrows():
#     emb_type = row["emb_type"]
#     target_path = row['model_save_path'].rstrip(f"{emb_type}_model.ckpt")
#     print(f"target_path:{target_path}")
#     model_path = row['rerun_model_path'].rstrip(f"{emb_type}_model.ckpt")
#     shutil.copytree(model_path, target_path)
#     print(f"copy {model_path} to {target_path} done")