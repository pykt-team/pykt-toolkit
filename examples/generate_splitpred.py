import os, sys
import json
import argparse
from tkinter.messagebox import YES

def str2bool(str):
    return True if str.lower() == "true" else False

dnames = {
    "assist2009": "Assist2009",
    "assist2015": "Assist2015", 
    "algebra2005": "Algebra2005",
    "bridge2algebra2006": "Bridge2006",
    "nips_task34": "NIPS34",
    "statics2011": "Statics2011",
    "poj": "POJ"
}

# 生成启动sweep的脚本
def main(params):
    src_dir, project_name, dataset_names, all_dir, launch_file = params["src_dir"], params["project_name"], params["dataset_names"], params["all_dir"], params["launch_file"]
    model_list = params["model_names"].split(",")
    best_models = params["best_models"]
    if not os.path.exists(all_dir):
        os.makedirs(all_dir)
    with open("../configs/wandb.json") as fin,\
        open(launch_file,"w") as fallsh:
        wandb_config = json.load(fin)
        pre = "WANDB_API_KEY=" + wandb_config["api_key"] + " wandb sweep "
        for dataset_name in dataset_names.split(","):
            files = os.listdir(src_dir)
            
            allbestdirs = []
            for model_name in model_list:
                if dataset_name in ["assist2009", "assist2015"]:
                    if model_name == "dkt_forget":
                        continue
                curbest = []
                curmdir = os.path.join(best_models, dataset_name, model_name)
                for f in os.listdir(curmdir):
                    if not os.path.exists(os.path.join(curmdir, f, "qid_model.ckpt")):
                        continue
                    curbest.append(os.path.join(curmdir, f))
                allbestdirs.extend(curbest)
            print(allbestdirs)
            fname = dataset_name + "_splitpred_qid_"+"_".join(model_list)+".yaml" # all
            ftarget = os.path.join(all_dir, fname)
            fout = open(ftarget, "w")  
            fpath = os.path.join(src_dir, "splitpred.yaml")
            print(fpath, ftarget)
            with open(fpath, "r") as fin:
                data = fin.read()
                data = data.replace("best_dirs", str(allbestdirs))
                fout.write("name: " + fname.split(".")[0] + "\n")
                fout.write(data)
            
            fallsh.write(pre + ftarget + " -p {}".format(project_name) + "\n")
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="./seedwandb/")
    parser.add_argument("--project_name", type=str, default="kt_splitpre")
    parser.add_argument("--dataset_names", type=str, default="assist2015")
    parser.add_argument("--model_names", type=str, default="dkt,dkvmn")
    parser.add_argument("--all_dir", type=str, default="all_wandbs")
    parser.add_argument("--launch_file", type=str, default="all_start.sh")
    parser.add_argument("--best_models", type=str, default="./all_bestmodels/")

    args = parser.parse_args()
    params = vars(args)
    print(params)
    main(params)
