import os, sys
import json
import argparse
from tkinter.messagebox import YES
with open("../configs/wandb.json") as fin:
    config = json.load(fin)
    WANDB_API_KEY = config["api_key"]

def str2bool(str):
    return True if str.lower() == "true" else False

# 生成启动sweep的脚本
def main(params):
    src_dir, project_name, dataset_names, model_names, folds, save_dir_suffix, all_dir, launch_file, generate_all = params["src_dir"], params["project_name"], params["dataset_names"], \
    params["model_names"], params["folds"], params["save_dir_suffix"], params["all_dir"], params["launch_file"], params["generate_all"]
    if not os.path.exists(all_dir):
        os.makedirs(all_dir)
    with open("../configs/wandb.json") as fin,\
        open(launch_file,"w") as fallsh:
        wandb_config = json.load(fin)
        WANDB_API_KEY = os.getenv("WANDB_API_KEY")
        if WANDB_API_KEY == None:
            WANDB_API_KEY = wandb_config["api_key"]
        print(WANDB_API_KEY)
        pre = "WANDB_API_KEY=" + WANDB_API_KEY + " wandb sweep "
        for dataset_name in dataset_names.split(","):
            files = os.listdir(src_dir)
            for m in model_names.split(","):
                for _type in [["qid"]]:
                    for fold in folds.split(","):
                        _type = [str(k) for k in _type]
                        fname = dataset_name + "_" + m + "_" + _type[0].replace("linear", "") + "_" + str(fold) + ".yaml"
                        ftarget = os.path.join(all_dir, fname)
                        fpath = m + ".yaml"
                        fpath = os.path.join(src_dir, fpath)
                        print(fpath, ftarget)
                        with open(fpath, "r") as fin,\
                            open(ftarget, "w") as fout:
                            data = fin.read()
        #                     data = data.replace("[\"dkt\"]", "[\"" + m + "\"]")
                            data = data.replace("xes", dataset_name)
                            data = data.replace("tiaocan", "tiaocan_"+dataset_name+save_dir_suffix)
                            data = data.replace("[\"qid\"]", str(_type))
                            data = data.replace("[0, 1, 2, 3, 4]", str([fold]))
                            data = data.replace('BATCH_SIZE',str(params["batch_size"]))
                            fout.write("name: " + fname.split(".")[0] + "\n")
                            fout.write(data)
                        
                        if not generate_all:
                            fallsh.write(pre + ftarget + " -p {}".format(project_name) + "\n")
        
        if generate_all:
            files = os.listdir(all_dir)
            files = sorted(files)
            for f in files:
                fpath = os.path.join(all_dir, f)
                fallsh.write(pre + fpath + " -p {}".format(project_name)  + "\n")
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="./seedwandb/")
    parser.add_argument("--project_name", type=str, default="kt_toolkits")
    parser.add_argument("--dataset_names", type=str, default="assist2015")
    parser.add_argument("--model_names", type=str, default="dkt,dkt+,dkt_forget,kqn,atktfix,dkvmn,sakt,saint,akt,gkt")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_dir_suffix", type=str, default="")
    parser.add_argument("--all_dir", type=str, default="all_wandbs")
    parser.add_argument("--launch_file", type=str, default="all_start.sh")
    parser.add_argument("--generate_all", type=str2bool, default="False")

    args = parser.parse_args()
    params = vars(args)
    print(params)
    main(params)
