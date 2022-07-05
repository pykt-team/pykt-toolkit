import os, sys
import json

with open("../configs/wandb.json") as fin:
    config = json.load(fin)
    WANDB_API_KEY = config["api_key"]

logf = sys.argv[1]
outf = open(sys.argv[2], "w")
start = int(sys.argv[3])
end = int(sys.argv[4])

dataset_name = sys.argv[5]
model_name = sys.argv[6]
nums = sys.argv[7].split(",")
print(len(sys.argv))
if len(sys.argv) == 8:
    project_name = "kt_toolkits"
else:
    project_name = sys.argv[8]

cmdpre = f"WANDB_API_KEY={WANDB_API_KEY} nohup "
endcmdpre =f"WANDB_API_KEY={WANDB_API_KEY} "

idx = 0
with open(logf, "r") as fin:
    i = 0
    lines = fin.readlines()
    l = []
    num = 0
    while i < len(lines):
        if lines[i].strip().startswith("wandb: Creating sweep from: "):
            fname = lines[i].strip().split(": ")[-1].split("/")[-1]
        else:
            print("error!")
        if lines[i+3].strip().startswith("wandb: Run sweep agent with: "):
            sweepid = lines[i+3].strip().split(": ")[-1]
        else:
            print("error!")
        fname = fname.split(".")[0]
        print(f"fname is {fname}")
        if not fname.startswith(dataset_name) or fname.find("_" + model_name + "_") == -1:
            i += 4
            continue
        print(f"dataset_name: {dataset_name}, model_name: {model_name}, fname: {fname}")
        if idx >= start and idx < end:
            cmd = "CUDA_VISIBLE_DEVICES=" + str(nums[num]) +" " + cmdpre + sweepid + " &"
            outf.write(cmd + "\n")
            num += 1
        idx += 1
        i += 4
