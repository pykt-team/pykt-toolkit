import os, sys
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="input_path")
args = parser.parse_args()

dres = dict()
with open(args.input_path, "r") as fin:
    for line in fin.readlines():
        line = line.strip("\r\n")
        lines = line.split("\t")
        if len(lines) == 6:
            idx, uid, qidx = int(lines[0]), int(lines[1]), int(lines[2])
            dres.setdefault(idx, dict())
            prob = float(lines[3])
            dres[idx][qidx] = prob

finalres = []
sorted_dres = sorted(dres.items(), key=lambda k: k[0], reverse=False)
for info in sorted_dres:
    idx, infos = info[0], info[1]
    sorted_infos = sorted(infos.items(), key=lambda k: k[0], reverse=False)
    probs = []
    for prob_infos in sorted_infos:
        probs.append(prob_infos[1])
    finalres.append(",".join([str(p) for p in probs]))

df_submit = pd.DataFrame({"responses":finalres})
df_submit.to_csv("prediction.csv",index=False)
