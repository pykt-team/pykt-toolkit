import pandas as pd
import os, sys

def process(dlabel, df, dbeta, K=20): # 只对训练验证数据做插入，训练时，只取训练数据部分的增强结果
    dfinal = dict()
    dtyp = {0: "original", 1: "insert", 2: "crop", 3: "mask", 4: "reorder"}
    for i, row in df.iterrows():
    
        uid = row["uid"]
        dcur = parse_row(row)
        import copy
        dnew = copy.deepcopy(dcur)
        typ = 0
        if dlabel[uid][0] != 0: # 不均匀
            clen = len(dcur['concepts'])
            if clen <= K:
                typ = 1
            else:
                typ = random.sample([1,2,3,4], 1)[0]
            if typ in [2, 3, 4] and clen <= K:
                typ = 1 # 若处理后的序列长度小于20，则该序列使用插入操作
            if typ == 1:
                num = int(clen*dbeta["insert"])
                dnew = insert_row(dnew, num)
            elif typ == 2:
                num = int(clen*dbeta["crop"])
                dnew = crop_row(dnew, num)
            elif typ == 3:
                num = int(clen*dbeta["mask"])
                dnew = mask_row(dnew, num)
            else:
                num = int(clen*dbeta["reorder"])
                dnew = reorder_row(dnew, num)
            for key in dnew:
                dfinal.setdefault(key, [])
                if key in ["uid", "fold"]:
                    dfinal[key].append(dnew[key])
                else:
                    dfinal[key].append(",".join(dnew[key]))
            dfinal.setdefault("type", [])
            dfinal["type"].append(dtyp[typ])
        for key in dcur:
            dfinal.setdefault(key, [])
            if key in ["uid", "fold"]:
                dfinal[key].append(dcur[key])
            else:
                dfinal[key].append(",".join(dcur[key]))
        dfinal.setdefault("type", [])
        dfinal["type"].append(dtyp[0])

    finaldf = pd.DataFrame(dfinal)  
    return finaldf

def cal_timedelta(ts):
    delta = []
    for i in range(0, len(ts) - 1):
        if int(ts[i+1]) == -1:
            break
        delta.append(int(ts[i+1]) - int(ts[i]))
    return delta

def split_sequences(df): #
    dstds = dict()
    idx = 0
    for i, row in df.iterrows():

        delta = cal_timedelta(row["timestamps"].split(","))
        import numpy as np
        num = np.std(delta) / 1000 / 60 / 60 # hour
        uid = row["uid"]
        dstds[uid] = num#[len(ts), num])
        idx += 1
    print(f"dstds: {len(dstds)}, idx: {idx}")
    sorted_dstds = sorted(dstds.items(), key = lambda d: d[1])
    dlabel = dict()
    idx = 0
    for inf in sorted_dstds:
#         print(inf[0], inf[1])
        if idx <= int(len(sorted_dstds)/5):
            dlabel[inf[0]] = [0, inf[1]] # 均匀
        elif idx > int(len(sorted_dstds)*4/5):
            dlabel[inf[0]] = [1, inf[1]] # 非常不均匀
        else:
            dlabel[inf[0]] = [2, inf[1]] # 不均匀
        idx += 1
    return dlabel

def get_deltas(dnew):
    delta = cal_timedelta(dnew["timestamps"])
    ddelta = dict()
    for i in range(0, len(delta)):
        ddelta[str(i)+"-"+str(i+1)] = delta[i]
    
    return ddelta

def parse_row(row):
    dres = dict()
    for key in row.keys():
        if key in ["fold", "uid"]:
            dres[key] = row[key]
            continue
        dres[key] = row[key].split(",")
    return dres

import random

def add_one(dcur, left, right, select_type="random"):
    if select_type == "random":
        selected_idx = random.randint(right, len(dcur["timestamps"])-1)
    elif select_type == "first":
        dfirst = dict()
        idxlist = []
        concepts = dcur["concepts"]
        for i in range(right, len(concepts)):
            curc = concepts[i]
            if curc not in dfirst:
                dfirst[curc] = i
                idxlist.append(i)
        selected_idx = random.sample(idxlist, 1)[0]
        
    t = int((int(dcur["timestamps"][left])+int(dcur["timestamps"][right])) / 2)
    t = str(t)
    dres = dict()
    for key in dcur:
        if key in ["fold", "uid"]:
            dres[key] = dcur[key]
            continue
        inf = dcur[key]
        if key != "timestamps":
            dres[key] = inf[0:left+1]+dcur[key][selected_idx:selected_idx+1]+inf[right:]
        else:
#             print(f"insert: {[t]}, len: {len([t])}")
            dres[key] = inf[0:left+1]+[t]+inf[right:]
#         print(f"key: {key}, totallen: {len(dres[key])}, left: {left}, right: {right}, lenleft: {len(inf[0:left+1])}, lenright: {len(inf[right:])}")
    keys, lens = [], []
    for key in dres:
        if key not in ["fold", "uid"]:
            keys.append(key)
            lens.append(len(dres[key]))
    if len(set(lens)) != 1:
        print(keys, lens)
        assert False
    return dres
    
def insert_row(dnew, num):
    
#             print(f"num: {num}, qlen: {len(dcur['questions'])}")
    for j in range(0, num):
        delta = get_deltas(dnew)
        sorted_delta = sorted(delta.items(), key = lambda k: k[1], reverse=True)
        inf = sorted_delta[0]
        left, right = inf[0].split("-")
        left, right = int(left), int(right)
        # 从自己的序列后半部分随机选择一个样本插入
#                 print(f"j: {j}, left: {left}, right: {right}")
        dnew = add_one(dnew, left, right)
#                 print(f"j: {j}, left: {left}, right: {right}, lenq: {len(dnew['questions'])}")
    return dnew

# 裁切得到标准差最小的子序列
def crop_row(dnew, num):
    slen = len(dnew['timestamps'])
    deltas = dict()
    for i in range(0, slen):
        if i + num > slen:
            break
        delta = cal_timedelta(dnew["timestamps"][i: i+num])
        import numpy as np
        stdnum = np.std(delta) / 1000 / 60 / 60 # hour
        deltas[i] = stdnum
    sorted_delta = sorted(deltas.items(), key = lambda d: d[1])
    start = sorted_delta[0][0]
    dfinal = dict()
    for key in dnew:
        if key in ["uid", "fold"]:
            dfinal[key] = dnew[key]
            continue
        dfinal[key] = dnew[key][start: start+num]
    
    return dfinal

def mask_one(dnew, left, right):
    dfinal = dict()
    for key in dnew:
        if key in ["uid", "fold"]:
            dfinal[key] = dnew[key]
            continue
        dfinal[key] = dnew[key][0:left+1] + dnew[key][right+1:]
    return dfinal

def mask_row(dnew, num):
    delta = get_deltas(dnew)
    sorted_delta = sorted(delta.items(), key = lambda k: k[1])
    delidxs = []
    for inf in sorted_delta[0:num]:
        delidxs.append(inf[0].split("-")[1]) # right
    dfinal = dict()
    for key in dnew:
        if key in ["uid", "fold"]:
            dfinal[key] = dnew[key]
            continue
        cur = []
        for i in range(0, len(dnew[key])):
            if i not in delidxs:
                cur.append(dnew[key][i])
        dfinal[key] = cur
    '''
    for j in range(0, num):
        delta = get_deltas(dnew)
        sorted_delta = sorted(delta.items(), key = lambda k: k[1])
        inf = sorted_delta[0]
        left, right = inf[0].split("-")
        left, right = int(left), int(right)

        dnew = mask_one(dnew, left, right)
    '''
    return dfinal

def reorder_row(dnew, num):
    slen = len(dnew['timestamps'])
    deltas = dict()
    for i in range(0, slen):
        if i + num > slen:
            break
        delta = cal_timedelta(dnew["timestamps"][i: i+num])
        import numpy as np
        stdnum = np.std(delta) / 1000 / 60 / 60 # hour
        deltas[i] = stdnum
    sorted_delta = sorted(deltas.items(), key = lambda d: d[1])
    start = sorted_delta[0][0]
    
    shuffled = []
    tmpkeys = set(list(dnew.keys())) - set(["uid", "fold", "timestamps"])
    for i in range(start, start+num):
        cur = []
        for key in tmpkeys:
            cur.append([key, dnew[key][i]])
        shuffled.append(cur)
    random.shuffle(shuffled)
    
    dshuffled = dict()
    for cur in shuffled:
        for inf in cur:
            key, val = inf[0], inf[1]
            dshuffled.setdefault(key, [])
            dshuffled[key].append(val)
    
    dfinal = dict()
    for key in dnew:
        if key in ["uid", "fold"]:
            dfinal[key] = dnew[key]
            continue
        if key != "timestamps":
            dfinal[key] = dnew[key][0: start] + dshuffled[key] + dnew[key][start+num:]
        else:
            dfinal[key] = dnew[key]
    
    return dfinal

