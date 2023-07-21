import pandas as pd
import json
import numpy as np
import os

ONE_KEYS = ["fold", "uid"]
ALL_KEYS = ["fold", "uid", "questions", "concepts", "responses", "timestamps", "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow","cidxs"]

def get_pretrain_data(seqlen, data_config):
    print(f"datapath:{data_config['dpath']}")
    uni_path = "/".join(data_config['dpath'].split("/")[:-1])
    full_data_path = os.path.join(data_config["dpath"], f"train_valid_quelevel_pretrain.csv")
    if not os.path.exists(full_data_path):    
        datasets = ["assist2009", "algebra2005", "bridge2algebra2006", "nips_task34", "ednet", "peiyou", "ednet5w"]
        new_df = merge_data(uni_path, datasets)
        finaldf, dkeyid2idx = id_mapping_que(new_df)
        finaldf.to_csv(full_data_path, index=None)
        ins, ss, qs, cs, seqnum = calStatistics(df=finaldf, stares=[], key="original train+valid question level")
        print(f"pretrain  sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
        for data in datasets:
            final_sub_df = finaldf[finaldf.dataset==data]
            ins_, ss_, qs_, cs_, seqnum_ = calStatistics(df=final_sub_df, stares=[], key="original train+valid question level")
            print(f"dataset:{data}, ins_:{ins_}, ss_:{ss_}, qs_:{qs_}, cs_:{cs_}, seqnum_:{seqnum_}")

    # build traning and valid set
    split_seqs = generate_sequences(df=finaldf, effective_keys={"uid","questions","concepts","responses","fold", "timestamps"}, min_seq_len=3, maxlen=seqlen)
    dpath = data_config["dpath"]
    split_seqs.to_csv(f"{dpath}/train_valid_sequences_quelevel_{seqlen}.csv", index=None)
    ins, ss, qs, cs, seqnum = calStatistics(df=split_seqs, stares=[], key="train+valid sequences question level")
    print(f"pretrain  sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    # map testing set
    return 

def get_test_data(seqlen, dataset_name):
    new_test_df = map_dataset(datasets)
    dpath = data_config["dpath"]
    if os.path.exists(f"{dpath}/test_quelevel_pretrain.csv"):
        test_df = pd.read_csv(f"{uni_path}/{data}/test_quelevel_pretrain.csv")
        test_window_seqs = generate_window_sequences(test_df, list({"uid","questions","concepts","responses","fold"}), maxlen=seqlen)
        test_window_seqs.to_csv(f"{dpath}/test_window_sequences_quelevel_pretrain_1024.csv")
    
    datasets = ["assist2009", "algebra2005", "bridge2algebra2006", "nips_task34", "ednet", "peiyou"]
    for dataset in datasets:
        sub_df = new_df[new_df.dataset==dataset]
        new_map_data = {"fold":[], "uid":[], "questions":[], "concepts":[], "responses":[], "timestamps":[]}
        for i,row in sub_df.iterrows():
            questions = [str(dkeyid2idx["questions"][dataset][x]) for x in row["questions"].split(",")]
            concepts = row["concepts"].split(",")
            new_concepts = []
            for ccc in concepts:
                ccc = ccc.split("_")
                concept = [str(dkeyid2idx["concepts"][dataset][c]) for c in ccc]
                # concept = [str(dkeyid2idx.get("concepts").get(dataset).get(c,len(dkeyid2idx['concepts'][dataset]))) for c in ccc]
                concept = "_".join(concept)
                new_concepts.append(concept)
            new_map_data["fold"].append(row["fold"])
            new_map_data["uid"].append(row["uid"])
            new_map_data["questions"].append(",".join(questions))
            new_map_data["concepts"].append(",".join(new_concepts))
            new_map_data["responses"].append(row["responses"])
            new_map_data["timestamps"].append(row["timestamps"])
            new_map_df = pd.DataFrame(new_map_data)
            new_map_df.to_csv(f"{uni_path}/{dataset}/test_quelevel_pretrain.csv")

    datasets = ["ednet5w"]
    for dataset in datasets:
        sub_df = new_df[new_df.dataset==dataset]
        new_map_data = {"fold":[], "uid":[], "questions":[], "concepts":[], "responses":[], "timestamps":[]}
        for i,row in sub_df.iterrows():
            questions = []
            for x in row["questions"].split(","):
                if x in dkeyid2idx["questions"]["ednet"]:
                    questions.append(str(dkeyid2idx["questions"]["ednet"][x]))
                else:
                    questions.append(str(dkeyid2idx["questions"][dataset][x]))
            
            concept = []
            concepts = row["concepts"].split(",")
            new_concepts = []
            for ccc in concepts:
                ccc = ccc.split("_")
                concept = []
                for c in ccc:
                    if c in dkeyid2idx['concepts']["ednet"]:
                        concept.append(str(dkeyid2idx["concepts"]["ednet"][c]))
                    else:
                        concept.append(str(dkeyid2idx["concepts"][dataset][c]))
                concept = "_".join(concept)
                new_concepts.append(concept)
            new_map_data["fold"].append(row["fold"])
            new_map_data["uid"].append(row["uid"])
            new_map_data["questions"].append(",".join(questions))
            new_map_data["concepts"].append(",".join(new_concepts))
            new_map_data["responses"].append(row["responses"])
            new_map_data["timestamps"].append(row["timestamps"])
            new_map_df = pd.DataFrame(new_map_data)
            new_map_df.to_csv(f"{uni_path}/{dataset}/test_quelevel_pretrain.csv")


    

def merge_data(uni_path, datasets):
    new_data = {"fold":[], "uid":[], "questions":[], "concepts":[], "responses":[], "dataset":[], "timestamps":[]}
    for dataset in datasets:
        df_train = pd.read_csv(f"{uni_path}/{dataset}/train_valid_quelevel.csv")
        df_test = pd.read_csv(f"{uni_path}/{dataset}/test_quelevel.csv")
        df = pd.concat([df_train, df_test])
    
    data_info_ = dict()
    with open(f"{uni_path}/{dataset}/keyid2idx.json", "r") as f:
        data_info = json.load(f)
        for key in data_info:
            data_info_.setdefault(key,dict())
            try:
                for item in data_info[key]:
                    data_info_[key][data_info[key][item]] = item
            except:
                print(f"{key}")
                continue
    for i,row in df.iterrows():
        uid = data_info_["uid"][row["uid"]]
        questions = row["questions"].split(",")
        # print(f"questions:{questions}")
        concepts = row["concepts"].split(",")
        # print(f"concepts:{concepts}")
        questions = [data_info_["questions"][int(q)] for q in questions]
        new_concepts = []
        for ccc in concepts:
            ccc = ccc.split("_")
            concept = [data_info_["concepts"][int(c)] for c in ccc]
            concept = "_".join(concept)
            new_concepts.append(concept)
        new_data["fold"].append(row["fold"])
        new_data["uid"].append(uid)
        new_data["questions"].append(",".join(questions))
        new_data["concepts"].append(",".join(new_concepts))
        new_data["responses"].append(row["responses"])
        new_data["dataset"].append(dataset)
        if "timestamps" in row:
            new_data["timestamps"].append(row["timestamps"])
        else:
            new_data["timestamps"].append(",".join([str(i) for i in range(len(questions))]))

    new_df = pd.DataFrame(new_data)
    
    return new_df


def id_mapping_que(df):
    id_keys = ["questions", "concepts", "uid"]
    dres = dict()
    dkeyid2idx = dict()
    # print(f"df.columns: {df.columns}")
    flag = True #判定数据集有没有变化
    pre_data = "assist2009"
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]
    for i, row in df.iterrows():
        dataset = row["dataset"]
        if dataset != "ednet5w":
            for j,key in enumerate(id_keys):
                if key not in df.columns:
                    continue
                if key == "timestamps":
                    dres[key].append(row[key])
                else:
                    dkeyid2idx.setdefault(key, dict())
                    dkeyid2idx[key].setdefault(dataset, dict())
                    dres.setdefault(key, [])
                    curids = []
                    # print(f"dkeyid2idx:{dkeyid2idx}")
                    for id in row[key].split(","):
                        sub_ids = id.split('_')
                        sub_curids = []
                        for sub_id in sub_ids:
                            if sub_id not in dkeyid2idx[key][dataset]:
                                if list(dkeyid2idx[key][dataset].values()) == []: #到新的数据集
                                    if dataset == "assist2009":
                                        dkeyid2idx[key][dataset][sub_id] = 0
                                        flag = False
                                    else:
                                        # print(f"key:{key}, dataset:{dataset}, pre_data:{pre_data}")
                                        # print(f"error:{dkeyid2idx[key][pre_data]}")
                                        dkeyid2idx[key][dataset][sub_id] = list(dkeyid2idx[key][pre_data].values())[-1]+1
                                else: #原本数据集
                                    dkeyid2idx[key][dataset][sub_id] = list(dkeyid2idx[key][dataset].values())[-1]+1
                                # dkeyid2idx[key][dataset][sub_id] = cnt+1
                            sub_curids.append(str(dkeyid2idx[key][dataset][sub_id]))
                            # cnt += 1
                        curids.append("_".join(sub_curids))
                    dres[key].append(",".join(curids))
            pre_data = dataset
        else:
            for j,key in enumerate(id_keys):
                if key not in df.columns:
                    continue
                if key == "timestamps":
                    dres[key].append(row[key])
                else:
                    dkeyid2idx.setdefault(key, dict())
                    dkeyid2idx[key].setdefault(dataset, dict())
                    dres.setdefault(key, [])
                    curids = []
                    # print(f"dkeyid2idx:{dkeyid2idx}")
                    for id in row[key].split(","):
                        sub_ids = id.split('_')
                        sub_curids = []
                        for sub_id in sub_ids:
                            if sub_id in dkeyid2idx[key]["ednet"]:
                                sub_curids.append(str(dkeyid2idx[key]["ednet"][sub_id]))
                            elif sub_id not in dkeyid2idx[key]["ednet"] and sub_id not in dkeyid2idx[key][dataset]:
                                if list(dkeyid2idx[key][dataset].values()) == []: #到新的数据集
                                    dkeyid2idx[key][dataset][sub_id] = list(dkeyid2idx[key]["ednet"].values())[-1]+1
                                else: #原本数据集
                                    dkeyid2idx[key][dataset][sub_id] = list(dkeyid2idx[key][dataset].values())[-1]+1
                                sub_curids.append(str(dkeyid2idx[key][dataset][sub_id]))
                            elif sub_id in dkeyid2idx[key][dataset]:
                                sub_curids.append(str(dkeyid2idx[key][dataset][sub_id]))
                            # cnt += 1
                        curids.append("_".join(sub_curids))
                    dres[key].append(",".join(curids))
    finaldf = pd.DataFrame(dres)
    return finaldf, dkeyid2idx


def calStatistics(df, stares, key):
    allin, allselect = 0, 0
    allqs, allcs = set(), set()
    for i, row in df.iterrows():
        rs = row["responses"].split(",")
        curlen = len(rs) - rs.count("-1")
        allin += curlen
        if "selectmasks" in row:
            ss = row["selectmasks"].split(",")
            slen = ss.count("1")
            allselect += slen
        if "concepts" in row:
            cs = row["concepts"].split(",")
            fc = list()
            for c in cs:
                cc = c.split("_")
                fc.extend(cc)
            curcs = set(fc) - {"-1"}
            allcs |= curcs
        if "questions" in row:
            qs = row["questions"].split(",")
            curqs = set(qs) - {"-1"}
            allqs |= curqs
    stares.append(",".join([str(s) for s in [key, allin, df.shape[0], allselect]]))
    return allin, allselect, len(allqs), len(allcs), df.shape[0]

def generate_sequences(df, effective_keys, min_seq_len=3, maxlen = 200, pad_val = -1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    dropnum = 0
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)

        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        while lenrs >= j + maxlen:   
            rest = rest - (maxlen)
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][j: j + maxlen]))#[str(k) for k in dcur[key][j: j + maxlen]]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))

            j += maxlen
        if rest < min_seq_len:# delete sequence len less than min_seq_len
            dropnum += rest
            continue
        
        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key not in ONE_KEYS:
                paded_info = np.concatenate([dcur[key][j:], np.array([pad_val] * pad_dim)])
                dres[key].append(",".join([str(k) for k in paded_info]))
            else:
                dres[key].append(dcur[key])
        dres["selectmasks"].append(",".join(["1"] * rest + [str(pad_val)] * pad_dim))
    
    # after preprocess data, report
    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    print(f"dropnum: {dropnum}")
    return finaldf

def save_dcur(row, effective_keys):
    dcur = dict()
    for key in effective_keys:
        if key not in ONE_KEYS:
            dcur[key] = row[key].split(",")#[int(i) for i in row[key].split(",")]
        else:
            dcur[key] = row[key]
    return dcur


def map_dataset(uni_path, datasets):
    new_data = {"fold":[], "uid":[], "questions":[], "concepts":[], "responses":[], "dataset":[], "timestamps":[]}
    for dataset in datasets:
        datapath = f"{uni_path}/{dataset}/test_quelevel.csv"
        df = pd.read_csv(datapath)
        data_info_ = dict()
        with open(f"{uni_path}/{dataset}/keyid2idx.json") as f:
            data_info = json.load(f)
            for key in data_info:
                data_info_.setdefault(key,dict())
                try:
                    for item in data_info[key]:
                        data_info_[key][data_info[key][item]] = item
                except:
                    print(f"{key}")
                    continue
        for i,row in df.iterrows():
            uid = data_info_["uid"][row["uid"]]
            questions = row["questions"].split(",")
            # print(f"questions:{questions}")
            concepts = row["concepts"].split(",")
            # print(f"concepts:{concepts}")
            questions = [data_info_["questions"][int(q)] for q in questions]
            new_concepts = []
            for ccc in concepts:
                ccc = ccc.split("_")
                concept = [data_info_["concepts"][int(c)] for c in ccc]
                concept = "_".join(concept)
                new_concepts.append(concept)
            new_data["fold"].append(row["fold"])
            new_data["uid"].append(uid)
            new_data["questions"].append(",".join(questions))
            new_data["concepts"].append(",".join(new_concepts))
            new_data["responses"].append(row["responses"])
            new_data["dataset"].append(dataset)
            if "timestamps" in row:
                new_data["timestamps"].append(row["timestamps"])
            else:
                new_data["timestamps"].append(",".join([str(i) for i in range(len(questions))]))
    new_test_df = pd.DataFrame(new_data)
    
    return new_test_df

def generate_window_sequences(df, effective_keys, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        lenrs = len(dcur["responses"])
        if lenrs > maxlen:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][0: maxlen]))#[str(k) for k in dcur[key][0: maxlen]]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            for j in range(maxlen+1, lenrs+1):
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key not in ONE_KEYS:
                        dres[key].append(",".join([str(k) for k in dcur[key][j-maxlen: j]]))
                    else:
                        dres[key].append(dcur[key])
                dres["selectmasks"].append(",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
        else:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    pad_dim = maxlen - lenrs
                    paded_info = np.concatenate([dcur[key][0:], np.array([pad_val] * pad_dim)])
                    dres[key].append(",".join([str(k) for k in paded_info]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * lenrs + [str(pad_val)] * pad_dim))
    
    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            # print(f"key: {key}, len: {len(dres[key])}")
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return finaldf