import os, sys
import pandas as pd
import numpy as np
import json, copy
from .split_datasets import read_data,ALL_KEYS,ONE_KEYS,extend_multi_concepts,save_dcur
from .split_datasets import train_test_split,KFold_split,calStatistics,get_max_concepts,id_mapping,write_config


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

def save_id2idx(dkeyid2idx, save_path):
    with open(save_path, "w+") as fout:
        fout.write(json.dumps(dkeyid2idx))
    
def id_mapping_que(df):
    id_keys = ["questions", "concepts","uid"]
    dres = dict()
    dkeyid2idx = dict()
    print(f"df.columns: {df.columns}")
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]
    for i, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            dkeyid2idx.setdefault(key, dict())
            dres.setdefault(key, [])
            curids = []
            for id in row[key].split(","):
                sub_ids = id.split('_')
                sub_curids = []
                for sub_id in sub_ids:
                    if sub_id not in dkeyid2idx[key]:
                        dkeyid2idx[key][sub_id] = len(dkeyid2idx[key])
                    sub_curids.append(str(dkeyid2idx[key][sub_id]))
                curids.append("_".join(sub_curids))
            dres[key].append(",".join(curids))
    finaldf = pd.DataFrame(dres)
    return finaldf, dkeyid2idx

def main(dname, fname, dataset_name, configf, min_seq_len = 3, maxlen = 200, kfold = 5):
    """split main function

    Args:
        dname (str): data folder path
        fname (str): the data file used to split, needs 6 columns, format is: (NA indicates the dataset has no corresponding info)
            uid,seqlen: 50121,4
            quetion ids: NA
            concept ids: 7014,7014,7014,7014
            responses: 0,1,1,1
            timestamps: NA
            cost times: NA
        dataset_name (str): dataset name
        configf (str): the dataconfig file path
        min_seq_len (int, optional): the min seqlen, sequences less than this value will be filtered out. Defaults to 3.
        maxlen (int, optional): the max seqlen. Defaults to 200.
        kfold (int, optional): the folds num needs to split. Defaults to 5.
        
    """
    stares = []

    total_df, effective_keys = read_data(fname)
    #cal max_concepts
    if 'concepts' in effective_keys:
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1

    oris, _, qs, cs, seqnum = calStatistics(total_df, stares, "original")
    print("="*20)
    print(f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

     # just for id map
    total_df, dkeyid2idx = id_mapping_que(total_df)
    dkeyid2idx["max_concepts"] = max_concepts

    save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))
    effective_keys.add("fold")

    df_save_keys = []
    for key in ALL_KEYS:
        if key in effective_keys:
            df_save_keys.append(key)

    # train test split
    train_df, test_df = train_test_split(total_df, 0.2)
    splitdf = KFold_split(train_df, kfold)
    splitdf[df_save_keys].to_csv(os.path.join(dname, "train_valid_quelevel.csv"), index=None)
    ins, ss, qs, cs, seqnum = calStatistics(splitdf, stares, "original train+valid question level")
    print(f"train+valid original interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    # generate sequences
    split_seqs = generate_sequences(splitdf, effective_keys, min_seq_len, maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(split_seqs, stares, "train+valid sequences question level")
    print(f"train+valid sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    split_seqs.to_csv(os.path.join(dname, "train_valid_sequences_quelevel.csv"), index=None)
    # print(f"split seqs dtypes: {split_seqs.dtypes}")



    # for test dataset
    # add default fold -1 to test!
    test_df["fold"] = [-1] * test_df.shape[0]  
    test_seqs = generate_sequences(test_df, list(effective_keys), min_seq_len, maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(test_df, stares, "test original question level")
    print(f"original test interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    ins, ss, qs, cs, seqnum = calStatistics(test_seqs, stares, "test sequences question level")
    print(f"test sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    print("="*20)

    test_window_seqs = generate_window_sequences(test_df, list(effective_keys), maxlen)

    
    test_df = test_df[df_save_keys]
    test_df.to_csv(os.path.join(dname, "test_quelevel.csv"), index=None)
    test_seqs.to_csv(os.path.join(dname, "test_sequences_quelevel.csv"), index=None)
    test_window_seqs.to_csv(os.path.join(dname, "test_window_sequences_quelevel.csv"), index=None)

    ins, ss, qs, cs, seqnum = calStatistics(test_window_seqs, stares, "test window question level")
    print(f"test window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    

    other_config = {
        "train_valid_original_file_quelevel": "train_valid_quelevel.csv", 
        "train_valid_file_quelevel": "train_valid_sequences_quelevel.csv",
        "test_file_quelevel": "test_sequences_quelevel.csv",
        "test_window_file_quelevel": "test_window_sequences_quelevel.csv",
        "test_original_file_quelevel": "test_quelevel.csv"
    }
    
    write_config(dataset_name=dataset_name, dkeyid2idx=dkeyid2idx, effective_keys=effective_keys, 
                configf=configf, dpath = dname, k=kfold,min_seq_len = min_seq_len, maxlen=maxlen,other_config=other_config)
    
    print("="*20)
    print("\n".join(stares))

