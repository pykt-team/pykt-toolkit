import pandas as pd
import os

def sort_samples(train_valid_path):
    df = pd.read_csv(train_valid_path)
    dict_len = dict()
    for i,row in df.iterrows():
        seq = row["responses"].split(",")
        new_seq = [x for x in seq if x != "-1"]
        dict_len[i] = len(new_seq)
    dict_len = sorted(dict_len.items(),key=lambda x:x[1],reverse=False)
    sort_index = [x[0] for x in dict_len]
    sort_df = pd.DataFrame(df, index=sort_index)
    # cl_dpath = os.path.join(dpath, "train_valid_sequences_cl.csv")
    # sort_df = df
    # sort_df = df.drop([df.columns[[0]]], axis=1)
    # sort_df.to_csv(cl_dpath, index=None)
    

    return sort_df
    

