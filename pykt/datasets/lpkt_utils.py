#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
import numpy as np

def generate_time2idx(data_config):
    df_train = pd.read_csv(os.path.join(data_config["dpath"], "train_valid.csv"))
    df_test = pd.read_csv(os.path.join(data_config["dpath"], "test.csv"))
    # df_window_test = pd.read_csv(os.path.join(data_config["dpath"], data_config["test_window_file"]))
    # df = pd.concat([df_train, df_test, df_window_test])
    df = pd.concat([df_train, df_test])
    
    # generate at2idx & it2idx
    at2idx, it2idx = dict(), dict()
    for i, row in df.iterrows():
        if "usetimes" in row:
            use_time = [int(float(t)) for t in row["usetimes"].split(",")]
            use_time = [x // 1000 for x in use_time]
            for at in use_time:
                if str(at) not in at2idx:
                    at2idx[str(at)] = len(at2idx)     
        if "timestamps" in row:  
            timestamps = [int(t) for t in row["timestamps"].split(",")]
            shft_timestamps = timestamps[:1] + timestamps[:-1]
            it = np.maximum(np.minimum((np.array(timestamps) - np.array(shft_timestamps)) // 1000 // 60, 43200),-1)
            for t in it:
                if str(t) not in it2idx:
                    it2idx[str(t)] = len(it2idx)
        else:
            # no timestamps
            it2idx["1"] = len(it2idx)
    at2idx["-1"] = len(at2idx)
    it2idx["-1"] = len(it2idx)
    return at2idx, it2idx