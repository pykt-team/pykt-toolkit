#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
import numpy as np

def generate_time2idx(data_config):
    df_train = pd.read_csv(os.path.join(data_config["dpath"], "train_valid.csv"))
    # df_test = pd.read_csv(os.path.join(data_config["dpath"], "test.csv"))
    # df_window_test = pd.read_csv(os.path.join(data_config["dpath"], data_config["test_window_file"]))
    # df = pd.concat([df_train, df_test, df_window_test])
    # df = pd.concat([df_train, df_test])
    df = df_train
    
    # generate at2idx & it2idx
    max_at, max_it = 0, 0
    # at2idx, it2idx = dict(), dict()
    for i, row in df.iterrows():
        if "usetimes" in row:
            start_time = [int(float(t)) for t in row["usetimes"].split(",")]
            max_at = max(start_time) if max(start_time) > max_at else max_at
            # for at in start_time:
            #     if str(at) not in at2idx:
            #         at2idx[str(at)] = len(at2idx)     
        if "timestamps" in row:  
            timestamps = [int(t) for t in row["timestamps"].split(",")]
            shft_timestamps = [0] + timestamps[:-1]
            it = np.maximum(np.minimum((np.array(timestamps) - np.array(shft_timestamps)) // 1000 // 60, 43200),-1).tolist()
            it = [0] + it[1:]
            max_it = max(it) if max(it) > max_it else max_it
            # it = np.maximum(np.minimum((np.array(timestamps) - np.array(shft_timestamps)) / 60, 43200),-1)
            # for t in it:
            #     if str(t) not in it2idx:
            #         it2idx[str(t)] = len(it2idx)
        else:
            # no timestamps
            # it2idx["1"] = len(it2idx)
            max_it = 1
    # at2idx["-1"] = len(at2idx)
    # it2idx["-1"] = len(it2idx)
    return max_at, max_it