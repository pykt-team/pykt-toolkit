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
        start_time = [int(float(t)) for t in row["usetimes"].split(",")]
        timestamps = [int(t) for t in row["timestamps"].split(",")]
        for at in start_time:
            if str(at) not in at2idx:
                at2idx[str(at)] = len(at2idx)            
        shft_timestamps = [0] + timestamps[:-1]
        it = np.maximum(np.minimum((np.array(timestamps) - np.array(shft_timestamps)) // 60, 43200),-1)
        for t in it:
            if str(t) not in it2idx:
                it2idx[str(t)] = len(it2idx)
    at2idx["-1"] = len(at2idx)
    it2idx["-1"] = len(it2idx)
    return at2idx, it2idx

def generate_qmatrix(data_config, gamma=0.0):
    df_train = pd.read_csv(os.path.join(data_config["dpath"], "train_valid.csv"))
    df_test = pd.read_csv(os.path.join(data_config["dpath"], "test.csv"))
    df = pd.concat([df_train, df_test])    

    problem2skill = dict()
    for i, row in df.iterrows():
        cids = [int(_) for _ in row["concepts"].split(",")]
        qids = [int(_) for _ in row["questions"].split(",")]
        for q,c in zip(qids, cids):
            if q in problem2skill:
                problem2skill[q].append(c)
            else:
                problem2skill[q] = [c]
    n_problem, n_skill = data_config["num_q"], data_config["num_c"]
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p in problem2skill.keys():
        for c in problem2skill[q]:
            q_matrix[p][c] = 1
    np.savez(os.path.join(data_config["dpath"], "qmatrix.npz"), matrix = q_matrix)
    return q_matrix

