#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
import numpy as np

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

