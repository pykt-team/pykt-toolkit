import torch

import os
import numpy as np
import pandas as pd

def get_gkt_graph(num_c, dpath, trainfile, testfile, graph_type="dense", tofile="./graph.npz"):
    graph = None
    df_train = pd.read_csv(os.path.join(dpath, trainfile))
    df_test = pd.read_csv(os.path.join(dpath, testfile))
    df = pd.concat([df_train, df_test])  
    if graph_type == 'dense':
        graph = build_dense_graph(num_c)
    elif graph_type == 'transition':
        graph = build_transition_graph(df, num_c)
    np.savez(os.path.join(dpath, tofile), matrix = graph)
    return graph

def build_transition_graph(df, concept_num):
    graph = np.zeros((concept_num, concept_num))
    for _, row in df.iterrows():
        questions = list(filter(lambda x: x != '-1',
                                row['concepts'].split(',')))
        seq_len = len(questions)
        for i in range(seq_len-1):
            pre = int(questions[i])
            next = int(questions[i+1])
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))

    def inv(x):
        if x == 0:
            return x
        return 1. / x

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    graph = torch.from_numpy(graph).float()
    
    return graph

def build_dense_graph(node_num):
    graph = 1. / (node_num - 1) * np.ones((node_num, node_num))
    np.fill_diagonal(graph, 0)
    graph = torch.from_numpy(graph).float()
    return graph