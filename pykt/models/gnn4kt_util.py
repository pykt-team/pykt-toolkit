import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_graph(data_config, topk):
    df_train = pd.read_csv(os.path.join(data_config["dpath"], "train_valid.csv"))
    df_test = pd.read_csv(os.path.join(data_config["dpath"], "test.csv"))
    df = pd.concat([df_train, df_test])
    qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
    if os.path.exists(qmatrix_path):
        q_matrix = np.load(qmatrix_path, allow_pickle=True)['matrix']
    else:
        q_matrix = generate_qmatrix(data_config, df)
    similar_dict = cal_question_relation(q_matrix)
    corr_matrix = cal_corr_matrix(df, corr_matrix=np.zeros([q_matrix.shape[0], q_matrix.shape[0]]))
    sorted_idx = np.argsort(-corr_matrix, axis=1)

    similar_corr_dict = dict()
    for i,q in enumerate(similar_dict):
        corr_list = similar_corr_dict.setdefault(q,{})
        for simi_q in similar_dict[q]:
            corr_list[simi_q] = corr_matrix[int(q)][int(simi_q)]

    sorted_q_dict = {}
    for qid in similar_corr_dict:
        sorted_dict = sorted(similar_corr_dict[qid].items(), key=lambda x: x[1],reverse=True)
        sorted_q_dict[qid] = sorted_dict

    final_graph = select_similar_q(topk, sorted_dict=sorted_q_dict, similar_dict=similar_dict, sorted_corr_matrix=sorted_idx)

    with open(os.path.join(data_config["dpath"],f"gnn4kt_graph_{topk}.txt"),"w") as f:
        cnt = 0
        for q in final_graph:
            cnt += 1
            for sim_q in final_graph[q]: #从相似矩阵里找
                f.write(str(q) + " " + str(sim_q) + "\n")

    return 
        

def generate_qmatrix(data_config, df, gamma=0.0):
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
        for c in problem2skill[p]:
            q_matrix[p][c] = 1
    # np.savez(os.path.join(data_config["dpath"], "qmatrix.npz"), matrix = q_matrix)
    return q_matrix

def cal_question_relation(q_matrix):
    similar_dict = dict()
    for i in range(q_matrix.shape[0]):
        similar_list = similar_dict.setdefault(str(i),[])
        for j in range(q_matrix.shape[0]):
            same_kc = q_matrix[i]*q_matrix[j]
            same_kc_num = np.count_nonzero(same_kc == 1)
            if same_kc_num > 0 and i != j:
                similar_list.append(j)
    
    return similar_dict


def cal_corr_matrix(df, corr_matrix):
    for i, row in df.iterrows():
        qids = [int(_) for _ in row["questions"].split(",")]
        qids = list(set(qids))
        for i,qid in enumerate(qids):
            for j in range(i+1, len(qids)):
                corr_matrix[int(qid)][int(qids[j])] += 1
                corr_matrix[int(qids[j])][int(qid)] += 1
    return corr_matrix

def select_similar_q(n, sorted_dict, similar_dict, sorted_corr_matrix):
    select_similar_dict = {}
    for q in similar_dict:
        sim_q_list = select_similar_dict.setdefault(q,[])
        for sim_q, v in sorted_dict[q][:n]:
            sim_q_list.append(sim_q)
        if len(sim_q_list) < n: #相同的知识点不到10个
            # print(f"{sim_q_list}")
            # print(f"{sorted_corr_matrix[int(q)][:n-len(sim_q_list)]}")
            sim_q_list += list(sorted_corr_matrix[int(q)][:n-len(sim_q_list)])
    return select_similar_dict

def load_graph(path, n):
    n += 1
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    # print(f"idx_map:{idx_map}")
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(self.in_features, self.out_features)
        # self.params = Parameter(torch.FloatTensor(in_features, out_features))
        # torch.nn.init.xavier_uniform_(self.params)

    def forward(self, features, adj, active=True):
        # support = torch.mm(features, self.params)
        # # print(f"support:{support.shape}")
        # # print(f"adj:{adj.shape}")
        # output = torch.spmm(adj, support)
        # if active:
        #     output = F.relu(output)
        # return output
        # GCN 步骤
        support = torch.matmul(adj, features)
        output = torch.matmul(adj, support)
        output = self.fc1(output)
        if active:
            output = F.relu(output)
        return output