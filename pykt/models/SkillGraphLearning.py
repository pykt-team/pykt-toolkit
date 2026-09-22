import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 与 QueGraphLearing 一致，支持 notebook 通过 _DATASET_DIR_OVERRIDE 指定数据根
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_DATASET_DIR_OVERRIDE = None


def _get_project_root():
    if _DATASET_DIR_OVERRIDE:
        return os.path.dirname(os.path.dirname(_DATASET_DIR_OVERRIDE))
    return _PROJECT_ROOT


def _get_dataset_dir(dataset_name):
    if _DATASET_DIR_OVERRIDE and dataset_name == "assist2009":
        return _DATASET_DIR_OVERRIDE
    return os.path.join(_get_project_root(), "data", dataset_name)

class GCNConv(nn.Module):  # 提取特征
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Parameter(torch.rand((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.rand((out_dim)))
        nn.init.zeros_(self.b)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.matmul(x, self.w)
        x = torch.sparse.mm(adj.float(), x)
        x = x + self.b
        return x



def get_kc_embedding(last_kc, kc_emb, padding_idx=-1):
    """
    将知识点ID转换为对应的embedding并进行平均池化

    Args:
        last_kc: shape [batch_size, seq_len, max_concepts], 每个位置包含多个知识点ID
        kc_emb: shape [kc_max, dim], 知识点embedding矩阵
        padding_idx: int, 填充值，默认为-1

    Returns:
        pooled_emb: shape [batch_size, seq_len, dim], 每个位置的知识点embedding平均值
    """
    batch_size, seq_len, max_concepts = last_kc.shape
    dim = kc_emb.size(1)
    device = last_kc.device

    # 创建mask来标识非填充位置
    mask = (last_kc != padding_idx)  # [batch_size, seq_len, max_concepts]

    # 将填充位置的ID设为0，避免索引越界
    # 注意：这里设为0是因为后面会用mask把这些位置的影响消除
    last_kc = last_kc.clamp(min=0)

    # 获取所有知识点的embedding
    # reshape是为了使用F.embedding
    flat_kc = last_kc.view(-1)  # [batch_size * seq_len * max_concepts]
    flat_emb = F.embedding(flat_kc, kc_emb)  # [batch_size * seq_len * max_concepts, dim]

    # 恢复原始形状
    emb = flat_emb.view(batch_size, seq_len, max_concepts, dim)  # [batch_size, seq_len, max_concepts, dim]

    # 扩展mask维度以适配embedding维度
    mask = mask.unsqueeze(-1).expand(-1, -1, -1, dim)  # [batch_size, seq_len, max_concepts, dim]

    # 使用mask将填充位置设为0
    masked_emb = emb * mask.float()

    # 计算每个位置的非填充知识点数量
    concept_counts = mask.sum(dim=2, keepdim=True)  # [batch_size, seq_len, 1, dim]

    # 避免除零错误
    concept_counts = concept_counts.clamp(min=1.0)

    # 计算平均值
    pooled_emb = masked_emb.sum(dim=2) / concept_counts.squeeze(2)  # [batch_size, seq_len, dim]

    return pooled_emb


class GCN_Graph_Learning(nn.Module):  # 实现GCN的图学习
    def __init__(self, d, p, num_gcn_layers):
        super(GCN_Graph_Learning, self).__init__()

        self.gcn_layers = nn.ModuleList([
            GCNConv(d, d, p) for _ in range(num_gcn_layers)
        ])

    def forward(self, x, adj):
        # 依次通过多层GCN
        embed = x
        for gcn_layer in self.gcn_layers:
            embed = gcn_layer(embed, adj)
            embed = F.relu(embed)  # 添加非线性激活

        return (x + embed)  # 添加残差连接


class Concepts_Embedding(nn.Module):
    def __init__(self, skill_max, d, p, dataset_name, num_gcn_layers=1):
        super(Concepts_Embedding, self).__init__()

        self.d_model = d

        # bert的嵌入
        self.bert_projection = nn.Linear(1024, self.d_model)

        # GCN_Graph_Learning 是图学习模块
        self.gcl = GCN_Graph_Learning(d, p, num_gcn_layers)

        # 问题嵌入和回答嵌入
        self.kc_embed = nn.Parameter(torch.ones((skill_max, 1024)))  # 问题嵌入
        self.ans_embed = nn.Embedding(2, d)  # 回答嵌入

        # nn.init.xavier_uniform_(self.kc_embed)

        # 通过bert的嵌入去初始化kc的嵌入
        file_path = os.path.join(_get_dataset_dir(dataset_name), f"kc_embeddings_{dataset_name}_bge.npy")
        bert_embed = np.load(file_path)
        bert_tensor = torch.from_numpy(bert_embed).float()

        # 加上KC——id本身的嵌入
        kc_id_embed = nn.Embedding(skill_max, 1024)

        self.kc_embed = nn.Parameter(self.bert_projection(self.kc_embed * bert_tensor + kc_id_embed.weight))

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, last_ans, last_skill, next_skill, matrix):
        """
        last_pro: 上一个问题的索引
        last_ans: 上一个问题的回答嵌入
        next_pro: 下一个问题的索引
        matrix: 邻接矩阵图(kc-kc)
        """

        # 通过GCN获得学习后的embedding权重矩阵
        kc_embed = self.gcl(self.kc_embed, matrix)
        # kc_embed = kc_embed.to(device)

        # 通过平均池化获得kc的嵌入
        last_kc_embed = get_kc_embedding(last_skill, kc_embed)  # [batch_size, seq_len, dim]
        next_kc_embed = get_kc_embedding(next_skill, kc_embed)  # [batch_size, seq_len, dim]

        # 获取回答的嵌入
        ans_embed = self.ans_embed(last_ans)

        X = (last_kc_embed + ans_embed)
        # print("X.shape: ", X.shape)

        return X, next_kc_embed
