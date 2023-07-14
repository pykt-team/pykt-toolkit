from __future__ import print_function, division
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from .gnn4kt_util import GNNLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMLayer, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))
        for _ in range(1, num_layers):
            self.layers.append(nn.LSTM(hidden_size, hidden_size, batch_first=True))
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs = []
        h, c = None, None
        for i in range(self.num_layers):
            x, (h, c) = self.layers[i](x)
            outputs.append(x)
        outputs = torch.stack(outputs, dim=0)
        # outputs,_ = self.lstm(x)
        return outputs

class GNN4KT(nn.Module):

    def __init__(self, n_question, n_pid, embed_l, hidden_size, dropout, num_layers=5, seq_len=200, 
    final_fc_dim=512, final_fc_dim2=256, emb_type="iekt", graph=None, mlp_layer_num=1, sigma=0.1, topk=10, emb_path="", pretrain_dim=768): #
        super(GNN4KT, self).__init__()

        self.model_name = "gnn4kt"
        self.emb_type = emb_type
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_q = n_pid
        self.num_c = n_question
        self.sigma = sigma
        self.emb_size = embed_l

        self.que_emb = nn.Embedding(self.num_q+1, embed_l)#question embeding
        self.concept_emb = nn.Parameter(torch.randn(self.num_c+1, embed_l).to(device), requires_grad=True)#concept embeding

        self.qa_embed = nn.Embedding(2, embed_l)

        self.lstm_layer = LSTMLayer(input_size=embed_l, hidden_size=hidden_size, num_layers=num_layers)
        # self.graph = nn.Parameter(torch.randn(200, 200).to(device), requires_grad=True)

        # self.graph = graph.to_dense()
        # GCN for inter information
        # self.gc = nn.ModuleList([GNNLayer(hidden_size, hidden_size) if _ != 0 else GNNLayer(embed_l, hidden_size) for _ in range(num_layers)])
        # self.out = nn.Sequential(
        #     nn.Linear(embed_l+hidden_size,
        #             final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
        #     nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
        #     ), nn.Dropout(self.dropout),
        #     nn.Linear(final_fc_dim2, 1)
        # )
        self.out_question_all = MLP(mlp_layer_num, hidden_size+embed_l,self.num_q, self.dropout)
        self.m = nn.Sigmoid()

    def get_avg_skill_emb(self,c):
        # add zero for padding
        concept_emb_cat = torch.cat(
            [torch.zeros(1, self.emb_size).to(device), 
            self.concept_emb], dim=0)
        # shift c

        related_concepts = (c+1).long()
        #[batch_size, seq_len, emb_dim]
        concept_emb_sum = concept_emb_cat[related_concepts, :].sum(
            axis=-2)

        #[batch_size, seq_len,1]
        concept_num = torch.where(related_concepts != 0, 1, 0).sum(
            axis=-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg


    def forward(self, dcur):
        # input_data
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)
        emb_q = self.que_emb(pid_data)#[batch,max_len-1,emb_size]
        emb_c = self.get_avg_skill_emb(q_data)#[batch,max_len-1,emb_size]
        q_embed_data = emb_q + emb_c
        qa_embed_data = self.qa_embed(target)


        bs, seqlen = pid_data.size(0), pid_data.size(1)


        # idx = torch.reshape(pid_data,(1, bs*seqlen))[0]

        # for i in range(self.num_layers):
        #     if i == 0:
        #         # qa_embed_data_ = torch.reshape(qa_embed_data, (bs*seqlen, -1))
        #         # print(f"qa_embed_data:{qa_embed_data.shape}")
        #         # print(f"sub_graph:{sub_graph.shape}")
        #         sub_graph = self.graph
        #         h = self.gc[i](self.que_emb.weight, sub_graph)#.reshape(bs*self.num_q,-1)[idx]
        #     # elif i == self.num_layers - 1:
        #     #     h = self.gc[i]((1-self.sigma)*h + self.sigma*output[i-1], sub_graph, active=False).reshape(bs*self.num_q,-1)[idx]
        #     # else:
        #     #     h = self.gc[i]((1-self.sigma)*h + self.sigma*output[i-1], sub_graph).reshape(bs*self.num_q,-1)[idx]
        # h = torch.reshape(h[idx], (bs, seqlen, -1))
        # qa_embed_data = h + qa_embed_data

        #利用lstm得到每层的emb
        output = self.lstm_layer(qa_embed_data) #[num_layers, batch_size, seqlen, hidden_size]

        # # GCN Module
        # # 根据当前数据构建小矩阵
        # output = torch.reshape(output, (self.num_layers, bs*seqlen, -1))
        # idx = torch.reshape(pid_data,(1, bs*seqlen))[0]
        # graph_ = self.graph[idx]
        # sub_graph = graph_[:, idx]

        
        # print(f"h:{h.shape}")
        input_combined = torch.cat((output[-1, :,:-1,:], q_embed_data[:,1:,:]), -1)
        y = self.out_question_all(input_combined)
        # y = self.out(input_combined)
        y = self.m(y)

        return y

