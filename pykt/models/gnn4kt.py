from __future__ import print_function, division
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
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

class LSTM4Graph(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM4Graph, self).__init__()
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

    def __init__(self, n_question, n_pid, embed_l, hidden_size, dropout, num_layers=5, seq_len=200, final_fc_dim=512, final_fc_dim2=256, emb_type="iekt", graph=None, mlp_layer_num=1, sigma=0.1, topk=10, num_attn_heads=8, d_ff=256, emb_path="", kq_same=1, pretrain_dim=768): #
        super(GNN4KT, self).__init__()

        self.model_name = "gnn4kt"
        self.emb_type = emb_type
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_q = n_pid
        self.num_c = n_question
        self.sigma = sigma
        self.emb_size = embed_l
        self.model_type = self.model_name
        self.kq_same = kq_same
        self.seq_len = seq_len

        self.que_emb = nn.Embedding(self.num_q+1, embed_l)#question embeding
        self.concept_emb = nn.Parameter(torch.randn(self.num_c+1, embed_l).to(device), requires_grad=True)#concept embeding

        self.qa_embed = nn.Embedding(2, embed_l)

        if emb_type.find("lstm") != -1:
            self.lstm_layer = nn.LSTM(input_size=embed_l, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.out_question_all = MLP(mlp_layer_num, hidden_size+embed_l,self.num_q, self.dropout)
        if emb_type.find("graph") != -1:
            self.graph = graph.to_dense()
        # GCN for inter information
            self.gc = nn.ModuleList([GNNLayer(hidden_size, hidden_size) if _ != 0 else GNNLayer(embed_l, hidden_size) for _ in range(num_layers)])
            self.matrix_a = nn.Parameter(torch.FloatTensor(self.num_q+1, self.seq_len))
            self.matrix_b = nn.Parameter(torch.FloatTensor(self.num_q+1, self.seq_len))
            torch.nn.init.xavier_uniform_(self.matrix_a)
            torch.nn.init.xavier_uniform_(self.matrix_b)
            if emb_type.find("lstm") != -1:
                self.lstm4graph = LSTM4Graph(input_size=embed_l, hidden_size=hidden_size, num_layers=num_layers)
            else:
                self.trf4graph = Architecture(n_question=n_question, n_blocks=self.num_layers, n_heads=num_attn_heads, dropout=dropout,
                                        d_model=hidden_size, d_feature=hidden_size / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)     
        if emb_type.find("trf") != -1:
            # Architecture Object. It contains stack of attention block
            self.model = Architecture(n_question=n_question, n_blocks=self.num_layers, n_heads=num_attn_heads, dropout=dropout,
                                        d_model=hidden_size, d_feature=hidden_size / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)            

            self.out = nn.Sequential(
            nn.Linear(hidden_size + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
            )
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
        q, c, r = dcur["qseqs"].long().to(device), dcur["cseqs"].long().to(device), dcur["rseqs"].long().to(device)
        qshft, cshft, rshft = dcur["shft_qseqs"].long().to(device), dcur["shft_cseqs"].long().to(device), dcur["shft_rseqs"].long().to(device)
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)
        emb_q = self.que_emb(pid_data)#[batch,max_len-1,emb_size]
        emb_c = self.get_avg_skill_emb(q_data)#[batch,max_len-1,emb_size]
        q_embed_data = emb_q + emb_c
        qa_embed_data = self.qa_embed(target)

        bs, seqlen = pid_data.size(0), pid_data.size(1)

        #利用lstm得到每层的emb
        if self.emb_type.find("lstm") != -1:
            if self.emb_type.find("graph") != -1:
                output = self.lstm4graph(q_embed_data) #[num_layers, batch_size, seqlen, hidden_size]
            else:
                output, _ = self.lstm_layer(qa_embed_data) #[num_layers, batch_size, seqlen, hidden_size]
        elif self.emb_type.find("trf") != -1:
            if self.emb_type.find("graph") != -1:
                last_output, output = self.trf4graph(q_embed_data, q_embed_data) #[num_layers, batch_size, seqlen, hidden_size]
            else:
                output, _ = self.model(q_embed_data, qa_embed_data) #[num_layers, batch_size, seqlen, hidden_size]
        if self.emb_type.find("graph") != -1:
            for i in range(self.num_layers):
                # print(f"matrix:{self.matrix_a.shape}")
                sub_graph_1 = torch.mm(self.graph, self.matrix_a).permute(1,0)
                sub_graph_2 = torch.mm(self.graph, self.matrix_b)
                sub_graph = torch.matmul(sub_graph_1, sub_graph_2).repeat(bs, 1, 1)

                if i == 0:
                    h = self.gc[i](q_embed_data, sub_graph)#.reshape(bs*self.num_q,-1)[idx]
                elif i == self.num_layers - 1:
                    h = self.gc[i]((1-self.sigma)*h + self.sigma*output[i-1], sub_graph, active=False)
                else:
                    h = self.gc[i]((1-self.sigma)*h + self.sigma*output[i-1], sub_graph)
            if self.emb_type.find("lstm") != -1:
                output, _ = self.lstm_layer(h + qa_embed_data)
            elif self.emb_type.find("trf") != -1:
                output, _ = self.model(q_embed_data, qa_embed_data+h)
        if self.emb_type.find("lstm") != -1:
            input_combined = torch.cat((output[:,:-1,:], q_embed_data[:,1:,:]), -1)
            y = self.out_question_all(input_combined)
        elif self.emb_type.find("trf") != -1:
            input_combined = torch.cat((output, q_embed_data), -1)
            y = self.out(input_combined).squeeze(-1)
        y = self.m(y)

        return y

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'gnn4kt'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        all_x = []
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
            all_x.append(x)
        all_x = torch.stack(all_x,dim=0)
        return x, all_x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)

