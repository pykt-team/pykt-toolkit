#!/usr/bin/env python
# coding=utf-8

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class DeepBKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256, 
            kq_same=1, final_fc_dim=512, num_attn_heads=8, seq_len=200, emb_type="qid", emb_path="", pretrain_dim=768, use_pos=True, qmatrix=None, lambda_r = 0.3):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.emb_type = emb_type
        self.model_name = "deepbkt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.model_type = self.model_name
        self.emb_type = emb_type
        self.use_pos = use_pos
        self.lambda_r = lambda_r
        self.agumentation = False
        self.bayesian = False
        self.forgetting = False
        embed_l = d_model
        
        if self.emb_type == "qid":
            self.agumentation = False
            self.bayesian = False
            self.forgetting = False
        elif self.emb_type == "agumentation":
            self.agumentation = True
            self.bayesian = False
            self.forgetting = False
        elif self.emb_type == "bayesian":
            self.agumentation = False
            self.bayesian = True
            self.forgetting = False
        elif self.emb_type == "forgetting":
            self.agumentation = False
            self.bayesian = False
            self.forgetting = True
        elif self.emb_type == "agumentation-bayesian":
            self.agumentation = True
            self.bayesian = True
            self.forgetting = False
        elif self.emb_type == "agumentation-forgetting":
            self.agumentation = True
            self.bayesian = False
            self.forgetting = True

        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
   
        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question, embed_l)
        self.qa_embed = nn.Embedding(2, embed_l)
        self.x_linear = nn.Linear(embed_l * 2, embed_l)
        self.que_embed = nn.Embedding(self.n_pid + 1, embed_l)

        if self.agumentation or self.bayesian:
            self.guess = nn.Embedding(self.n_question + 1, 1)
            self.slipping = nn.Embedding(self.n_question + 1, 1)
            self.Sigmoid = nn.Sigmoid()

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff, seq_len=seq_len, kq_same=self.kq_same, model_type=self.model_type, forgetting=self.forgetting, use_pos=self.use_pos)
        
        if not self.bayesian:
            self.out = nn.Sequential(
                nn.Linear(d_model + embed_l,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(256, 1)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(d_model + embed_l,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(256, 1), nn.Sigmoid(
                )
            )            
        self.reset()

        # self.qmatrix = nn.Embedding.from_pretrained(qmatrix, freeze=True)
        self.qmatrix_t = nn.Embedding.from_pretrained(qmatrix.permute(1,0), freeze=True)

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, q_data, target, pid_data=None, qtest=False):
        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        emb_type = self.emb_type

        if self.agumentation:
            kc_slipping = self.slipping(q_data)
            kc_slipping = torch.squeeze(self.Sigmoid(kc_slipping))
            kc_guess = self.guess(q_data)
            kc_guess = torch.squeeze(self.Sigmoid(kc_guess))
            new_target = torch.where(target == 0, torch.bernoulli(kc_slipping), 1 - torch.bernoulli(kc_guess))
            q_data = q_data.repeat(2,1,1).reshape(-1, seqlen)
            pid_data = pid_data.repeat(2,1,1).reshape(-1, seqlen)
            target = torch.stack([target, new_target]).reshape(-1,seqlen)

        # Batch First
        q_embed_data = self.q_embed(q_data)
        
        if self.n_pid > 0: # have problem id
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            final_q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder                 
            relation_que = torch.reshape(self.qmatrix_t(q_data), [batch_size*seqlen, -1]) # lookup all the kcs
            relation_que_emb = torch.mm(relation_que, self.que_embed.weight)
            que_num = torch.where(relation_que!= 0, 1, 0).sum(axis=-1).unsqueeze(-1)
            relation_que_emb = torch.reshape(relation_que_emb / que_num, [batch_size,seqlen,-1])
            new_q_embed_data = q_embed_data + relation_que_emb
            remb = target.unsqueeze(2).expand_as(new_q_embed_data)
            xemb = torch.cat((new_q_embed_data, remb), 2)
            final_qa_embed_data = self.x_linear(xemb)

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        d_output = self.model(final_q_embed_data, final_qa_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)

        m = nn.Sigmoid()
        if self.bayesian and not self.agumentation:
            print(f"using bayesian")
            kc_slipping = self.slipping(q_data)
            kc_slipping = torch.squeeze(m(kc_slipping))
            kc_guess = self.guess(q_data)
            kc_guess = torch.squeeze(m(kc_guess))
            d_ones = torch.ones(1, 1).expand_as(output).to(device)
            output = output * (1 - kc_slipping) + (d_ones - output) * kc_guess
        elif self.bayesian and not self.agumentation:
            print(f"using agumentation and bayesian")
            preds = preds.reshape(-1, batch_size, seqlen)
            d_ones = torch.ones(1, 1).expand_as(preds[0]).to(device)
            original_preds = output * (1 - kc_slipping) + (d_ones - output) * kc_guess
            preds[0, :, :] = original_preds
            output = preds.reshape(batch_size * seqlen, -1)
            
        preds = m(output)

        if not qtest and not self.agumentation:
            return preds
        elif not qtest and self.agumentation:
            preds = preds.reshape(-1, batch_size, seqlen)
            return preds[0], preds[1], c_reg_loss
        elif qtest and self.agumentation:
            preds = preds.reshape(-1, batch_size, seqlen)
            concat_q = concat_q.reshape(-1, batch_size, seqlen)
            return preds[0], concat_q[0]
        elif qtest and not self.agumentation:
            return preds, concat_q


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, seq_len, dropout, kq_same, model_type, forgetting, use_pos):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type
        self.use_pos = use_pos
        self.seq_len = seq_len

        if self.use_pos:
            self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=self.seq_len)   
        self.blocks_1 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, forgetting=forgetting)
            for _ in range(n_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, forgetting=forgetting)
            for _ in range(n_blocks*2)
        ])

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        if self.use_pos:
            q_posemb = self.position_emb(q_embed_data)
            q_embed_data = q_embed_data + q_posemb
            qa_posemb = self.position_emb(qa_embed_data)
            qa_embed_data = qa_embed_data + qa_posemb
        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data


        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # # encoder
        # # for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
        # #     y = block(mask=1, query=y, key=y, values=y) # yt^
        # # flag_first = True
        # for block in self.blocks_2:
        #     x = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever

        # encoder
        for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
            y = block(mask=1, query=y, key=y, values=y) # yt^
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False) # False: 没有FFN, 第一层只有self attention, 对应于xt^
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
                # print(x[0,0,:])
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same, forgetting):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, forgetting=forgetting)

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
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, forgetting=True):
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
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self.forgetting = forgetting

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
        gammas = self.gammas
        forgetting = self.forgetting
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas, forgetting)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, forgetting=True):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    # print(f"forgetting: {forgetting}")
    if forgetting:
        print(f"using exp forgetting")
        x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
            scores_ = scores_ * mask.float().to(device) # 结果和上一步一样
            distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
            disttotal_scores = torch.sum(
                scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
            # print(f"distotal_scores: {disttotal_scores}")
            position_effect = torch.abs(
                x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
            # bs, 8, sl, sl positive distance
            dist_scores = torch.clamp(
                (disttotal_scores-distcum_scores)*position_effect, min=0.) # score <0 时，设置为0
            dist_scores = dist_scores.sqrt().detach()
        m = nn.Softplus()
        gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
        total_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
        scores = scores * total_effect

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
