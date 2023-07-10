import copy
import math
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum
from ..utils.utils import debug_print
torch.set_printoptions(precision=4, sci_mode=False)
torch.set_printoptions(profile="full")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=0).astype('bool') 
    return torch.from_numpy(future_mask)

def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])

def compute_corr_dict(qseqs, corr_dict):
    batch, seqlen = qseqs.shape[0], qseqs.shape[1]
    qseqs_cpu = qseqs.detach().cpu().numpy()
    corr= np.zeros((seqlen, seqlen))
    for i in range(batch):
        corr_temp= np.zeros((seqlen, seqlen))
        for j in range(seqlen):
            for k in range(seqlen):
                if qseqs_cpu[i][j] in corr_dict.keys() and qseqs_cpu[i][k] in corr_dict[qseqs_cpu[i][j]].keys():
                    corr_temp[j][k] = corr_dict[qseqs_cpu[i][j]][qseqs_cpu[i][k]]
        corr = np.concatenate((corr, corr_temp), axis=0)
    corr = np.reshape(corr, (batch+1, seqlen, seqlen))[1:,:,:]
    
    return corr

def compute_corr_matrix(qseqs, corr_matrix):
    batch, seqlen = qseqs.shape[0], qseqs.shape[1]
    qseqs_cpu = qseqs.detach().cpu().numpy()
    corr= np.zeros((seqlen, seqlen))
    for i in range(batch):
        corr_temp = corr_matrix[ np.ix_(qseqs_cpu[i], qseqs_cpu[i]) ]
        corr = np.concatenate((corr, corr_temp), axis=0)
    corr = np.reshape(corr, (batch+1, seqlen, seqlen))[1:,:,:]
    return corr

def computeTime(time_seq, time_span, batch_size, size):
    if time_seq.numel() == 0:
        seq = torch.arange(size)
        time_seq = seq.unsqueeze(0).repeat(batch_size, 1)
    #batch_size = time_seq.shape[0]
    #size = time_seq.shape[1]

    time_matrix= torch.abs(torch.unsqueeze(time_seq, axis=1).repeat(1,size,1).reshape((batch_size, size*size,1)) - \
                 torch.unsqueeze(time_seq,axis=-1).repeat(1, 1, size,).reshape((batch_size, size*size,1)))

    # time_matrix[time_matrix>time_span] = time_span
    time_matrix = time_matrix.reshape((batch_size,size,size))
    
    return time_matrix.to(device)

def attention(query, key, value, rel, l1, l2, timestamp, mask=None, dropout=None):
    """Compute scaled dot product attention.
    """
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    scores = scores.masked_fill(mask, -1e32)
    prob_attn = F.softmax(scores, dim=-1)

    rel_attn = rel.masked_fill(mask, -1e5)
    rel_attn = nn.Softmax(dim=-1)(rel_attn)
    time_stamp = torch.exp(-torch.abs(timestamp.float()))
    time_stamp = time_stamp.masked_fill(mask, -1e5)
    time_attn = F.softmax(time_stamp, dim=-1)

    # padding first row to avoid label leakage
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
    prob_attn = torch.cat([pad_zero, prob_attn[:, :, 1:, :]], dim=2)
    time_attn = torch.cat([pad_zero, time_attn[:, :, 1:, :]], dim=2)
    rel_attn = torch.cat([pad_zero, rel_attn[:, :, 1:, :]], dim=2)
    
    # Add attention by different proportions
    # prob_attn = F.softmax(prob_attn + rel_attn, dim=-1)
    prob_attn = (1-l1)*prob_attn + l1*rel_attn
    prob_attn = (1-l2)*prob_attn + l2*time_attn
    
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, rel, l1, l2, timestamp, mask=None):
        batch_size, seq_length = query.shape[:2]

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        rel = rel.unsqueeze(1).repeat(1,self.num_heads,1,1)
        timestamp = timestamp.unsqueeze(1).repeat(1,self.num_heads,1,1)
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
        out, self.prob_attn = attention(query, key, value, rel, l1, l2, timestamp, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        return out, self.prob_attn


class RKT(nn.Module):
    def __init__(self, num_c, num_q, embed_size, num_attn_layers, num_heads, batch_size, 
                  grad_clip, theta, seq_len=200, drop_prob=0.1, time_span=100000, emb_type="qid", emb_path=""):
        """Self-attentive knowledge tracing.
        Arguments:
            num_q (int): number of questions
            num_c (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            drop_prob (float): dropout probability
            theta (float): threshold for relation
        """
        super(RKT, self).__init__()
        self.model_name = "rkt"
        self.emb_type = emb_type
        self.num_c = num_c
        self.num_q = num_q
        self.embed_size = embed_size
        self.time_span = time_span
        self.grad_clip = grad_clip
        self.theta = theta
        
        if num_q <= 0:
            self.item_embeds = nn.Embedding(num_c + 1, embed_size , padding_idx=0)
        else:
            self.item_embeds = nn.Embedding(num_q + 1, embed_size , padding_idx=0)
        # self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)

        self.position_emb = CosinePositionalEmbedding(d_model=embed_size, max_len=seq_len)

        self.lin_in = nn.Linear(2*embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lin_out = nn.Linear(embed_size, 1)
        self.l1 = nn.Parameter(torch.rand(1))
        self.l2 = nn.Parameter(torch.rand(1))

    def get_inputs(self, item_inputs, label_inputs):
        item_inputs = self.item_embeds(item_inputs)
        # skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        inputs = torch.cat([item_inputs, item_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs  
        inputs[..., self.embed_size:] *= 1 - label_inputs  
        return inputs

    def get_query(self, item_ids):
        item_ids = self.item_embeds(item_ids)
        # skill_ids = self.skill_embeds(skill_ids)
        query = torch.cat([item_ids], dim=-1)
        return query

    def forward(self, dcur, rel_dict, train=True):
        q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)
        timestamp = torch.cat((t[:,0:1], tshft), dim=1)
        
        # filter the dataset only with question, no concept
        if self.num_q <= 0:
            input = q_data
        else:
            input = pid_data

        inputs = self.get_inputs(input, target)
        query = self.get_query(input)
        inputs = F.relu(self.lin_in(inputs))

        inputs_posemb = self.position_emb(inputs)
        inputs = inputs + inputs_posemb

        batch_size, seq_len = input.shape[0], input.shape[1]
        time = computeTime(timestamp, self.time_span, batch_size, seq_len) 
        mask = future_mask(inputs.size(-2)).to(device)
        
        if self.num_q > 100000:
            rel = compute_corr_dict(input, rel_dict)
        else:
            rel = compute_corr_matrix(input, rel_dict)
        rel = np.where(rel < self.theta, 0, rel) 
        rel = torch.Tensor(rel).to(device)
        
        outputs, attn  = self.attn_layers[0](query, inputs, inputs, rel, self.l1, self.l2, time, mask)
        outputs = self.dropout(outputs)
        
        for l in self.attn_layers[1:]:
            residual, attn = l(query, outputs, outputs, rel, self.l1, self.l2, time, mask)
            outputs = self.dropout(outputs + F.relu(residual))
        out = self.lin_out(outputs).squeeze(-1)
        m = nn.Sigmoid()
        pred = m(out)
        
        return pred, attn


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

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