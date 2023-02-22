import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


device = "cpu" if not torch.cuda.is_available() else "cuda"

class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
                Linear(self.emb_size, self.emb_size),
                ReLU(),
                Dropout(self.dropout),
                Linear(self.emb_size, self.emb_size),
                # Dropout(self.dropout),
            )
    def forward(self, in_fea):
        return self.FFN(in_fea)

def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)

def lt_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).to(dtype=torch.bool).to(device)

def pos_encode(seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0).to(device)

def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



from entmax import sparsemax, entmax15

def change_attn_scores(scores, emb_type, k_index, device,sparse_ratio=0.1,mask=None):
    """
    mask the scores and output new scores
    Args:
        scores (_type_): _description_
        emb_type (_type_): _description_
        k_index (_type_): _description_
        device (_type_): _description_
        mask (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    if emb_type.find("stride") == -1 and emb_type.find("local") == -1:
        scores.masked_fill_(mask == 0, -1e32)
        if emb_type.find("sparsemax") != -1 :
            # print(f"using attn_type: sparsemax")
            scores = sparsemax(scores, dim=-1)
        elif emb_type.find("entmax15") != -1:
            # print(f"using attn_type:entmax15")
            scores = entmax15(scores, dim=-1)
        else:
            # print(f"using attn_type: std_softmax")
            scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"scores_before:{scores}")
    # 对于每一个ai，独立的生成一个【0，1】之间的随机数（uniformly sampled）
    if emb_type.find("uniform_attn") != -1:
        scores = torch.rand(bs,head,seqlen,seqlen).to(device)
        scores.masked_fill_(mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    elif emb_type.find("sparseattn") != -1:
        # scorted_attention
        scores_a = scores[:, :, :k_index, :]
        scores_b = scores[:, :, k_index:, :].reshape(bs*head*(seqlen-k_index), -1)
        sorted_scores,sorted_idx = torch.sort(scores_b,descending=True)
        scores_t = sorted_scores[:,k_index-1:k_index].repeat(1,seqlen)
        scores_b = torch.where(scores_b - scores_t >= torch.tensor(0).to(device), scores_b, torch.tensor(-1e32).to(device)).reshape(bs,head,seqlen-k_index,-1)
        scores = torch.cat([scores_a, scores_b], dim=2)
        if emb_type == "qid_sparseattn":
            # print(f"top_k:softmatx")
            scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
        else:
            # print(f"top_k:entmax15")
            scores = entmax15(scores, dim=-1)  # BS,8,seqlen,seqlen
    elif emb_type.find("local") != -1:
        mask_a = torch.tril(torch.ones(seqlen, seqlen),-1).to(device)
        mask_b = torch.triu(torch.ones(seqlen, seqlen), -k_index).to(device)
        new_mask = mask_a * mask_b
        scores.masked_fill_(new_mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)
    elif emb_type.find("accumulative") != -1:
        # print(f"running local accumulative-attn")
        scores = torch.reshape(scores, (bs*head*seqlen,-1))
        sorted_scores,sorted_idx = torch.sort(scores,descending=True)
        acc_scores = torch.cumsum(sorted_scores,dim=1)
        acc_scores_a = torch.where(acc_scores<=0.999,acc_scores,torch.tensor(0).to(device).float())
        acc_scores_b = torch.where(acc_scores>=sparse_ratio,1,0)
        idx = torch.argmax(acc_scores_b,dim=1, keepdim=True)
        new_mask = torch.zeros(bs*head*seqlen,seqlen).to(device)
        a = torch.ones(bs*head*seqlen,seqlen).to(device)
        new_mask.scatter_(1,idx,a) 
        idx_matrix = torch.arange(seqlen).repeat(bs*seqlen*head,1).to(device)
        new_mask = torch.where(idx_matrix - idx <= 0,0,1).float()
        sorted_scores = new_mask * sorted_scores
        sorted_scores = torch.where(sorted_scores==0.0,torch.tensor(-1).to(device).float(),sorted_scores)
        tmp_scores, indices= torch.max(sorted_scores,dim=1)
        tmp_scores = tmp_scores.unsqueeze(-1).repeat(1,seqlen)
        new_scores = torch.where(tmp_scores-scores>=0,torch.tensor(-1e32).to(device).float(),scores).reshape((bs,head,seqlen,-1))
        # scores = F.softmax(new_scores, dim=-1)
        if emb_type == "qid_accumulative_attn":
            # print(f"accumulative:softmax")
            scores = F.softmax(new_scores, dim=-1)  # BS,8,seqlen,seqlen
        else:
            # print(f"accumulative:entmax15")
            scores = entmax15(new_scores, dim=-1)  # BS,8,seqlen,seqlen
    return scores