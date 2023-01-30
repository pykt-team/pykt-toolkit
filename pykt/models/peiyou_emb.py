import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from .pretrain_utils import RobertaEncode  

device = "cpu" if not torch.cuda.is_available() else "cuda"  

class QuestionEncoder(Module):
    def __init__(self, num_q, emb_type, emb_size, dropout, emb_paths, pretrain_dim=768) -> None:
        super().__init__()
        
        self.emb_type = emb_type
        
        self.id_encoder = Embedding(num_q, emb_size)
        self.content_encoder = RobertaEncode(emb_size, dropout, emb_paths['content'], pretrain_dim)
        self.analysis_encoder = RobertaEncode(emb_size, dropout, emb_paths['analysis'], pretrain_dim)
        self.type_encoder = Embedding(2, emb_size)
        
        if self.emb_type.startswith("qidcat"):
            self.reduction = Sequential(
                Linear(emb_size*4, emb_size*2), Dropout(dropout), ReLU(),
                Linear(emb_size*2, emb_size))
        
    def forward(self, qs, types):
        qid_emb = self.id_encoder(qs)
        cont_emb = self.content_encoder(qs)
        ana_emb = self.analysis_encoder(qs)
        type_emb = self.type_encoder(types)
        
        if self.emb_type.startswith("qidadd"):
            return qid_emb + cont_emb + ana_emb + type_emb
        elif self.emb_type.startswith("qidmul"):
            vt = torch.mul(qid_emb, cont_emb, ana_emb, type_emb)
            return vt
        elif self.emb_type.startswith("qidcat"):
            emb = torch.cat([qid_emb, cont_emb, ana_emb, type_emb], dim=-1)
            emb = self.reduction(emb)
            return emb
        return 
    
class KCRouteEncoder(Module):
    def __init__(self, num_level, num_c, emb_type, emb_size, dropout, emb_path, pretrain_dim=768) -> None:
        super().__init__()
        self.num_c = num_c
        self.emb_size = emb_size
        self.num_level = num_level
        self.emb_type = emb_type
        
        if self.emb_type.find("rc") != -1:
            self.content_emb = RobertaEncode(emb_size, dropout, emb_path, pretrain_dim, 2)
            if not self.emb_type.endswith("rcon"):
                self.cid_emb = nn.Parameter(torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True)#concept embeding
            
            if self.emb_type.endswith("rcidcat"):
                self.reduction = Linear(emb_size*2, emb_size)
        
            self.weight = nn.Parameter(torch.randn(num_level).to(device), requires_grad=True)
        # elif self.emb_type.endswith("onlycid"): # don't equal to akt peiyou
        #     self.cid_emb = nn.Embedding(self.num_c, self.emb_size)
        else:
            self.content_emb = RobertaEncode(emb_size, dropout, emb_path, pretrain_dim, 2)
            if self.emb_type.endswith("cadd") or self.emb_type.endswith("cmul") or self.emb_type.endswith("ccat"):
                self.cid_emb = nn.Embedding(self.num_c, self.emb_size)
            if self.emb_type.endswith("ccat"):
                self.reduction = Linear(emb_size*2, emb_size)
            
        
    def forward(self, croutes, tailcs): # cs: batch_size, sequence_len, num_level
        # if self.emb_type.endswith("onlycid"):
        #     return self.cid_emb(tailcs)
        if self.emb_type.endswith("onlycon"):
            return self.content_emb(tailcs)
        if self.emb_type.endswith("cadd"):
            return self.cid_emb[tailcs] + self.content_emb(tailcs)
        if self.emb_type.endswith("cmul"):
            return torch.mul(self.cid_emb[tailcs], self.content_emb(tailcs))
        if self.emb_type.endswith("ccat"):
            ccat = torch.cat([self.cid_emb[tailcs], self.content_emb(tailcs)], dim=-1)
            return self.reduction(ccat)
        
        # add zero for padding -1
        if not self.emb_type.endswith("rcon"):
            concept_emb_cat = torch.cat([torch.zeros(2, self.emb_size).to(device), self.cid_emb], dim=0)
            # shift c
            related_concepts = (croutes+2).long() # 0->-2->kc route pad, 1->-1->sequence pad
            cemb1 = concept_emb_cat[related_concepts, :]
        
        cemb2 = self.content_emb(croutes+2) ### TODO croutes -2和-1
        
        if self.emb_type.endswith("rcid"): # only route cid
            cemb = cemb1
        elif self.emb_type.endswith("rcon"): # only route content
            cemb = cemb2
        elif self.emb_type.endswith("rcadd"):
            cemb = cemb1 + cemb2
        elif self.emb_type.endswith("rcmul"):
            cemb = torch.mul(cemb1, cemb2)
        elif self.emb_type.endswith("rccat"):
            cemb = torch.cat([cemb1, cemb2], dim=-1)
            cemb = self.reduction(cemb)
    
        # weights
        indexs = torch.from_numpy(np.arange(0, self.num_level)).unsqueeze(0).expand(related_concepts.shape[0], related_concepts.shape[1], self.num_level).to(device)
        is_avail = torch.where(related_concepts != 0, 1, 0)
        indexs = torch.where(is_avail == 1, indexs, -1)      
        new_weights = torch.where(is_avail > 0, self.weight[indexs], torch.tensor(float("-inf"), dtype=torch.float32).to(device)) 
        # print(new_weights.shape)
        # assert False
        alphas = torch.softmax(new_weights, dim=-1).unsqueeze(-2)
        # alphas = torch.tensor(alphas, dtype=torch.float32)
        # print(alphas.shape, cemb.shape)
        cemb = torch.matmul(alphas, cemb).squeeze(-2)
        return cemb
    
def mean_max_mul(embs):
    onedim = []
    for emb in embs:
        onedim.append(emb.reshape(-1).unsqueeze(0))
    merge = torch.cat(onedim, dim=0)
    
    mean = torch.mean(merge, dim=0).reshape_as(embs[0])
    max = torch.max(merge, dim=0)[0].reshape_as(embs[0])
    mul = torch.mul(embs[0], embs[1])
    for k in range(2, len(embs)):
        mul = torch.mul(mul, embs[k])
    res = torch.cat([mean, max, mul], dim=-1)
    return res