import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F  

device = "cpu" if not torch.cuda.is_available() else "cuda"  

# class KCRouteEncode(Module):
#     def __init__(self, num_level, num_c, emb_size, emb_path, pretrain_dim=768) -> None:
#         super().__init__()
#         self.num_c = num_c
#         self.emb_size = emb_size
#         self.num_level = num_level
        
#         self.content_emb = Embedding.from_pretrained(pd.read_pickle(emb_path))
#         self.cid_emb = nn.Parameter(torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True)#concept embeding
        
#         self.weight = nn.Parameter(torch.randn(num_level).to(device), requires_grad=True)
        
#     def forward(self, cs): # cs: batch_size, sequence_len, num_level
#         # add zero for padding -1
#         concept_emb_cat = torch.cat([torch.zeros(1, 20), self.cid_emb], dim=0)
#         # shift c
#         related_concepts = (cs+1).long()
#         cemb1 = concept_emb_cat[related_concepts, :]
#         cemb2 = self.content_emb(cs)
#         cemb = cemb1 + cemb2
    
#         # weights
#         indexs = torch.from_numpy(np.arange(0, self.num_level)).unsqueeze(0).expand(related_concepts.shape[0], related_concepts.shape[1], self.num_level)
#         is_avail = torch.where(related_concepts != 0, 1, 0)
#         indexs = torch.where(is_avail == 1, indexs, -1)        
#         new_weights = torch.where(is_avail > 0, self.weight[indexs], torch.tensor(float("-inf"), dtype=torch.float32))
#         # print(new_weights)
#         alphas = torch.softmax(new_weights, dim=-1).unsqueeze(-2)
#         alphas = torch.tensor(alphas, dtype=torch.float64)
#         # print(alphas.shape, cemb.shape)
#         cemb = torch.matmul(alphas, cemb).squeeze(-2)
#         return cemb
    
class RobertaEncode(Module): # OOM
    def __init__(self, emb_size, dropout, emb_path, pretrain_dim=768, add=0) -> None:
        super().__init__()
        
        embs = self.load_embs(emb_path)
        embs = torch.FloatTensor(embs).to(device)
        
        if add > 0:
            embs = torch.cat([torch.zeros(add, pretrain_dim).to(device), embs], dim=0)
        self.emb_layer = Embedding.from_pretrained(embs)
        print(f"add: {add}, emb len: {len(embs)}")
        self.l1 = Linear(pretrain_dim, pretrain_dim)
        self.dropout = Dropout(dropout)
        self.l2 = Linear(pretrain_dim, emb_size)
    
    def load_embs(self, emb_path):
        embs = []
        import json
        with open(emb_path) as fin:
            obj = json.load(fin)
        for i in range(0, len(obj)):
            embs.append(obj[str(i)])
        return embs
    
    def forward(self, qs):
        e = self.l2(self.dropout(self.l1(self.emb_layer(qs))))
        return e
        