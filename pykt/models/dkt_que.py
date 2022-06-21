import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel,QueEmb


class DKTQueNet(nn.Module):
    def __init__(self, num_q,num_c,emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768,device='cpu'):
        super().__init__()
        self.model_name = "dkt_que"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.que_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type=emb_type,device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim)
        self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.hidden_size, num_q)
        
    def forward(self, q, c ,r):
        xemb = self.que_emb(q,c,r)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        return y

class DKTQue(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768,device='cpu',seed=0):
        model_name = "dkt_que"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = DKTQueNet(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device)
        self.model = self.model.to(device)
    
    def train_one_step(self,data):
        y,data_new = self.predict_one_step(data,return_details=True)
        loss = self.get_loss(y,data_new['rshft'],data_new['sm'])#get loss
        return y,loss

    def predict_one_step(self,data,return_details=False):
        data_new = self.batch_to_device(data)
        y = self.model(data_new['q'].long(),data_new['c'],data_new['r'].long())
        y = (y * F.one_hot(data_new['qshft'].long(), self.model.num_q)).sum(-1)
        if return_details:
            return y,data_new
        else:
            return y