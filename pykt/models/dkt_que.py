import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel,QueEmb


class DKTQueNet(nn.Module):
    def __init__(self, num_q,num_c,emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu'):
        super().__init__()
        self.model_name = "dkt_que"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.que_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type=self.emb_type,model_name=self.model_name,device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim)
        self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.hidden_size, num_q)

        if self.emb_type == "qaid+qc_merge":
            from .utils import transformer_FFN
            self.dnn =  transformer_FFN(self.emb_size, dropout)
            self.cpredict = nn.Linear(self.emb_size, self.num_c)
        
        
    def forward(self, q, c ,r, train=False):
        xemb = self.que_emb(q,c,r)
        # print(f"xemb.shape is {xemb.shape}")
        # h, _ = self.lstm_layer(xemb)
        # h = self.dropout_layer(h)
        # y = self.out_layer(h)
        # y = torch.sigmoid(y)

        # add predc
        # if self.emb_type == "qaid+qc_merge+predc":
        #     concept_avg = self.que_emb.get_avg_skill_emb(c)
        #     predcs = self.cpredict(h+concept_avg) # add concept_avg
        #     predcs = torch.sigmoid(predcs)
        if self.emb_type == "qaid+qc_merge+predc":
            concept_avg = self.que_emb.get_avg_skill_emb(c)
            que_emb = self.que_emb.que_emb(q)
            que_c_emb = torch.cat([concept_avg,que_emb],dim=-1)
            que_c_emb = self.que_emb.que_c_linear(que_c_emb)

            qcemb = self.dnn(que_c_emb)
            predcs = self.cpredict(qcemb) # add concept_avg
            predcs = torch.sigmoid(predcs)

            h, _ = self.lstm_layer(xemb+qcemb)
            h = self.dropout_layer(h)
            y = self.out_layer(h)
            y = torch.sigmoid(y)

        if train:
            return y, predcs
        else:
            return y

class DKTQue(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',seed=0):
        model_name = "dkt_que"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = DKTQueNet(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device)
        self.model = self.model.to(device)

        if emb_type == "qaid+qc_merge+predc":
            self.num_c = num_c
            self.device = device
            self.closs = nn.MultiLabelMarginLoss()
    
    def train_one_step(self,data,process=True):
        y,data_new,closs = self.predict_one_step(data,return_details=True,process=process,train=True)
        loss = self.get_loss(y,data_new['rshft'],data_new['sm'])#get loss
        loss = 1*loss+0.5*closs
        return y,loss

    def predict_one_step(self,data,return_details=False,process=True,train=False):
        data_new = self.batch_to_device(data,process=process)
        if train:
            y, predcs = self.model(data_new['q'].long(),data_new['c'],data_new['r'].long(),train)
            # add predcs loss
            # 预测当前题目的当前知识点
            flag = data_new['sm']==1
            masked = predcs[flag]
            # print(f"masked: {masked.shape}, predcs: {predcs.shape}")
            maxc = data_new['c'].shape[-1]
            pad = torch.ones(predcs.shape[0], predcs.shape[1], self.num_c-maxc).to(self.device)
            pad = -1 * pad
            ytrues = torch.cat([data_new['c'], pad], dim=-1).long()
            closs = self.closs(masked, ytrues[flag])
            # print(f"masked: {masked.shape}, predcs: {predcs.shape}, closs: {closs}")
            # print(ytrues[0,0:5,:])
            # print(data_new['c'][0,0:5,:])
        else:
            y = self.model(data_new['q'].long(),data_new['c'],data_new['r'].long(),train)
        y = (y * F.one_hot(data_new['qshft'].long(), self.model.num_q)).sum(-1)
        if return_details:
            if not train:
                return y,data_new
            else:
                return y,data_new,closs
        else:
            if not train:
                return y
            else:
                return y,closs