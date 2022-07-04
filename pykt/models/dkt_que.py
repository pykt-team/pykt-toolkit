import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel,QueEmb
from pykt.utils import debug_print
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
class DKTQueNet(nn.Module):
    def __init__(self, num_q,num_c,emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu'):
        super().__init__()
        self.model_name = "dkt_que"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size

       
        self.emb_type,self.loss_mode,self.predict_mode,self.output_mode = emb_type.split("|-|")
        self.predict_next = self.output_mode == "next"#predict all question

        self.que_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type=self.emb_type,model_name=self.model_name,device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim)

        if self.emb_type in ["iekt"]:
            self.lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, batch_first=True)
        else:
            self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)

        self.dropout_layer = nn.Dropout(dropout)
        
        if self.emb_type in ["qcaid","qcaid_h"]:
            self.h_q_merge = nn.Linear(self.hidden_size*2, self.hidden_size)
            self.h_c_merge = nn.Linear(self.hidden_size*2, self.hidden_size)

        if self.predict_next:
            if self.emb_type in ["iekt"]:
                self.out_layer_question = MLP(1,self.hidden_size*3,1,dropout)
                self.out_layer_concept = MLP(1,self.hidden_size*3,num_c,dropout)
            else:
                self.que_next_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type="qid",model_name="qid",device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim)#qid is used to predict next question
                #q_n 表示预测下一个题目而不是全部题目，知识点还是预测所有的
                self.out_layer_question = MLP(1,self.hidden_size*3,1,dropout)
                self.out_layer_concept = nn.Linear(self.hidden_size, num_c)
        else:
            if self.emb_type in ["iekt"]:
                self.out_layer_question = MLP(1,self.hidden_size*3,1,dropout)
                self.out_layer_concept = MLP(1,self.hidden_size*3,num_c,dropout)
            else:
                self.out_layer_question = nn.Linear(self.hidden_size, num_q)
                self.out_layer_concept = nn.Linear(self.hidden_size, num_c)
        
        
    def forward(self, q, c ,r,data=None):
        if self.emb_type in ["qcaid","qcaid_h"]:
            xemb,emb_q,emb_c = self.que_emb(q,c,r)
        elif self.emb_type in ["iekt"]:
            _,emb_qca,emb_qc,emb_q,emb_c = self.que_emb(q,c,r)
            emb_qc_current = emb_qc[:,:-1,:]
            emb_qc_shift = emb_qc[:,1:,:]
            emb_qca_current = emb_qca[:,:-1,:]
            emb_qca_shift = emb_qca[:,1:,:]
        else:
            xemb = self.que_emb(q,c,r)

        if self.emb_type in ["iekt"]:
            h, _ = self.lstm_layer(emb_qca_current)
        else:
        # print(f"xemb.shape is {xemb.shape}")
            h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)

        if self.predict_next:
            if self.emb_type in ['iekt']:
                h = torch.cat([emb_qc_shift,h],axis=-1)
            else:
                xemb_next = self.que_next_emb(data['qshft'],data['cshft'],data['rshft'])
                h = self.h_q_merge(torch.cat([xemb_next,h],axis=-1))

        if self.emb_type == "qcaid":
            h_q = h
            h_c = h
        elif self.emb_type == "qcaid_h":
            h_q = self.h_q_merge(torch.cat([h,emb_q],dim=-1))
            h_c = self.h_c_merge(torch.cat([h,emb_c],dim=-1))
        elif self.emb_type == "iekt":
            h_q = h#[batch_size,seq_len,hidden_size*3]
            h_c = h#[batch_size,seq_len,hidden_size*3]
        elif self.emb_type == "qid":
            h_q = h
            h_c = h

        y_question = torch.sigmoid(self.out_layer_question(h_q))
        y_concept = torch.sigmoid(self.out_layer_concept(h_c))
        return y_question,y_concept

class DKTQue(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',seed=0):
        model_name = "dkt_que"
       
        debug_print(f"emb_type is {emb_type}",fuc_name="DKTQue")

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = DKTQueNet(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device)
        
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
       

    
    def train_one_step(self,data,process=True):
        y_question,y_concept,data_new = self.predict_one_step(data,return_details=True,process=process)
        loss_question = self.get_loss(y_question,data_new['rshft'],data_new['sm'])#get loss
        loss_concept = self.get_loss(y_concept,data_new['rshft'],data_new['sm'])#get loss
        if self.model.loss_mode=="c":
            loss = loss_concept
        elif self.model.loss_mode=="q":
            loss = loss_question
        elif self.model.loss_mode=="qc":
            loss = (loss_question+loss_concept)/2
        return y_question,loss

    def predict_one_step(self,data,return_details=False,process=True):
        data_new = self.batch_to_device(data,process=process)
        if self.model.emb_type in ["iekt"]:
            y_question,y_concept = self.model(data_new['cq'].long(),data_new['cc'],data_new['cr'].long(),data=data_new)
        else:
            y_question,y_concept = self.model(data_new['q'].long(),data_new['c'],data_new['r'].long(),data=data_new)
        # print(y_question.shape,y_concept.shape)
        if self.model.predict_next:
            y_question = y_question.squeeze(-1)
        else:
            y_question = (y_question * F.one_hot(data_new['qshft'].long(), self.model.num_q)).sum(-1)
        # print(y_question.shape,y_concept.shape)
        concept_mask = torch.where(data_new['cshft'].long()==-1,False,True)
        concept_index = F.one_hot(torch.where(data_new['cshft']!=-1,data_new['cshft'],0),self.model.num_c)
        concept_sum = (y_concept.unsqueeze(2).repeat(1,1,4,1)*concept_index).sum(-1)
        concept_sum = concept_sum*concept_mask
        y_concept = concept_sum.sum(-1)/torch.where(concept_mask.sum(-1)!=0,concept_mask.sum(-1),1)

        if return_details:
            return y_question,y_concept,data_new
        else:
            if self.model.predict_mode=="c":
                y = y_concept
            elif self.model.predict_mode=="q":
                y = y_question
            elif self.model.predict_mode=="qc":
                y = (y_question+y_concept)/2

            return y