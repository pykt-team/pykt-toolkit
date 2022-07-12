import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel,QueEmb
from pykt.utils import debug_print
from sklearn import metrics
from torch.utils.data import DataLoader
from .loss import Loss

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
    def __init__(self, num_q,num_c,emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',mlp_layer_num=1,other_config={}):
        super().__init__()
        self.model_name = "dkt_que"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.mlp_layer_num = mlp_layer_num
        self.device = device
        self.other_config = other_config
        self.qc_predict_mode_lambda = self.other_config.get('qc_predict_mode_lambda',1)
        self.qc_loss_mode_lambda = self.other_config.get('qc_loss_mode_lambda',1)
        self.loss_question_concept_lambda = self.other_config.get('loss_question_concept_lambda',1)

        
       
        self.emb_type,self.loss_mode,self.predict_mode,self.output_mode,self.attention_mode = emb_type.split("|-|")
        self.predict_next = self.output_mode == "next"#predict all question

        self.que_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type=self.emb_type,model_name=self.model_name,device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim)

        if self.emb_type in ["iekt"]:
            self.lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, batch_first=True)
            if self.attention_mode in ["attention"]:
                self.multihead_attn = nn.MultiheadAttention(self.hidden_size, 1,batch_first=True,kdim=self.hidden_size, vdim=self.hidden_size)
                #保持输出的维度和query的维度一致
        else:
            self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)

        if self.loss_mode in ["q_ccs","c_ccs","qc_ccs"]:
            self.kcs_lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.kcs_input_num = 1
        else:
            self.kcs_input_num = 0

        self.dropout_layer = nn.Dropout(dropout)
        
        if self.emb_type in ["qcaid","qcaid_h"]:
            self.h_q_merge = nn.Linear(self.hidden_size*2, self.hidden_size)
            self.h_c_merge = nn.Linear(self.hidden_size*2, self.hidden_size)

        

        if self.predict_next:
            if self.emb_type in ["iekt"]:
                if self.attention_mode in ["attention"]:
                    self.out_layer_question = MLP(self.mlp_layer_num,self.hidden_size*(4+self.kcs_input_num),1,dropout)
                    self.out_layer_concept = MLP(self.mlp_layer_num,self.hidden_size*(4+self.kcs_input_num),num_c,dropout)
                    if self.loss_mode in ["q_ccs","c_ccs","qc_ccs"]:
                        self.out_concept_classifier = MLP(self.mlp_layer_num,self.hidden_size,num_c,dropout)
                    else:
                        self.out_concept_classifier = MLP(self.mlp_layer_num,self.hidden_size,num_c,dropout)#concept classifier predict the concepts in
                else:
                    self.out_layer_question = MLP(self.mlp_layer_num,self.hidden_size*(3+self.kcs_input_num),1,dropout)
                    self.out_layer_concept = MLP(self.mlp_layer_num,self.hidden_size*(3+self.kcs_input_num),num_c,dropout)
                    if self.loss_mode in ["q_ccs","c_ccs","qc_ccs"]:
                        self.out_concept_classifier = MLP(self.mlp_layer_num,self.hidden_size,num_c,dropout)
                    else:
                        self.out_concept_classifier = MLP(self.mlp_layer_num,self.hidden_size,num_c,dropout)#concept classifier predict the concepts in 
            else:
                self.que_next_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type="qid",model_name="qid",device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim)#qid is used to predict next question
                #q_n 表示预测下一个题目而不是全部题目，知识点还是预测所有的
                self.out_layer_question = nn.Linear(self.hidden_size, 1)
                self.out_layer_concept = nn.Linear(self.hidden_size, num_c)
                self.out_concept_classifier = nn.Linear(self.hidden_size, num_c)
        else:
            if self.emb_type in ["iekt"]:
                self.out_layer_question = MLP(self.mlp_layer_num,self.hidden_size,num_q,dropout)
                self.out_layer_concept = MLP(self.mlp_layer_num,self.hidden_size,num_c,dropout)
                self.out_concept_classifier = MLP(self.mlp_layer_num,self.hidden_size,num_c,dropout)
            else:
                self.out_layer_question = nn.Linear(self.hidden_size, num_q)
                self.out_layer_concept = nn.Linear(self.hidden_size, num_c)
                self.out_concept_classifier = nn.Linear(self.hidden_size, num_c)

 
        
    def forward(self, q, c ,r,data=None):
        if self.emb_type in ["qcaid","qcaid_h"]:
            xemb,emb_q,emb_c = self.que_emb(q,c,r)
        elif self.emb_type in ["iekt"]:
            _,emb_qca,emb_qc,emb_q,emb_c = self.que_emb(q,c,r)#[batch_size,emb_size*4],[batch_size,emb_size*2],[batch_size,emb_size*1],[batch_size,emb_size*1]
           
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

        if self.loss_mode in ["q_ccs","c_ccs","qc_ccs"]:
            h_ccs,_ = self.kcs_lstm_layer(emb_q[:,1:,:])
            # print(f"h.shape is {h.shape}")
            h = torch.cat([h,h_ccs],dim=-1)#add the last hidden state of kcs lstm to the last hidden state of lstm
            # print(f"h.shape is {h.shape}")
        

        if self.predict_next:
            if self.emb_type in ['iekt']:
                seq_len = q.shape[-1]
                if self.attention_mode in ["attention"]:
                    nopeek_mask = np.triu(np.ones((seq_len, seq_len)), k=0)
                    attn_mask = torch.from_numpy(nopeek_mask).to(self.device)
                    attn_mask = attn_mask + attn_mask*(-100000)#-100000 is used to mask the attention not use -inf to avoid nan value
                   
                    attn_output, attn_output_weights = self.multihead_attn(emb_c, emb_c, emb_c,attn_mask=attn_mask)
                    # attn_output_weights = attn_output_weights[:,1:,:]
                    attn_output = attn_output[:,1:,:]
                    h = torch.cat([emb_qc_shift,attn_output,h],axis=-1)
                else:
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
        if self.loss_mode in ["q_ccs","c_ccs","qc_ccs"]:
            y_question_concepts = torch.sigmoid(self.out_concept_classifier(h_ccs))
        else:
            # 知识点分类当作多标签分类
            y_question_concepts = torch.softmax(self.out_concept_classifier(emb_q[:,1:,:]),axis=-1)
        return y_question,y_concept,y_question_concepts

class DKTQue(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',seed=0,mlp_layer_num=1,other_config={}):
        model_name = "dkt_que"
       
        debug_print(f"emb_type is {emb_type}",fuc_name="DKTQue")

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = DKTQueNet(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,mlp_layer_num=mlp_layer_num,other_config=other_config)
       
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
       
    def get_merge_loss(self,loss_question,loss_concept,loss_question_concept):
        if self.model.loss_mode=="c":
            loss = loss_concept
        elif self.model.loss_mode in ["c_cc","c_ccs"]:
            loss = (loss_concept+self.model.loss_question_concept_lambda*loss_question_concept)/(1+self.model.loss_question_concept_lambda)
        elif self.model.loss_mode=="q":
            loss = loss_question
        elif self.model.loss_mode in ["q_cc","q_ccs"]:#concept classifier
            loss = (loss_question+loss_question_concept)/2
        elif self.model.loss_mode in ["cc"]:#concept classifier
            loss = loss_question_concept
        elif self.model.loss_mode=="qc":
            loss = (loss_question+loss_concept*self.model.qc_loss_mode_lambda)/(1+self.model.qc_loss_mode_lambda)
        elif self.model.loss_mode in ["qc_cc","qc_ccs"]:#concept classifier
            loss = (loss_question+loss_concept+loss_question_concept)/3
        return loss

    def train_one_step(self,data,process=True,return_all=False):
        y,y_question,y_concept,y_question_concepts,y_qc_predict,qc_target,data_new = self.predict_one_step(data,return_details=True,process=process)
        # print(f"y_question_concepts shape is {y_question_concepts.shape}")
        loss_question = self.get_loss(y_question,data_new['rshft'],data_new['sm'])#get loss
        loss_concept = self.get_loss(y_concept,data_new['rshft'],data_new['sm'])#get loss
        
        #知识点分类当作多分类
        loss_func = Loss(self.model.other_config.get("loss_type","ce"),
                        epsilon=self.model.other_config.get("epsilon",1.0),
                        gamma=self.model.other_config.get("gamma",2)).get_loss
        loss_question_concept = loss_func(y_qc_predict,qc_target)

        print(f"loss_question is {loss_question:.4f},loss_concept is {loss_concept:.4f},loss_question_concept is {loss_question_concept:.4f}")
        
        loss = self.get_merge_loss(loss_question,loss_concept,loss_question_concept)
        if return_all:
            return y_question,y_concept,y_question_concepts,loss
        else:
            return y_question,loss

    def predict(self,dataset,batch_size,return_ts=False,process=True):
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_scores = []
            y_qc_true_list = []
            y_qc_pred_list =[]
            for data in test_loader:
                new_data = self.batch_to_device(data,process=process)
                y,y_question,y_concept,y_question_concepts,y_qc_predict,qc_target,data_new = self.predict_one_step(data,return_details=True)
                y = torch.masked_select(y, new_data['sm']).detach().cpu()
                t = torch.masked_select(new_data['rshft'], new_data['sm']).detach().cpu()
                y_trues.append(t.numpy())
                y_scores.append(y.numpy())

                y_qc_true_list.append(qc_target.detach().cpu().numpy())
                y_qc_pred_list.append(y_qc_predict.detach().cpu().numpy().argmax(axis=-1))

                
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        kc_ts = np.concatenate(y_qc_true_list, axis=0)
        kc_ps = np.concatenate(y_qc_pred_list, axis=0)


        return ps,ts,kc_ts, kc_ps

    def evaluate(self,dataset,batch_size,acc_threshold=0.5):
        ps,ts,y_qc_true_hot, y_qc_pred_hot = self.predict(dataset,batch_size=batch_size)
        kt_auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= acc_threshold else 0 for p in ps]
        kt_acc = metrics.accuracy_score(ts, prelabels)
        kc_em_acc = metrics.accuracy_score(y_qc_true_hot, y_qc_pred_hot)
        eval_result = {"auc":kt_auc,"acc":kt_acc,"kc_em_acc":kc_em_acc}
        return eval_result
        

    def predict_one_step(self,data,return_details=False,process=True):
        data_new = self.batch_to_device(data,process=process)
        if self.model.emb_type in ["iekt"]:
            y_question,y_concept,y_question_concepts = self.model(data_new['cq'].long(),data_new['cc'],data_new['cr'].long(),data=data_new)
        else:
            y_question,y_concept,y_question_concepts = self.model(data_new['q'].long(),data_new['c'],data_new['r'].long(),data=data_new)
        # print(y_question.shape,y_concept.shape)
        if self.model.predict_next:
            y_question = y_question.squeeze(-1)
        else:
            y_question = (y_question * F.one_hot(data_new['qshft'].long(), self.model.num_q)).sum(-1)

        #get y_concept
        # print(y_question.shape,y_concept.shape)
        concept_mask = torch.where(data_new['cshft'].long()==-1,False,True)
        concept_index = F.one_hot(torch.where(data_new['cshft']!=-1,data_new['cshft'],0),self.model.num_c)
        concept_sum = (y_concept.unsqueeze(2).repeat(1,1,4,1)*concept_index).sum(-1)
        concept_sum = concept_sum*concept_mask
        y_concept = concept_sum.sum(-1)/torch.where(concept_mask.sum(-1)!=0,concept_mask.sum(-1),1)



        #知识点分类当作多分类
        concept_target = data_new['cshft'][:,:,0].flatten(0,1)
        y_question_concepts = y_question_concepts.flatten(0,1)
        qc_target = concept_target[concept_target!=-1]
        y_qc_predict = y_question_concepts[concept_target!=-1,:]


        if self.model.predict_mode=="c":
            y = y_concept
        elif self.model.predict_mode=="q":
            y = y_question
        elif self.model.predict_mode in ["qc","qc_cc"]:
            y = (y_question+y_concept*self.model.qc_predict_mode_lambda)/(1+self.model.qc_predict_mode_lambda)
        if return_details:
            return y,y_question,y_concept,y_question_concepts,y_qc_predict,qc_target,data_new
        else:
            return y