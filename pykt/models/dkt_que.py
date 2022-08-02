import os
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel,QueEmb
from pykt.utils import debug_print
from sklearn import metrics
from torch.utils.data import DataLoader
from .loss import Loss
from scipy.special import softmax

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



def get_outputs(self,emb_qc_shift,h,data,add_name=""):
    if "an" in self.output_mode:
        #next predict
        h_next = torch.cat([emb_qc_shift,h],axis=-1)
        y_question_next = torch.sigmoid(self.out_question_next(h_next))
        y_concept_next = torch.sigmoid(self.out_concept_next(h_next))
        #all predict
        y_question_all = torch.sigmoid(self.out_question_all(h))
        y_concept_all = torch.sigmoid(self.out_concept_all(h))
        outputs = {"y_question_next"+add_name:y_question_next,"y_concept_next"+add_name:y_concept_next,
                    "y_question_all"+add_name:y_question_all,"y_concept_all"+add_name:y_concept_all}
        
        outputs["y_question_next"+add_name] = outputs["y_question_next"+add_name].squeeze(-1)
        outputs["y_concept_next"+add_name] = self.get_avg_fusion_concepts(outputs["y_concept_next"+add_name],data['cshft'])
        outputs["y_question_all"+add_name] = (outputs["y_question_all"+add_name] * F.one_hot(data['qshft'].long(), self.num_q)).sum(-1)
        outputs["y_concept_all"+add_name] = self.get_avg_fusion_concepts(outputs["y_concept_all"+add_name],data['cshft'])
    return outputs


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
        self.input_attn = False
        if self.input_attn:
            self.qc_attn = nn.MultiheadAttention(self.hidden_size*2, num_heads=1,batch_first=True)
            self.qca_attn = nn.MultiheadAttention(self.hidden_size*4, num_heads=1,batch_first=True)
            self.qc_attn_linear = nn.Linear(self.hidden_size*4,self.hidden_size*2)

        self.contrast_mode = self.other_config.get("contrast_mode")#cm_v1,cm_v2
        self.cm_add_name = f"_{self.contrast_mode}"
        

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
        
        if "an" in self.output_mode:#all and next merge, qc-c_an#all 使用qc，next使用 c
            if self.emb_type in ["iekt"]:
                if self.attention_mode in ["attention"]:
                    self.out_question_next = MLP(self.mlp_layer_num,self.hidden_size*(4+self.kcs_input_num),1,dropout)
                    self.out_concept_next = MLP(self.mlp_layer_num,self.hidden_size*(4+self.kcs_input_num),num_c,dropout)
                else:
                    self.out_question_next = MLP(self.mlp_layer_num,self.hidden_size*(3+self.kcs_input_num),1,dropout)
                    self.out_concept_next = MLP(self.mlp_layer_num,self.hidden_size*(3+self.kcs_input_num),num_c,dropout)
                self.out_question_all = MLP(self.mlp_layer_num,self.hidden_size,num_q,dropout)
                self.out_concept_all = MLP(self.mlp_layer_num,self.hidden_size,num_c,dropout)

            else:
                pass
        elif self.output_mode in ["an_irt"]:
            trainable = self.other_config.get("irt_w_trainable",1)==1
            # self.irt_w = nn.Parameter(torch.randn(3).to(device), requires_grad=True)
            self.irt_w = nn.Parameter(torch.tensor([1.0,1.0,1.0]).to(device), requires_grad=trainable)
            # self.irt_w = nn.Parameter(torch.tensor([0.75,0.75,1.5]).to(device), requires_grad=True)
        else:#单独预测模式
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
        
           
    def attn_help(self,seq_len,emb,x):
        nopeek_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        attn_mask = torch.from_numpy(nopeek_mask).to(self.device)
        attn_mask = attn_mask + attn_mask*(-100000)#-100000 is used to mask the attention not use -inf to avoid nan value
        attn_output, _ = emb(x, x, x,attn_mask=attn_mask)
        return attn_output
        
    def get_avg_fusion_concepts(self,y_concept,cshft):
        """获取知识点 fusion 的预测结果
        """
        max_num_concept = cshft.shape[-1]
        concept_mask = torch.where(cshft.long()==-1,False,True)
        concept_index = F.one_hot(torch.where(cshft!=-1,cshft,0),self.num_c)
        concept_sum = (y_concept.unsqueeze(2).repeat(1,1,max_num_concept,1)*concept_index).sum(-1)
        concept_sum = concept_sum*concept_mask#remove mask
        y_concept = concept_sum.sum(-1)/torch.where(concept_mask.sum(-1)!=0,concept_mask.sum(-1),1)
        return y_concept

    def forward(self, q, c ,r,data=None):
        seq_len = q.shape[-1]
        if self.emb_type in ["qcaid","qcaid_h"]:
            xemb,emb_q,emb_c = self.que_emb(q,c,r)
        elif self.emb_type in ["iekt"]:
            _,emb_qca,emb_qc,emb_q,emb_c = self.que_emb(q,c,r)#[batch_size,emb_size*4],[batch_size,emb_size*2],[batch_size,emb_size*1],[batch_size,emb_size*1]
            emb_qc_shift = emb_qc[:,1:,:]
            emb_qca_current = emb_qca[:,:-1,:]
 
        else:
            xemb = self.que_emb(q,c,r)

        if self.input_attn:
            emb_qc_attn = self.attn_help(seq_len,self.qc_attn,emb_qc)
            emb_qc += emb_qc_attn
            # emb_qc = torch.cat([torch.sigmoid(emb_qc)*emb_qc_attn,emb_qc],axis=-1)
            # emb_qc = self.qc_attn_linear(emb_qc)
            
            emb_qca_attn = self.attn_help(seq_len,self.qca_attn,emb_qca)
            emb_qca+=emb_qca_attn
          
            emb_qc_shift = emb_qc[:,1:,:]
            emb_qca_current = emb_qca[:,:-1,:]
            

        if self.emb_type in ["iekt"]:
            h_raw, _ = self.lstm_layer(emb_qca_current)
        else:
        # print(f"xemb.shape is {xemb.shape}")
            h_raw, _ = self.lstm_layer(xemb)

        h = self.dropout_layer(h_raw)
        if self.contrast_mode in ['cm_v1','cm_v2']:
            h2 = nn.Dropout(0.8)(h_raw)

        if self.loss_mode in ["q_ccs","c_ccs","qc_ccs"]:
            h_ccs,_ = self.kcs_lstm_layer(emb_q[:,1:,:])
            # print(f"h.shape is {h.shape}")
            h = torch.cat([h,h_ccs],dim=-1)#add the last hidden state of kcs lstm to the last hidden state of lstm
            # print(f"h.shape is {h.shape}")

        if "an" in self.output_mode:
            outputs = get_outputs(self,emb_qc_shift,h,data,add_name="")
            #h2
            if self.contrast_mode in ['cm_v1','cm_v2']:
                outputs2 = get_outputs(self,emb_qc_shift,h2,data,add_name=self.cm_add_name)
                outputs.update(outputs2)

            if "cc" in self.loss_mode:
                y_question_concepts_next,y_question_concepts_all = None,None
                outputs['y_question_concepts_next'] = y_question_concepts_next
                outputs['y_question_concepts_all'] = y_question_concepts_all
           
            return outputs

        else:
            if self.predict_next:
                if self.emb_type in ['iekt']:
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
            outputs = {"y_question":y_question,"y_concept":y_concept}

            if self.loss_mode in ["q_ccs","c_ccs","qc_ccs"]:
                y_question_concepts = torch.sigmoid(self.out_concept_classifier(h_ccs))
                outputs['y_question_concepts'] = y_question_concepts
            elif self.loss_mode in ["q_cc","c_cc","qc_cc","qc_cc_dyn","cc","cc_dyn"]:
                # 知识点分类当作多标签分类
                y_question_concepts = torch.softmax(self.out_concept_classifier(emb_q[:,1:,:]),axis=-1)
                outputs['y_question_concepts'] = y_question_concepts
            return outputs

class DKTQue(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',seed=0,mlp_layer_num=1,other_config={}):
        model_name = "dkt_que"
       
        debug_print(f"emb_type is {emb_type}",fuc_name="DKTQue")

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = DKTQueNet(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,mlp_layer_num=mlp_layer_num,other_config=other_config)
       
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.eval_result = {}
    
    def get_merge_loss(self,loss_question,loss_concept,loss_question_concept,loss_mode):
        if loss_mode in ["c","c_fr"]:
            loss = loss_concept
        elif loss_mode in ["c_cc","c_ccs"]:
            loss = (loss_concept+self.model.loss_question_concept_lambda*loss_question_concept)/(1+self.model.loss_question_concept_lambda)
        elif loss_mode in ["q","q_fr"]:
            loss = loss_question
        elif loss_mode in ["q_cc","q_ccs"]:#concept classifier
            loss = (loss_question+loss_question_concept)/2
        elif loss_mode in ["cc"]:#concept classifier
            loss = loss_question_concept
        elif loss_mode in ["qc","qc_fr"]:
            loss = (loss_question+loss_concept*self.model.qc_loss_mode_lambda)/(1+self.model.qc_loss_mode_lambda)
        elif loss_mode in ["qc_cc","qc_ccs"]:#concept classifier
            loss = (loss_question+loss_concept+loss_question_concept)/3
        elif loss_mode in ['cc_dyn',"qc_cc_dyn"]:
            acc_kt = self.eval_result.get("acc",1)
            acc_kc = self.eval_result.get("kc_em_acc",1)
            c_dyn_a = self.model.other_config.get("c_dyn_a",0)
            c_dyn_b = self.model.other_config.get("c_dyn_b",0)
            alpha_kt = (acc_kc+c_dyn_a)/(acc_kt+acc_kc+c_dyn_a+c_dyn_b)
            alpha_kc = (acc_kt+c_dyn_b)/(acc_kt+acc_kc+c_dyn_a+c_dyn_b)
            print(f"acc_kt={acc_kt},acc_kc={acc_kc},alpha_kt={alpha_kt},alpha_kc={alpha_kc},c_dyn_a={c_dyn_a},c_dyn_b={c_dyn_b}")
            
            if loss_mode in ['cc_dyn']:
                loss_kt = loss_concept
            elif loss_mode in ['qc_cc_dyn']:
                loss_kt = (loss_question+loss_concept)

            loss = alpha_kt*loss_kt + alpha_kc*loss_question_concept
        
        return loss
   


    def train_one_step(self,data,process=True,return_all=False):
        
        outputs,data_new = self.predict_one_step(data,return_details=True,process=process)
        if "an" in self.model.output_mode:
            #all 
            loss_question_all = self.get_loss(outputs['y_question_all'],data_new['rshft'],data_new['sm'])#question level loss
            loss_concept_all = self.get_loss(outputs['y_concept_all'],data_new['rshft'],data_new['sm'])#kc level loss
            #next
            loss_question_next = self.get_loss(outputs['y_question_next'],data_new['rshft'],data_new['sm'])#question level loss
            loss_concept_next = self.get_loss(outputs['y_concept_next'],data_new['rshft'],data_new['sm'])#kc level loss
            
            loss_question_concept = -1
        else:
            loss_question = self.get_loss(outputs['y_question'],data_new['rshft'],data_new['sm'])#question level loss
            loss_concept = self.get_loss(outputs['y_concept'],data_new['rshft'],data_new['sm'])#kc level loss
            if "cc" in self.model.loss_mode:
                # 知识点分类当作多分类
                loss_func = Loss(self.model.other_config.get("loss_type","ce"),
                                    epsilon=self.model.other_config.get("epsilon",1.0),
                                    gamma=self.model.other_config.get("gamma",2)).get_loss
                loss_question_concept = loss_func(outputs['y_qc_predict'],outputs['qc_target'])#question concept level loss
            else:
                loss_question_concept = -1

        if "an" in self.model.output_mode:
            all_loss_mode,next_loss_mode =  self.model.loss_mode.replace("_dyn","").split("_")[0].split("-")
            loss_all = self.get_merge_loss(loss_question_all,loss_concept_all,loss_question_concept,all_loss_mode)   
            loss_next = self.get_merge_loss(loss_question_next,loss_concept_next,loss_question_concept,next_loss_mode)
            loss_same = F.mse_loss(outputs['y_qc_all'],outputs['y_concept_next'])
            loss_raw_kt = self.get_loss(outputs['y'],data_new['rshft'],data_new['sm'])
            if self.model.contrast_mode in ['cm_v1','cm_v2']:
                loss_raw_kt2 = self.get_loss(outputs['y'+self.model.cm_add_name],data_new['rshft'],data_new['sm'])
                loss_cm = F.mse_loss(outputs['y'],outputs['y'+self.model.cm_add_name])*100
            else:
                loss_raw_kt2 = 0
                loss_cm = 0
            if self.model.output_mode=="an_irt":
                loss_kt = self.get_loss(outputs['y'],data_new['rshft'],data_new['sm'])#question level loss
                l2 = self.model.other_config.get("l2",1e-5)
                w_norm = (self.model.irt_w ** 2.).sum() * l2
                loss = loss_kt + w_norm #+ loss_all#+loss_next
                print(f"loss={loss:.3f},loss_kt={loss_kt:.3f},w_norm={w_norm:.3f},self.model.irt_w is {self.model.irt_w}")
            else:
                if "dyn" in self.model.loss_mode:
                    dyn_a = self.model.other_config.get("dyn_a",0)
                    dyn_b = self.model.other_config.get("dyn_b",0)
                    auc_all =  self.eval_result.get("y_qc_all_kt_auc",1)
                    auc_next =  self.eval_result.get("y_concept_next_kt_auc",1)
                    auc_all,auc_next = softmax(np.array([auc_all,auc_next])/self.model.other_config.get("temperature",0.003))
                    alpha_all = (auc_next+dyn_a)/(auc_all+dyn_a+auc_next+dyn_b)
                    alpha_next = (auc_all+dyn_b)/(auc_all+dyn_a+auc_next+dyn_b)
                    loss = alpha_all*loss_all + alpha_next*loss_next
                    print(f"auc_all={auc_all},auc_next={auc_next},alpha_all={alpha_all},alpha_next={alpha_next},dyn_a={dyn_a},dyn_b={dyn_b}")
                else:
                    loss_next_lambda = self.model.other_config.get("loss_next_lambda",0.5)
                    loss_all_lambda = self.model.other_config.get("loss_all_lambda",0.5)
                    loss_same_lambda = self.model.other_config.get("loss_same_lambda",0)
                    loss = loss_all*loss_all_lambda+loss_next*loss_next_lambda + loss_same*loss_same_lambda
                    loss = loss/(loss_next_lambda+loss_all_lambda+loss_same_lambda)
                    loss = loss + loss_raw_kt2 + loss_cm
                print(f"loss={loss:.3f},loss_all={loss_all:.3f},loss_next={loss_next:.3f},loss_same={loss_same:.3f},loss_raw_kt={loss_raw_kt:.3f},loss_raw_kt2={loss_raw_kt2:.3f},loss_cm={loss_cm:.3f}")
            return outputs['y'],loss#y_question没用
        else:
            print(f"loss_question is {loss_question:.3f},loss_concept is {loss_concept:.3f},loss_question_concept is {loss_question_concept:.3f}")
            
            loss = self.get_merge_loss(loss_question,loss_concept,loss_question_concept,loss_mode=self.model.loss_mode)

            return outputs['y'],loss

    def predict(self,dataset,batch_size,return_ts=False,process=True):
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_qc_true_list = []
            y_qc_pred_list =[]
            y_pred_dict = {}
            for data in test_loader:
                new_data = self.batch_to_device(data,process=process)
                outputs,data_new = self.predict_one_step(data,return_details=True)
               
                for key in outputs:
                    # print(f"key is {key},shape is {outputs[key].shape}")
                    if not key.startswith("y") or key in ['y_qc_predict']:
                        continue
                    elif key not in y_pred_dict:
                       y_pred_dict[key] = []
                    # print(f"outputs is {outputs}")
                    y = torch.masked_select(outputs[key], new_data['sm']).detach().cpu()#get label
                    y_pred_dict[key].append(y.numpy())
                
                t = torch.masked_select(new_data['rshft'], new_data['sm']).detach().cpu()
                y_trues.append(t.numpy())

                if "cc" in self.model.loss_mode:
                    y_qc_true_list.append(outputs['qc_target'].detach().cpu().numpy())
                    y_qc_pred_list.append(outputs['y_qc_predict'].detach().cpu().numpy().argmax(axis=-1))
        
        results = y_pred_dict
        for key in results:
            # print(f"y type is {key}")
            results[key] = np.concatenate(results[key], axis=0)
            # print(f"{key} shape is {results[key].shape}")
        # print(f"results is {results}")
                
        ts = np.concatenate(y_trues, axis=0)
        results['ts'] = ts
        # print(f"ts shape is {ts.shape}")

        if "cc" in self.model.loss_mode:
            kc_ts = np.concatenate(y_qc_true_list, axis=0)
            kc_ps = np.concatenate(y_qc_pred_list, axis=0)
            results['kc_ts'] = kc_ts
            results['kc_ps'] = kc_ps
        
        return results

    def evaluate(self,dataset,batch_size,acc_threshold=0.5):
        # ps,ts,y_qc_true_hot, y_qc_pred_hot = self.predict(dataset,batch_size=batch_size)
        results = self.predict(dataset,batch_size=batch_size)
        eval_result = {}
        ts = results["ts"]
        for key in results:
            if not key.startswith("y") or key in ['y_qc_predict']:
                pass
            else:
                ps = results[key]
                kt_auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
                prelabels = [1 if p >= acc_threshold else 0 for p in ps]
                kt_acc = metrics.accuracy_score(ts, prelabels)
                if key!="y":
                    eval_result["{}_kt_auc".format(key)] = kt_auc
                    eval_result["{}_kt_acc".format(key)] = kt_acc
                else:
                    eval_result["auc"] = kt_auc
                    eval_result["acc"] = kt_acc
        
        if "cc" in self.model.loss_mode:
            kc_em_acc = metrics.accuracy_score(results['kc_ts'], results['kc_ps'])
        else:
            kc_em_acc = 0
        eval_result["kc_em_acc"] = kc_em_acc
        self.eval_result = eval_result
        return eval_result
        
    def get_qc_predict_result(self,y_question_concepts,data_new):
        #知识点分类当作多分类
        concept_target = data_new['cshft'][:,:,0].flatten(0,1)
        y_question_concepts = y_question_concepts.flatten(0,1)
        qc_target = concept_target[concept_target!=-1]
        y_qc_predict = y_question_concepts[concept_target!=-1,:]
        return qc_target,y_qc_predict

    def get_merge_y(self,y_question,y_concept,predict_mode):
        if predict_mode=="c":
            y = y_concept
        elif predict_mode=="q":
            y = y_question
        elif predict_mode in ["qc","qc_cc"]:
            y = (y_question+y_concept*self.model.qc_predict_mode_lambda)/(1+self.model.qc_predict_mode_lambda)
        return y

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        data_new = self.batch_to_device(data,process=process)
        if self.model.emb_type in ["iekt"]:
            outputs = self.model(data_new['cq'].long(),data_new['cc'],data_new['cr'].long(),data=data_new)
        else:
            outputs = self.model(data_new['q'].long(),data_new['c'],data_new['r'].long(),data=data_new)
                
        # print(y_question.shape,y_concept.shape)
        if return_raw:#return raw probability, for future reward
            return outputs,data_new
        else:
            if "an" in self.model.output_mode:
                # y_question_all,[batch_size,seq_len,question_num]
                # one-hot, [batch_size,seq_len,question_num]
                # qshft,[batch_size,seq_len]
                pass
            else:
                if self.model.predict_next:
                    outputs['y_question'] = outputs['y_question'].squeeze(-1)
                else:
                    outputs['y_question'] = (outputs['y_question'] * F.one_hot(data_new['qshft'].long(), self.model.num_q)).sum(-1)
                outputs['y_concept'] = self.get_avg_fusion_concepts(outputs['y_concept'],data_new['cshft'])

                if "cc" in self.model.loss_mode:
                    qc_target,y_qc_predict = self.get_qc_predict_result(outputs['y_question_concepts'],data_new)
                    del outputs['y_question_concepts']
                    outputs["qc_target"] = qc_target
                    outputs["y_qc_predict"] = y_qc_predict
                
        if "an" in self.model.output_mode:
            all_predict_mode,next_predict_mode = self.model.predict_mode.split("_")[0].split("-")
            y_qc_all = self.get_merge_y(outputs['y_question_all'],outputs['y_concept_all'],all_predict_mode)
            outputs['y_qc_all'] = y_qc_all
            y_qc_next = self.get_merge_y(outputs['y_question_next'],outputs['y_concept_next'],next_predict_mode)
            outputs['y_qc_next'] = y_qc_next

            if self.model.contrast_mode in ['cm_v1','cm_v2']: 
                y_qc_all2 = self.get_merge_y(outputs['y_question_all'+self.model.cm_add_name],outputs['y_concept_all'+self.model.cm_add_name],all_predict_mode)
                y_qc_next2 = self.get_merge_y(outputs['y_question_next'+self.model.cm_add_name],outputs['y_concept_next'+self.model.cm_add_name],next_predict_mode)
                outputs["y"+self.model.cm_add_name] = (y_qc_all2+y_qc_next2)/2

            if self.model.output_mode=="an_irt":
                def sigmoid_inverse(x):
                    # return x
                    return torch.log(x/(1-x+1e-7)+1e-7)
                y = self.model.irt_w[0]*sigmoid_inverse(outputs['y_question_all']) + self.model.irt_w[1]*sigmoid_inverse(outputs['y_concept_all']) - self.model.irt_w[2]*sigmoid_inverse(outputs['y_question_next'])
                # print(f"y is {y}")
                y = torch.sigmoid(y)
            else:
                output_next_lambda = self.model.other_config.get("output_next_lambda",0.5)
                output_all_lambda = self.model.other_config.get("output_all_lambda",0.5)
                y = (y_qc_all*output_all_lambda+y_qc_next*output_next_lambda)/(output_all_lambda+output_next_lambda)
            outputs['y'] = y
        else:
            y = self.get_merge_y(outputs['y_question'],outputs['y_concept'],self.model.predict_mode)
            outputs['y'] = y

        if return_details:
            return outputs,data_new
        else:
            return y