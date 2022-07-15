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

    
        
    def forward(self, q, c ,r,data=None):
        if self.emb_type in ["qcaid","qcaid_h"]:
            xemb,emb_q,emb_c = self.que_emb(q,c,r)
        elif self.emb_type in ["iekt"]:
            _,emb_qca,emb_qc,emb_q,emb_c = self.que_emb(q,c,r)#[batch_size,emb_size*4],[batch_size,emb_size*2],[batch_size,emb_size*1],[batch_size,emb_size*1]
           
            # emb_qc_current = emb_qc[:,:-1,:]
            emb_qc_shift = emb_qc[:,1:,:]
            emb_qca_current = emb_qca[:,:-1,:]
            # emb_qca_shift = emb_qca[:,1:,:]
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
        if "an" in self.output_mode:
            #next predict
            h_next = torch.cat([emb_qc_shift,h],axis=-1)
            y_question_next = torch.sigmoid(self.out_question_next(h_next))
            y_concept_next = torch.sigmoid(self.out_concept_next(h_next))

            #all predict
            h_all = h
            y_question_all = torch.sigmoid(self.out_question_all(h_all))
            y_concept_all = torch.sigmoid(self.out_concept_all(h_all))
            outputs = {"y_question_next":y_question_next,"y_concept_next":y_concept_next,
                        "y_question_all":y_question_all,"y_concept_all":y_concept_all}

            if "cc" in self.loss_mode:
                y_question_concepts_next,y_question_concepts_all = None,None
                outputs['y_question_concepts_next'] = y_question_concepts_next
                outputs['y_question_concepts_all'] = y_question_concepts_all
           
            return outputs

        else:
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
            elif self.loss_mode in ["q_cc","c_cc","qc_cc","cc"]:
                # 知识点分类当作多标签分类
                y_question_concepts = torch.softmax(self.out_concept_classifier(emb_q[:,1:,:]),axis=-1)
            else:
                y_question_concepts = None
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
        elif loss_mode in ['c_dyn']:
            acc_kt = self.eval_result.get("acc",1)
            acc_kc = self.eval_result.get("kc_em_acc",1)
            c_dyn_a = self.model.other_config.get("c_dyn_a",0)
            c_dyn_b = self.model.other_config.get("c_dyn_b",0)
            alpha_kt = (acc_kc+c_dyn_a)/(acc_kt+acc_kc+c_dyn_a+c_dyn_b)
            alpha_kc = (acc_kt+c_dyn_b)/(acc_kt+acc_kc+c_dyn_a+c_dyn_b)
            print(f"acc_kt={acc_kt},acc_kc={acc_kc},alpha_kt={alpha_kt},alpha_kc={alpha_kc},c_dyn_a={c_dyn_a},c_dyn_b={c_dyn_b}")
            loss = alpha_kt*loss_concept + alpha_kc*loss_question_concept
        
        return loss
   
    def get_avg_fusion_concepts(self,y_concept,cshft):
        """获取知识点 fusion 的预测结果
        """
        concept_mask = torch.where(cshft.long()==-1,False,True)
        concept_index = F.one_hot(torch.where(cshft!=-1,cshft,0),self.model.num_c)
        concept_sum = (y_concept.unsqueeze(2).repeat(1,1,4,1)*concept_index).sum(-1)
        concept_sum = concept_sum*concept_mask#remove mask
        y_concept = concept_sum.sum(-1)/torch.where(concept_mask.sum(-1)!=0,concept_mask.sum(-1),1)
        return y_concept


    def train_one_step(self,data,process=True,return_all=False):
        
        if "fr" in self.model.loss_mode:#only for predict all
            y_question_raw,y_concept_raw,y_question_concepts_raw,data_new = self.predict_one_step(data,return_details=True,process=process,return_raw=True)
            seq_len = data_new['qshft'].shape[1]
            loss_question = 0
            loss_concept = 0
            num_inter = 0
            for i in range(seq_len):
                #new mask
                fr_window = self.model.other_config.get("fr_window",1)
                sm_step = data_new['sm'][:,i:i+fr_window]
                num_inter_step = sm_step.sum()
                if num_inter_step==0:
                    break
                rshft_step = data_new['rshft'][:,i:i+fr_window]
                cshft_step = data_new['cshft'][:,i:i+fr_window]
                qshft_step = data_new['qshft'][:,i:i+fr_window]

                valid_seq_len = min(fr_window,rshft_step.shape[1])
                num_inter+=num_inter_step
                #new y_concept
                current_step_concept_raw = y_concept_raw[:,i].unsqueeze(1)
                # print(f"current_step_concept_raw shape is {current_step_concept_raw.shape}")
                current_concept_expand = current_step_concept_raw.repeat(1,valid_seq_len,1)
                # current_concept_expand = current_step_concept_raw.repeat(-1,valid_seq_len,-1)
                y_concept_step = self.get_avg_fusion_concepts(current_concept_expand,cshft_step)
      
                loss_concept_step =self.get_loss(y_concept_step,rshft_step,sm_step)
                loss_concept = loss_concept+loss_concept_step*num_inter_step
            
               
                #new y_question
                current_step_question_raw = y_question_raw[:,i].unsqueeze(1)
                current_question_expand = current_step_question_raw.repeat(1,valid_seq_len,1)
                # current_question_expand = current_step_question_raw.repeat(-1,valid_seq_len,-1)
                # print(f"current_question_expand shape is {current_question_expand.shape}")
                y_question_step = (current_question_expand * F.one_hot(qshft_step.long(), self.model.num_q)).sum(-1)
                loss_question_step = self.get_loss(y_question_step,rshft_step,sm_step)#question level loss
                loss_question = loss_question + loss_question_step*num_inter_step
               

            loss_question = loss_question/num_inter
            loss_concept = loss_concept/num_inter
            loss_question_concept = 0
            y_question = None
        else: 
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
            dyn_a = self.model.other_config.get("dyn_a",0)
            dyn_b = self.model.other_config.get("dyn_b",0)

            loss_all = self.get_merge_loss(loss_question_all,loss_concept_all,loss_question_concept,all_loss_mode)   
            loss_next = self.get_merge_loss(loss_question_next,loss_concept_next,loss_question_concept,next_loss_mode)
            auc_all =  self.eval_result.get("y_qc_all_kt_auc",1)
            auc_next =  self.eval_result.get("y_concept_next_kt_auc",1)
            loss_same = F.mse_loss(outputs['y_qc_all'],outputs['y_concept_next'])*0
            if "dyn" in self.model.loss_mode:
                alpha_all = (auc_next+dyn_a)/(auc_all+dyn_a+auc_next+dyn_b)
                alpha_next = (auc_all+dyn_b)/(auc_all+dyn_a+auc_next+dyn_b)
                loss = alpha_all*loss_all + alpha_next*loss_next
                print(f"auc_all={auc_all},auc_next={auc_next},alpha_all={alpha_all},alpha_next={alpha_next},dyn_a={dyn_a},dyn_b={dyn_b}")
            else:
                loss_all = loss_all + loss_next + loss_same

           
            loss = (loss_all*0.5+loss_next*0.5)
            print(f"loss={loss:.4f},loss_all={loss_all:.4f},loss_next={loss_next:.4f},loss_same={loss_same:.4f}")
            return outputs['y'],loss#y_question没用
        else:
            print(f"loss_question is {loss_question:.4f},loss_concept is {loss_concept:.4f},loss_question_concept is {loss_question_concept:.4f}")
            
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
                    qc_target = outputs['qc_target']
                    y_qc_predict = outputs['y_qc_predict']
                    y_qc_true_list.append(qc_target.detach().cpu().numpy())
                    y_qc_pred_list.append(y_qc_predict.detach().cpu().numpy().argmax(axis=-1))
        
        for key in outputs:
            y_pred_dict[key] = np.concatenate(y_pred_dict[key], axis=0)
            print(f"y type is {key},shape is {y_pred_dict[key].shape}")
                

        ts = np.concatenate(y_trues, axis=0)
        print(f"ts shape is {ts.shape}")

        if "cc" in self.model.loss_mode:
            kc_ts = np.concatenate(y_qc_true_list, axis=0)
            kc_ps = np.concatenate(y_qc_pred_list, axis=0)
        else:
            kc_ts = None
            kc_ps = None
        results = {"ts":ts,"kc_ts":kc_ts,"kc_ps":kc_ps}
        results.update(y_pred_dict)
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
        if "an" in self.model.output_mode:
            outputs = self.model(data_new['cq'].long(),data_new['cc'],data_new['cr'].long(),data=data_new)
        else:
            if self.model.emb_type in ["iekt"]:
                y_question,y_concept,y_question_concepts = self.model(data_new['cq'].long(),data_new['cc'],data_new['cr'].long(),data=data_new)
            else:
                y_question,y_concept,y_question_concepts = self.model(data_new['q'].long(),data_new['c'],data_new['r'].long(),data=data_new)
        # print(y_question.shape,y_concept.shape)
        if return_raw:#return raw probability, for future reward
            return y_question,y_concept,y_question_concepts,data_new
        else:
            if "an" in self.model.output_mode:
                outputs["y_question_next"] = outputs["y_question_next"].squeeze(-1)
                outputs["y_concept_next"] = self.get_avg_fusion_concepts(outputs["y_concept_next"],data_new['cshft'])

                outputs["y_question_all"] = (outputs["y_question_all"] * F.one_hot(data_new['qshft'].long(), self.model.num_q)).sum(-1)
                outputs["y_concept_all"] = self.get_avg_fusion_concepts(outputs["y_concept_all"],data_new['cshft'])
               
            else:
                if self.model.predict_next:
                    y_question = y_question.squeeze(-1)
                else:
                    y_question = (y_question * F.one_hot(data_new['qshft'].long(), self.model.num_q)).sum(-1)

                y_concept = self.get_avg_fusion_concepts(y_concept,data_new['cshft'])
                if "cc" in self.model.loss_mode:
                    qc_target,y_qc_predict = self.get_qc_predict_result(y_question_concepts,data_new)
                else:
                    qc_target,y_qc_predict = None,None

        
        if "an" in self.model.output_mode:
            all_predict_mode,next_predict_mode = self.model.predict_mode.split("_")[0].split("-")
            y_qc_all = self.get_merge_y(outputs['y_question_all'],outputs['y_concept_all'],all_predict_mode)
            y_qc_next = self.get_merge_y(outputs['y_question_next'],outputs['y_concept_next'],next_predict_mode)
            y = (y_qc_all*0.5+y_qc_next*0.5)
            outputs['y_qc_all'] = y_qc_all
            outputs['y'] = y
        else:
            y = self.get_merge_y(y_question,y_concept,self.model.predict_mode)

        if return_details:
            if "an" in self.model.output_mode:
                return outputs,data_new
            else:
                outputs = {"y":y,"y_question":y_question,"y_concept":y_concept,"y_question_concepts":y_question_concepts,"y_qc_predict":y_qc_predict,"qc_target":qc_target}
                return outputs,data_new
        else:
            return y