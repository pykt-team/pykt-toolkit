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

    def forward(self, x,return_raw_x=False):
        for lin in self.lins:
            x = F.relu(lin(x))
        if return_raw_x:
            return x,self.out(self.dropout(x))
        else:
            return self.out(self.dropout(x))



def get_outputs(self,emb_qc_shift,h,data,add_name="",model_type='question'):
    outputs = {}
  
    if model_type == 'question':
        h_next = torch.cat([emb_qc_shift,h],axis=-1)
        y_question_next = torch.sigmoid(self.out_question_next(h_next))
        emb_question_all,out_question_all = self.out_question_all(h,return_raw_x=True)
        y_question_all = torch.sigmoid(out_question_all)
        outputs["y_question_next"+add_name] = y_question_next.squeeze(-1)
        outputs["y_question_all"+add_name] = (y_question_all * F.one_hot(data['qshft'].long(), self.num_q)).sum(-1)
        outputs["emb_question_all"] = emb_question_all
        
    else: 
        h_next = torch.cat([emb_qc_shift,h],axis=-1)
        # next
        emb_concept_next,out_concept_next = self.out_concept_next(h_next,return_raw_x=True)
        y_concept_next = torch.sigmoid(out_concept_next)

        #all predict
        emb_concept_all,out_concept_all = self.out_concept_all(h,return_raw_x=True)
        y_concept_all = torch.sigmoid(out_concept_all)

        # add results to outputs
        outputs["y_concept_next"+add_name] = self.get_avg_fusion_concepts(y_concept_next,data['cshft'])
        outputs["y_concept_all"+add_name] = self.get_avg_fusion_concepts(y_concept_all,data['cshft'])
        outputs["emb_concept_next"] = emb_concept_next
        outputs["emb_concept_all"] = emb_concept_all
    return outputs

class QIKTNCDNet(nn.Module):
    def __init__(self, num_q,num_c,emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',mlp_layer_num=1,other_config={}):
        super().__init__()
        self.model_name = "qikt_ncd"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.mlp_layer_num = mlp_layer_num
        self.device = device
        self.other_config = other_config
        self.output_mode = self.other_config.get('output_mode','an')


        self.emb_type = emb_type
      

        self.que_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type=self.emb_type,model_name=self.model_name,device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim)
       
       
        self.que_lstm_layer = nn.LSTM(self.emb_size*4, self.hidden_size, batch_first=True)
        self.concept_lstm_layer = nn.LSTM(self.emb_size*2, self.hidden_size, batch_first=True)
       
        self.dropout_layer = nn.Dropout(dropout)
        

        self.out_question_next = MLP(self.mlp_layer_num,self.hidden_size*3,1,dropout)
        self.out_question_all = MLP(self.mlp_layer_num,self.hidden_size,num_q,dropout)

        self.out_concept_next = MLP(self.mlp_layer_num,self.hidden_size*3,num_c,dropout)
        self.out_concept_all = MLP(self.mlp_layer_num,self.hidden_size,num_c,dropout)
        # self.out_merge_multi_h = MLP(self.mlp_layer_num*2,self.hidden_size*5,1,dropout)
        #ncd
        self.hs = nn.Sequential(MLP(self.mlp_layer_num,self.hidden_size,num_c,dropout),
                                    nn.Sigmoid())
        self.q_diff = nn.Sequential(MLP(self.mlp_layer_num,self.hidden_size*2,num_c,dropout),
                                    nn.Sigmoid())
        self.q_disc = nn.Sequential(MLP(self.mlp_layer_num,self.hidden_size*2,num_c,dropout),
                                    nn.Sigmoid())
        self.ncd_output = nn.Sequential(MLP(self.mlp_layer_num,num_c,num_c,dropout),
                                    nn.Sigmoid())
                                    
                                    
       

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
    
        _,emb_qca,emb_qc,emb_q,emb_c = self.que_emb(q,c,r)#[batch_size,emb_size*4],[batch_size,emb_size*2],[batch_size,emb_size*1],[batch_size,emb_size*1]
        

        emb_qc_shift = emb_qc[:,1:,:]
        emb_qca_current = emb_qca[:,:-1,:]
        # question model
        que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0])
        que_outputs = get_outputs(self,emb_qc_shift,que_h,data,add_name="",model_type="question")
        outputs = que_outputs

        # concept model
        emb_ca = torch.cat([emb_c.mul((1-r).unsqueeze(-1).repeat(1,1, self.emb_size)),
                                emb_c.mul((r).unsqueeze(-1).repeat(1,1, self.emb_size))], dim = -1)# s_t 扩展，分别对应正确的错误的情况
                                
        emb_ca_current = emb_ca[:,:-1,:]

        concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current)[0])
        concept_outputs = get_outputs(self,emb_qc_shift,concept_h,data,add_name="",model_type="concept")
        outputs.update(concept_outputs)

        #NCD
        h_ncd = (self.hs(concept_h)-self.q_diff(emb_qc_shift))*self.q_disc(emb_qc_shift)
        y_ncd_raw = self.ncd_output(h_ncd).squeeze(-1)
        y_ncd = self.get_avg_fusion_concepts(y_ncd_raw,data['cshft'])
        outputs['y_ncd'] = y_ncd
        return outputs

class QIKTNCD(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", pretrain_dim=768,device='cpu',seed=0,mlp_layer_num=1,other_config={},**kwargs):
        model_name = "qikt_ncd"
       
        debug_print(f"emb_type is {emb_type}",fuc_name="QIKTNCD")

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = QIKTNCDNet(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,mlp_layer_num=mlp_layer_num,other_config=other_config)
       
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
        self.eval_result = {}
        self.output_lambda_name_list = ["output_c_all_lambda","output_c_next_lambda","output_q_all_lambda","output_q_next_lambda","output_ncd_lambda"]
        self.loss_lambda_name_list = [x.replace("output_","loss_") for x in self.output_lambda_name_list]
        self.output_name_list = ['y_concept_all','y_concept_next','y_question_all','y_question_next','y_ncd']
        # print(f"other_config is {other_config}")


    def train_one_step(self,data,process=True,return_all=False):
        outputs,data_new = self.predict_one_step(data,return_details=True,process=process)
        def get_loss_lambda(x):
            x = x.replace("loss_","")
            return self.model.other_config.get(f'loss_{x}',0)*self.model.other_config.get(f'output_{x}',0)

        loss_lambda_list  = [get_loss_lambda(x) for x in self.loss_lambda_name_list]
        # print(f"loss_lambda_list is {loss_lambda_list}")    
        # over all
        loss_kt = self.get_loss(outputs['y'],data_new['rshft'],data_new['sm'])

        total_loss = loss_kt
        loss_print_str = f"loss_kt is {loss_kt:.3f}"

        for loss_lambda,output_name in zip(loss_lambda_list,self.output_name_list):
            if loss_lambda==0:
                continue
            loss_item = self.get_loss(outputs[output_name],data_new['rshft'],data_new['sm'])
            total_loss += loss_item * loss_lambda

            loss_print_str+=f",{output_name} loss is {loss_item:.3f}"
        loss_print_str+=f",total_loss is {total_loss:.3f}"
        print(loss_print_str)
        
        return outputs['y'],total_loss#y_question没用


    def predict(self,dataset,batch_size,return_ts=False,process=True):
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_pred_dict = {}
            for data in test_loader:
                new_data = self.batch_to_device(data,process=process)
                outputs,data_new = self.predict_one_step(data,return_details=True)
               
                for key in outputs:
                    if not key.startswith("y") or key in ['y_qc_predict']:
                        continue
                    elif key not in y_pred_dict:
                       y_pred_dict[key] = []
                    y = torch.masked_select(outputs[key], new_data['sm']).detach().cpu()#get label
                    y_pred_dict[key].append(y.numpy())
                
                t = torch.masked_select(new_data['rshft'], new_data['sm']).detach().cpu()
                y_trues.append(t.numpy())


        results = y_pred_dict
        for key in results:
            results[key] = np.concatenate(results[key], axis=0)
        ts = np.concatenate(y_trues, axis=0)
        results['ts'] = ts
        return results

    def evaluate(self,dataset,batch_size,acc_threshold=0.5):
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
        
        self.eval_result = eval_result
        return eval_result

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        data_new = self.batch_to_device(data,process=process)
        outputs = self.model(data_new['cq'].long(),data_new['cc'],data_new['cr'].long(),data=data_new)
        
        output_lambda_list  = [self.model.other_config.get(x,0) for x in self.output_lambda_name_list]
        # print(f"output_lambda_list is {output_lambda_list}")    

        if self.model.output_mode=="an_irt":
            def sigmoid_inverse(x,epsilon=1e-8):
                return torch.log(x/(1-x+epsilon)+epsilon)
            y = 0 
            for output_lambda,output_name in zip(output_lambda_list,self.output_name_list):
                y+=sigmoid_inverse(outputs[output_name])*output_lambda
            y = torch.sigmoid(y)
        else:
            # output weight
            y = 0 
            for output_lambda,output_name in zip(output_lambda_list,self.output_name_list):
                if output_lambda==0:
                    continue
                y += output_lambda*outputs[output_name]
            y = y/sum(output_lambda_list)
        outputs['y'] = y

        if return_details:
            return outputs,data_new
        else:
            return y