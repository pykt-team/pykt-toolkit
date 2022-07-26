import os
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel,QueEmb
from torch.distributions import Categorical
from .iekt_utils import mygru,funcs
from pykt.utils import debug_print

class IEKTNet(nn.Module): 
    def __init__(self, num_q,num_c,emb_size,max_concepts,lamb=40,n_layer=1,cog_levels=10,acq_levels=10,dropout=0,gamma=0.93, emb_type='qc_merge', emb_path="", pretrain_dim=768,device='cpu',train_mode="sample"):
        super().__init__()
        self.model_name = "iekt_ce"
        self.emb_size = emb_size
        self.concept_num = num_c
        self.max_concept = max_concepts
        self.device = device
        self.emb_type = emb_type
        self.predictor = funcs(n_layer, emb_size * 5, 1, dropout)
        self.cog_matrix = nn.Parameter(torch.randn(cog_levels, emb_size * 2).to(self.device), requires_grad=True) 
        self.acq_matrix = nn.Parameter(torch.randn(acq_levels, emb_size * 2).to(self.device), requires_grad=True)
        self.select_preemb = funcs(n_layer, emb_size * 3, cog_levels, dropout)#MLP
        self.checker_emb = funcs(n_layer, emb_size * 12, acq_levels, dropout) 
        # self.prob_emb = nn.Parameter(torch.randn(num_q, emb_size).to(self.device), requires_grad=True)#题目表征
        self.gamma = gamma
        self.lamb = lamb
        self.gru_h = mygru(0, emb_size * 4, emb_size)
        # self.concept_emb = nn.Parameter(torch.randn(self.concept_num, emb_size).to(self.device), requires_grad=True)#知识点表征
        self.sigmoid = torch.nn.Sigmoid()
        self.que_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type=self.emb_type,model_name=self.model_name,device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim)
        self.train_mode = train_mode
        

    def get_ques_representation(self, q, c):
        """Get question representation equation 3

        Args:
            q (_type_): question ids
            c (_type_): concept ids

        Returns:
            _type_: _description_
        """
       
        v = self.que_emb(q,c)

        return v


    def pi_cog_func(self, x, softmax_dim = 1):
        return torch.softmax(self.select_preemb(x), dim = softmax_dim)

    def obtain_v(self, q, c, h, x, emb):
        """_summary_

        Args:
            q (_type_): _description_
            c (_type_): _description_
            h (_type_): _description_
            x (_type_): _description_
            emb (_type_): m_t

        Returns:
            _type_: _description_
        """

        #debug_print("start",fuc_name='obtain_v')
        v = self.get_ques_representation(q,c)
        predict_x = torch.cat([h, v], dim = 1)#equation4
        h_v = torch.cat([h, v], dim = 1)#equation4 为啥要计算两次？
        prob = self.predictor(torch.cat([
            predict_x, emb
        ], dim = 1))#equation7
        return h_v, v, prob, x

    def update_state(self, h, v, emb, operate):
        """_summary_

        Args:
            h (_type_): rnn的h
            v (_type_): question 表示
            emb (_type_): s_t knowledge acquistion sensitivity
            operate (_type_): label

        Returns:
            next_p_state {}: _description_
        """
        #equation 13
        v_cat = torch.cat([
            v.mul(operate.repeat(1, self.emb_size * 2)),
            v.mul((1 - operate).repeat(1, self.emb_size * 2))], dim = 1)#v_t扩展，分别对应正确的错误的情况
        e_cat = torch.cat([
            emb.mul((1-operate).repeat(1, self.emb_size * 2)),
            emb.mul((operate).repeat(1, self.emb_size * 2))], dim = 1)# s_t 扩展，分别对应正确的错误的情况
        inputs = v_cat + e_cat#起到concat作用
        
        h_t_next = self.gru_h(inputs, h)#equation14
        return h_t_next
    

    def pi_sens_func(self, x, softmax_dim = 1):
        return torch.softmax(self.checker_emb(x), dim = softmax_dim)

    def forward(self,data):
        sigmoid_func = torch.nn.Sigmoid()
        data_new = data
        
        data_len = data_new['cc'].shape[0]
        seq_len = data_new['cc'].shape[1]
        h = torch.zeros(data_len, self.emb_size).to(self.device)
        uni_prob_list = []

        rt_x = torch.zeros(data_len, 1, self.emb_size * 2).to(self.device)
        for seqi in range(0, seq_len):#序列长度
            #debug_print(f"start data_new, c is {data_new}",fuc_name='train_one_step')
            ques_h = torch.cat([
                self.get_ques_representation(q=data_new['cq'][:,seqi], c=data_new['cc'][:,seqi]),
                h], dim = 1)#equation4
            
            # d = 64*3 [题目,知识点,h]
            flip_prob_emb = self.pi_cog_func(ques_h)#(batch_size,cog_levels)
            # print(f"flip_prob_emb is {flip_prob_emb},shape is {flip_prob_emb.shape}")

            if self.train_mode == "sample":
                m = Categorical(flip_prob_emb)#equation 5 的 f_p
                emb_ap = m.sample()#equation 5
                emb_p = self.cog_matrix[emb_ap,:]#equation 6
            elif self.train_mode == "attention":
                emb_p = flip_prob_emb.matmul(self.cog_matrix)#
            elif self.train_mode == "argmax":
                emb_ap = flip_prob_emb.argmax(axis=-1)
                emb_p = self.cog_matrix[emb_ap,:]#equation 6


            h_v, v, logits, rt_x = self.obtain_v(q=data_new['cq'][:,seqi], c=data_new['cc'][:,seqi], 
                                                        h=h, x=rt_x, emb=emb_p)#equation 7
            prob = sigmoid_func(logits)#equation 7 sigmoid

            out_operate_groundtruth = data_new['cr'][:,seqi].unsqueeze(-1) #获取标签
            
            out_x_groundtruth = torch.cat([
                h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1-out_operate_groundtruth).repeat(1, h_v.size()[-1]).float())],
                dim = 1)#equation9

            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)) 
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1-out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                dim = 1)#equation10                
            out_x = torch.cat([out_x_groundtruth, out_x_logits], dim = 1)#equation11
            # print(f"data_new['cr'] is {data_new['cr']}")
            ground_truth = data_new['cr'][:,seqi]
            # print(f"ground_truth shape is {ground_truth.shape},ground_truth is {ground_truth}")
            flip_prob_emb = self.pi_sens_func(out_x)##equation12中的f_e

            if self.train_mode == "sample":
                m = Categorical(flip_prob_emb)
                emb_a = m.sample()
                emb = self.acq_matrix[emb_a,:]#equation12 s_t
            elif self.train_mode == "attention":
                emb = flip_prob_emb.matmul(self.acq_matrix)#
            elif self.train_mode == "argmax":
                emb_a = flip_prob_emb.argmax(axis=-1)
                emb = self.acq_matrix[emb_a,:]
           
            h = self.update_state(h, v, emb, ground_truth.unsqueeze(1))#equation13～14
            uni_prob_list.append(prob)
            
        y = torch.cat(uni_prob_list, dim = 1)
        return y



class IEKTCE(QueBaseModel):
    def __init__(self, num_q,num_c,emb_size,max_concepts,lamb=40,n_layer=1,cog_levels=10,acq_levels=10,dropout=0,gamma=0.93, emb_type='qid', emb_path="", pretrain_dim=768,device='cpu',seed=0,train_mode = "sample"):
        model_name = "iekt_ce"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)

        self.model = IEKTNet(num_q=num_q,num_c=num_c,lamb=lamb,emb_size=emb_size,max_concepts=max_concepts,n_layer=n_layer,cog_levels=cog_levels,acq_levels=acq_levels,dropout=dropout,gamma=gamma, emb_type=emb_type, emb_path=emb_path, pretrain_dim=pretrain_dim,device=device,train_mode=train_mode)

        self.model = self.model.to(device)
        # self.step = 0
    
    def train_one_step(self,data,process=True):
        y,data_new = self.predict_one_step(data,return_details=True,process=process)
        loss = self.get_loss(y,data_new['rshft'],data_new['sm'])#get loss
        return y,loss

    

    def predict_one_step(self,data,return_details=False,process=True):
        data_new = self.batch_to_device(data,process)
        y = self.model(data_new)
        y = y[:,1:]
        if return_details:
            return y,data_new
        else:
            return y