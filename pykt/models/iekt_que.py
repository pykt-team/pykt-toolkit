import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel,QueEmb
from torch.distributions import Categorical
import math
from .iekt_utils import batch_data_to_device,mygru,funcsgru,funcs
from pykt.utils import debug_print
import traceback

class IEKTQueNet(nn.Module): 
    def __init__(self, num_q,num_c,emb_size,max_concepts,batch_size,lamb=40,n_layer=1,cog_levels=10,acq_levels=10,dropout=0,gamma=0.93, emb_type='qid', emb_path="", pretrain_dim=768,device='cpu'):
        super().__init__()
        self.model_name = "iekt_que"
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
        self.prob_emb = nn.Parameter(torch.randn(num_q, emb_size).to(self.device), requires_grad=True)#题目表征
        self.gamma = gamma
        self.lamb = lamb
        self.gru_h = mygru(0, emb_size * 4, emb_size)
        showi0 = []
        for i in range(0, batch_size):
            showi0.append(i)
        self.show_index = torch.tensor(showi0).to(self.device)
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num, emb_size).to(self.device), requires_grad=True)#知识点表征
        self.sigmoid = torch.nn.Sigmoid()

    def get_ques_representation(self, q, c, data_len):
        """Get question representation equation3
        Input example, batch is 8.
        (tensor([592,   1, 461, 490,  20,  37, 257, 247]),
         tensor([[49,  0,  0,  0],
                [ 1,  2,  3,  0],
                [33, 34,  0,  0],
                [26,  0,  0,  0],
                [ 9, 10,  0,  0],
                [12,  0,  0,  0],
                [52,  7,  0,  0],
                [51,  0,  0,  0]]),
         tensor([[1., 0., 0., 0.],
                [1., 1., 1., 0.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]]),
        8)

        Args:
            p (_type_): question ids
            related_concept_index (_type_): concepts ids
            filter0 (_type_): concepts ids mask 矩阵，1表示符合这个知识点，0表示不符合
            data_len (_type_): batch size

        Returns:
            _type_: _description_
        """
        #debug_print("start",fuc_name='get_ques_representation')
        
        #debug_print(f"start concepts_cat , c is {c},c shape is {c.shape}",fuc_name='get_ques_representation')
        concepts_cat = torch.cat(
            [torch.zeros(1, self.emb_size).to(self.device),
            self.concept_emb],
            dim = 0).unsqueeze(0).repeat(data_len, 1, 1)#确保0取到值为全0
        
        # #debug_print(f"start r_index , c is {c},c shape is {c.shape}",fuc_name='get_ques_representation')
        r_index = self.show_index[0: data_len].unsqueeze(1).repeat(1, self.max_concept)
        
        #debug_print(f"start related_concepts , c is {c},c shape is {c.shape}",fuc_name='get_ques_representation')
        related_concepts = concepts_cat[r_index, c+1,:]
        
        #debug_print(f"start filter0, c is {c},c shape is {c.shape}",fuc_name='get_ques_representation')
        filter0 = torch.where(c!=-1,1,0)

        #debug_print("start filter_sum",fuc_name='get_ques_representation')
        filter_sum = torch.sum(filter0, dim = 1)
        
        #debug_print(f"r_index is {r_index}, shape is {r_index.shape},device is {r_index.device}",fuc_name='get_ques_representation')
        #debug_print(f"filter_sum is {filter_sum}, shape is {filter_sum.shape},device is {filter_sum.device}",fuc_name='get_ques_representation')

        # debug_print(f"start div",fuc_name='get_ques_representation')

        div = torch.where(filter_sum == 0, 
            torch.tensor(1).to(self.device), 
            filter_sum
            ).unsqueeze(1).repeat(1, self.emb_size)

        # debug_print(f"start div,filter_sum is {filter_sum}",fuc_name='get_ques_representation')
        # print(traceback.format_exc())
        # debug_print("start concept_level_rep",fuc_name='get_ques_representation')
        concept_level_rep = torch.sum(related_concepts, dim = 1) / div
        
        prob_cat = torch.cat([
            torch.zeros(1, self.emb_size).to(self.device),
            self.prob_emb], dim = 0)
        
        item_emb = prob_cat[q]

        v = torch.cat(
            [concept_level_rep,
            item_emb],
            dim = 1)
        return v


    def pi_cog_func(self, x, softmax_dim = 1):
        return F.softmax(self.select_preemb(x), dim = softmax_dim)

    def obtain_v(self, q, c, data_len, h, x, emb):
        """_summary_

        Args:
            this_input (_type_): _description_
            h (_type_): _description_
            x (_type_): rt_x = torch.zeros(data_len, 1, args.dim * 2).to(self.device) 不知道干啥用的
            emb (_type_): m_t

        Returns:
            _type_: _description_
        """
        #debug_print("start",fuc_name='obtain_v')
        v = self.get_ques_representation(q,c, data_len)
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
        return F.softmax(self.checker_emb(x), dim = softmax_dim)



class IEKTQue(QueBaseModel):
    def __init__(self, num_q,num_c,emb_size,max_concepts,batch_size,lamb=40,n_layer=1,cog_levels=10,acq_levels=10,dropout=0,gamma=0.93, emb_type='qid', emb_path="", pretrain_dim=768,device='cpu',seed=0):
        model_name = "iekt_que"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)

        self.model = IEKTQueNet(num_q=num_q,num_c=num_c,lamb=lamb,emb_size=emb_size,max_concepts=max_concepts,batch_size=batch_size,n_layer=n_layer,cog_levels=cog_levels,acq_levels=acq_levels,dropout=dropout,gamma=gamma, emb_type=emb_type, emb_path=emb_path, pretrain_dim=pretrain_dim,device=device)

        self.model = self.model.to(device)
        # self.step = 0
    
    def train_one_step(self,data):
        # self.step+=1
        # debug_print(f"step is {self.step},data is {data}","train_one_step")
        # debug_print(f"step is {self.step}","train_one_step")
        BCELoss = torch.nn.BCEWithLogitsLoss()
        train_sigmoid = torch.nn.Sigmoid()
        data_new = self.batch_to_device(data)
        
        data_len = data_new['cc'].shape[0]
        seq_len = data_new['cc'].shape[1]
        h = torch.zeros(data_len, self.model.emb_size).to(self.device)
        p_action_list, pre_state_list, emb_action_list, op_action_list, actual_label_list, states_list, reward_list, predict_list, ground_truth_list = [], [], [], [], [], [], [], [], []

        rt_x = torch.zeros(data_len, 1, self.model.emb_size * 2).to(self.device)
        for seqi in range(0, seq_len):#序列长度
            #debug_print(f"start data_new, c is {data_new}",fuc_name='train_one_step')
            ques_h = torch.cat([
                self.model.get_ques_representation(q=data_new['cq'][:,seqi], c=data_new['cc'][:,seqi], data_len=data_len),
                h], dim = 1)#equation4
            # d = 64*3 [题目,知识点,h]
            flip_prob_emb = self.model.pi_cog_func(ques_h)

            m = Categorical(flip_prob_emb)#equation 5 的 f_p
            emb_ap = m.sample()#equation 5
            emb_p = self.model.cog_matrix[emb_ap,:]#equation 6

            h_v, v, logits, rt_x = self.model.obtain_v(q=data_new['cq'][:,seqi], c=data_new['cc'][:,seqi], data_len=data_len, 
                                                        h=h, x=rt_x, emb=emb_p)#equation 7
            prob = train_sigmoid(logits)#equation 7 sigmoid

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

            ground_truth = data_new['cr'][:,seqi].squeeze(-1)

            flip_prob_emb = self.model.pi_sens_func(out_x)##equation12中的f_e

            m = Categorical(flip_prob_emb)
            emb_a = m.sample()
            emb = self.model.acq_matrix[emb_a,:]#equation12 s_t

            h = self.model.update_state(h, v, emb, ground_truth.unsqueeze(1))#equation13～14
            
            emb_action_list.append(emb_a)#s_t 列表
            p_action_list.append(emb_ap)#m_t
            states_list.append(out_x)
            pre_state_list.append(ques_h)#上一个题目的状态
            
            ground_truth_list.append(ground_truth)
            predict_list.append(logits.squeeze(1))
            this_reward = torch.where(out_operate_logits.squeeze(1).float() == ground_truth,
                            torch.tensor(1).to(self.device), 
                            torch.tensor(0).to(self.device))# if condition x else y,这里相当于统计了正确的数量
            reward_list.append(this_reward)

        #以下是强化学习部分内容
        seq_num = data['smasks'].sum(axis=-1)+1
        emb_action_tensor = torch.stack(emb_action_list, dim = 1)
        p_action_tensor = torch.stack(p_action_list, dim = 1)
        state_tensor = torch.stack(states_list, dim = 1)
        pre_state_tensor = torch.stack(pre_state_list, dim = 1)
        reward_tensor = torch.stack(reward_list, dim = 1).float() / (seq_num.unsqueeze(-1).repeat(1, seq_len)).float()#equation15
        logits_tensor = torch.stack(predict_list, dim = 1)
        ground_truth_tensor = torch.stack(ground_truth_list, dim = 1)
        loss = []
        tracat_logits = []
        tracat_ground_truth = []
        
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            this_reward_list = reward_tensor[i]
        
            this_cog_state = torch.cat([pre_state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, pre_state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)
            this_sens_state = torch.cat([state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)

            td_target_cog = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_cog = td_target_cog
            delta_cog = delta_cog.detach().cpu().numpy()

            td_target_sens = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_sens = td_target_sens
            delta_sens = delta_sens.detach().cpu().numpy()

            advantage_lst_cog = []
            advantage = 0.0
            for delta_t in delta_cog[::-1]:
                advantage = self.model.gamma * advantage + delta_t[0]#equation17
                advantage_lst_cog.append([advantage])
            advantage_lst_cog.reverse()
            advantage_cog = torch.tensor(advantage_lst_cog, dtype=torch.float).to(self.device)
            
            pi_cog = self.model.pi_cog_func(this_cog_state[:-1])
            pi_a_cog = pi_cog.gather(1,p_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_cog = -torch.log(pi_a_cog) * advantage_cog#equation16
            
            loss.append(torch.sum(loss_cog))

            advantage_lst_sens = []
            advantage = 0.0
            for delta_t in delta_sens[::-1]:
                # advantage = args.gamma * args.beta * advantage + delta_t[0]
                advantage = self.model.gamma * advantage + delta_t[0]
                advantage_lst_sens.append([advantage])
            advantage_lst_sens.reverse()
            advantage_sens = torch.tensor(advantage_lst_sens, dtype=torch.float).to(self.device)
            
            pi_sens = self.model.pi_sens_func(this_sens_state[:-1])
            pi_a_sens = pi_sens.gather(1,emb_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_sens = - torch.log(pi_a_sens) * advantage_sens#equation18
            loss.append(torch.sum(loss_sens))
            

            this_prob = logits_tensor[i][0: this_seq_len]
            this_groud_truth = ground_truth_tensor[i][0: this_seq_len]

            tracat_logits.append(this_prob)
            tracat_ground_truth.append(this_groud_truth)

        bce = BCELoss(torch.cat(tracat_logits, dim = 0), torch.cat(tracat_ground_truth, dim = 0))   
        y = torch.cat(tracat_logits, dim = 0)
        label_len = torch.cat(tracat_ground_truth, dim = 0).size()[0]
        loss_l = sum(loss)
        loss = self.model.lamb * (loss_l / label_len) +  bce#equation21
        return y,loss

    def predict_one_step(self,data,return_details=False):
        eval_sigmoid = torch.nn.Sigmoid()
        data_new = self.batch_to_device(data)
        data_len = data['smasks'].shape[0]
        seq_len = data_new['cc'].shape[1]
        h = torch.zeros(data_len, self.model.emb_size).to(self.device)
        batch_probs, uni_prob_list, actual_label_list, states_list, reward_list =[], [], [], [], []
        H = None
        if 'eernna' in self.model_name:
            H = torch.zeros(data_len, 1, self.model.emb_size).to(self.device)
        else:
            H = torch.zeros(data_len, self.model.concept_num, self.model.emb_size).to(self.device)
        rt_x = torch.zeros(data_len, 1, self.model.emb_size * 2).to(self.device)
        for seqi in range(0, seq_len):
            ques_h = torch.cat([
                    self.model.get_ques_representation(q=data_new['cq'][:,seqi], c=data_new['cc'][:,seqi], data_len=data_len),
                    h], dim = 1)
            flip_prob_emb = self.model.pi_cog_func(ques_h)

            m = Categorical(flip_prob_emb)
            emb_ap = m.sample()
            emb_p = self.model.cog_matrix[emb_ap,:]

            # h_v, v, logits, rt_x = self.model.obtain_v(data_new, h, rt_x, emb_p)#
            h_v, v, logits, rt_x = self.model.obtain_v(q=data_new['cq'][:,seqi], c=data_new['cc'][:,seqi], data_len=data_len, 
                                                        h=h, x=rt_x, emb=emb_p)#equation 7
            prob = eval_sigmoid(logits)
            out_operate_groundtruth = data_new['cr'][:,seqi].unsqueeze(-1)
            out_x_groundtruth = torch.cat([
                h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1-out_operate_groundtruth).repeat(1, h_v.size()[-1]).float())],
                dim = 1)

            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)) 
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1-out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                dim = 1)                
            out_x = torch.cat([out_x_groundtruth, out_x_logits], dim = 1)

            ground_truth = data_new['cr'][:,seqi].squeeze(-1)

            flip_prob_emb = self.model.pi_sens_func(out_x)
            
            m = Categorical(flip_prob_emb)
            emb_a = m.sample()
            emb = self.model.acq_matrix[emb_a,:]

            h = self.model.update_state(h, v, emb, ground_truth.unsqueeze(1))
            uni_prob_list.append(prob.detach())
        prob_tensor = torch.cat(uni_prob_list, dim = 1)
        print(f"prob_tensor shape is {prob_tensor.shape}")
        return prob_tensor[:,1:]