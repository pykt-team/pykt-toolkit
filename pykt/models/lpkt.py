#!/usr/bin/env python
# coding=utf-8

import torch
from torch import nn
# from models.utils import RobertaEncode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LPKT(nn.Module):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, gamma=0.03, dropout=0.2, q_matrix="", emb_type="qid", emb_path="", pretrain_dim=768, use_time=True):
        super(LPKT, self).__init__()
        self.model_name = "lpkt"
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        q_matrix[q_matrix==0] = gamma
        self.q_matrix = q_matrix
        self.n_question = n_question
        print(f"n_question:{self.n_question}")
        self.emb_type = emb_type
        self.use_time = use_time

        self.at_embed = nn.Embedding(n_at + 10, d_k)
        torch.nn.init.xavier_uniform_(self.at_embed.weight)
        self.it_embed = nn.Embedding(n_it + 10, d_k)
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(n_exercise + 10, d_e)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)

        if emb_type.startswith("qidcatr"):
            self.interaction_emb = nn.Embedding(self.num_exercise * 2, self.d_k)
            self.catrlinear = nn.Linear(self.d_k * 2, self.d_k)
            self.pooling = nn.MaxPool1d(2, stride=2)
            self.avg_pooling = nn.AvgPool1d(2, stride=2)
        if emb_type.startswith("qidrobertacatr"):
            self.catrlinear = nn.Linear(self.d_k * 3, self.d_k)
            self.pooling = nn.MaxPool1d(3, stride=3)
            self.avg_pooling = nn.AvgPool1d(3, stride=3)
        if emb_type.find("roberta") != -1:
            self.roberta_emb = RobertaEncode(self.d_k, emb_path, pretrain_dim)

        self.linear_0 = nn.Linear(d_a + d_e, d_k)
        torch.nn.init.xavier_uniform_(self.linear_0.weight)
        self.linear_1 = nn.Linear(d_a + d_e + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_2 = nn.Linear(4 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(4 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(d_e + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)
        self.linear_6 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_6.weight)
        self.linear_7 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_7.weight)
        self.linear_8 = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_8.weight)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, e_data, a_data, it_data=None, at_data=None, qtest=False):
        emb_type = self.emb_type
        batch_size, seq_len = e_data.size(0), e_data.size(1)
        e_embed_data = self.e_embed(e_data)
        if self.use_time:
            if at_data != None:
                at_embed_data = self.at_embed(at_data)
            it_embed_data = self.it_embed(it_data)
        a_data = a_data.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)
        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_question + 1, self.d_k)).repeat(batch_size, 1, 1).to(device)
        h_tilde_pre = None
        if emb_type == "qid":
            if self.use_time and at_data != None:
                all_learning = self.linear_1(torch.cat((e_embed_data, at_embed_data, a_data), 2))
            else:
                all_learning = self.linear_0(torch.cat((e_embed_data, a_data), 2))
        learning_pre = torch.zeros(batch_size, self.d_k).to(device)

        pred = torch.zeros(batch_size, seq_len).to(device)
        hidden_state = torch.zeros(batch_size, seq_len, self.d_k).to(device)

        for t in range(0, seq_len - 1):
            e = e_data[:, t]
            # q_e: (bs, 1, n_skill)
            q_e = self.q_matrix[e].view(batch_size, 1, -1).to(device)
            if self.use_time:
                it = it_embed_data[:, t]
                # Learning Module
                if h_tilde_pre is None:
                    c_pre = torch.unsqueeze(torch.sum(torch.squeeze(q_e,dim=1), 1),-1)
                    h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)/c_pre
                learning = all_learning[:, t]
                learning_gain = self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
                learning_gain = self.tanh(learning_gain)
                gamma_l = self.linear_3(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            else:
                # Learning Module
                if h_tilde_pre is None:
                    c_pre = torch.unsqueeze(torch.sum(torch.squeeze(q_e,dim=1), 1),-1)
                    h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)/c_pre
                learning = all_learning[:, t]
                learning_gain = self.linear_6(torch.cat((learning_pre, learning, h_tilde_pre), 1))
                learning_gain = self.tanh(learning_gain)
                gamma_l = self.linear_7(torch.cat((learning_pre, learning, h_tilde_pre), 1))                
            gamma_l = self.sig(gamma_l)
            LG = gamma_l * ((learning_gain + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))

            # Forgetting Module
            # h_pre: (bs, n_skill, d_k)
            # LG: (bs, d_k)
            # it: (bs, d_k)
            n_skill = LG_tilde.size(1)
            if self.use_time:
                gamma_f = self.sig(self.linear_4(torch.cat((
                    h_pre,
                    LG.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                    it.repeat(1, n_skill).view(batch_size, -1, self.d_k)
                ), 2)))
            else:
                gamma_f = self.sig(self.linear_8(torch.cat((
                    h_pre,
                    LG.repeat(1, n_skill).view(batch_size, -1, self.d_k)
                ), 2)))              
            h = LG_tilde + gamma_f * h_pre

            # Predicting Module
            c_tilde = torch.unsqueeze(torch.sum(torch.squeeze(self.q_matrix[e_data[:, t + 1]].view(batch_size, 1, -1),dim=1), 1),-1)
            h_tilde = self.q_matrix[e_data[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k)/c_tilde
            # print(f"h_tilde: {h_tilde.shape}")
            y = self.sig(self.linear_5(torch.cat((e_embed_data[:, t + 1], h_tilde), 1))).sum(1) / self.d_k
            pred[:, t + 1] = y
            hidden_state[:, t+1, :] = h_tilde

            # prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde
        if not qtest:
            return pred
        else:
            return pred, hidden_state[:,:-1,:], e_embed_data