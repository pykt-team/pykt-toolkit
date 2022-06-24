import os

import numpy as np
import torch
from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DKTRasch(Module):
    def __init__(self, num_c, num_q, emb_size, dropout=0.1, emb_type='qid', emb_path="", gamma=0.03, pretrain_dim=768, separate_qa=False, use_interac=False, use_qmatrix=False, qmatrix=None, kt_state=False):
        super().__init__()
        self.use_interac = use_interac
        self.use_qmatrix = use_qmatrix
        self.kt_state = kt_state
        print(f"kt_state: {self.kt_state}") 
        if not self.use_interac and not self.use_qmatrix and not self.kt_state:
            self.model_name = "dkt_rasch"
        elif self.use_interac and not self.use_qmatrix and not self.kt_state:
            self.model_name = "dkt_interac"
        elif not self.use_interac and self.use_qmatrix and not self.kt_state:
            self.model_name = "dkt_qmatrix"
        elif not self.use_interac and self.use_qmatrix and self.kt_state:
            self.model_name = "dkt_mastery"
        self.num_c = num_c
        self.n_question= num_c
        # print(f"n_question: {self.n_question}")        
        self.n_pid = num_q
        # print(f"n_pid: {self.n_pid}")
        self.emb_size = emb_size
        self.hidden_size = emb_size
        embed_l = emb_size
        self.emb_type = emb_type
        self.separate_qa = separate_qa
        self.qmatrix = qmatrix
        self.gamma = gamma

        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1) # 题目难度
            if self.use_qmatrix:
                qmatrix[qmatrix==0] = gamma
                self.q_embed_diff = Embedding.from_pretrained(qmatrix, freeze=False)
            else:
                self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上
            if self.kt_state:
                self.h0 = nn.Parameter(torch.Tensor(self.n_question + 1, embed_l))

        if emb_type.startswith("qid"):
            # self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question+1, embed_l)
            if self.separate_qa: 
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l) # interaction emb
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        self.out_layer1 = Linear(self.hidden_size * 2, self.num_c)
        self.linear1 = Linear(self.hidden_size * 2, self.hidden_size)
        
    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, pid_data=None, res=False):
        emb_type = self.emb_type
        batch_size = q_data.shape[0]
        if emb_type == "qid":
            # x = q + self.num_c * r
            # xemb = self.interaction_emb(x)
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        if self.n_pid > 0: # have problem id
            if self.use_qmatrix:
                pid_embed_data = self.q_embed_diff(pid_data)
                q_embed_diff_data = torch.nn.functional.softmax(pid_embed_data,-1)
                # print(f"q_embed_diff_data: {q_embed_diff_data.shape}")
                q_embed_diff_data = torch.einsum('bij, jk -> bik', q_embed_diff_data, self.q_embed.weight)
                if self.kt_state:
                    # print(f"pid_embed_data:{pid_embed_data.shape}")
                    h_pre = self.h0.unsqueeze(0).repeat(batch_size, 1, 1)
                    # print(f"h_pre:{h_pre.shape}")
                    h_tilde = pid_embed_data.bmm(h_pre)
                    # print(f"h_tilde:{h_tilde.shape}")
                    # print(f"qa_embed_data:{qa_embed_data.shape}")
                    # qa_embed_data = qa_embed_data + h_tilde

            else:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder
            if not self.use_interac:
                qa_embed_diff_data = self.qa_embed_diff(target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            else:
                interac_data = q_data + self.num_c * target
                qa_embed_diff_data = self.qa_embed_diff(interac_data)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）

        if self.kt_state:
            qa_embed_data = torch.cat([qa_embed_data, h_tilde], dim=-1)
            qa_embed_data = self.linear1(qa_embed_data)

        xemb = qa_embed_data
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        if res:
            y = self.out_layer1(torch.cat([h, q_embed_data], dim=-1))
        else:
            y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y