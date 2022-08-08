# coding: utf-8
from tkinter.tix import DirTree
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from .utils import ut_mask
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from .utils import transformer_FFN, ut_mask, pos_encode
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CATKT(nn.Module):
    def __init__(self, num_q, num_c, skill_dim, answer_dim, hidden_dim, attention_dim=80, epsilon=10, beta=0.2, dropout=0.2, emb_type="qid", 
            num_layers=1, num_attn_heads=5, l1=0.5, l2=0.5, l3=0.5, start=50,
            emb_path="", fix=True):
        super(CATKT, self).__init__()
        self.model_name = "catkt"
        self.fix = fix
        print(f"fix: {fix}")
        # if self.fix == True:
        #     self.model_name = "catktfix"
        self.emb_type = emb_type
        self.skill_dim=skill_dim
        self.answer_dim=answer_dim
        self.hidden_dim = hidden_dim
        self.num_q = num_q
        self.num_c = num_c
        self.epsilon = epsilon
        self.beta = beta
        self.rnn = nn.LSTM(self.skill_dim+self.answer_dim, self.hidden_dim, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_dim*2, self.num_c)
        self.sig = nn.Sigmoid()
        
        self.skill_emb = nn.Embedding(self.num_c+1, self.skill_dim)
        self.skill_emb.weight.data[-1]= 0
        
        self.answer_emb = nn.Embedding(2+1, self.answer_dim)
        self.answer_emb.weight.data[-1]= 0
        
        self.attention_dim = attention_dim
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)

        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.l1 = l1
            self.l2 = l2
            self.l3 = l3
            # self.emb_size = self.skill_dim+self.answer_dim # xemb = concat
            self.hidden_size = self.hidden_dim
            if self.num_q > 0:
                if self.emb_type.find("addq") != -1:
                    self.question_emb = Embedding(self.num_q, self.skill_dim+self.answer_dim)
                else:
                    self.question_emb = Embedding(self.num_q, self.skill_dim) # 1.2
            if self.emb_type.find("trans") != -1:
                self.nhead = num_attn_heads
                if self.emb_type.find("catq") != -1:
                    d_model = self.skill_dim*2+self.answer_dim# * 2
                else:
                    d_model = self.skill_dim + self.answer_dim
                encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
                encoder_norm = LayerNorm(d_model)
                self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
                self.qdrop = Dropout(dropout)
                self.qclasifier = Linear(d_model, self.num_c)
                if self.emb_type.find("catq") != -1:
                    self.rnn = nn.LSTM(d_model, self.hidden_dim, batch_first=True)
                # self.fc = nn.Linear(self.hidden_dim*2, self.num_c)

            self.closs = CrossEntropyLoss()
            # 加一个预测历史准确率的任务
            if self.emb_type.find("his") != -1:
                self.start = start
                self.hisclasifier = nn.Sequential(
                    nn.Linear(self.hidden_dim*2, self.hidden_dim), nn.ELU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_dim, 1))
                self.hisloss = nn.MSELoss()

    
    def attention_module(self, lstm_output):
        # lstm_output = lstm_output[0:1, :, :]
        # print(f"lstm_output: {lstm_output.shape}")
        att_w = self.mlp(lstm_output)
        # print(f"att_w: {att_w.shape}")
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        # print(f"att_w: {att_w.shape}")

        if self.fix == True:
            attn_mask = ut_mask(lstm_output.shape[1])
            att_w = att_w.transpose(1,2).expand(lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[1]).clone()
            att_w = att_w.masked_fill_(attn_mask, float("-inf"))
            alphas = torch.nn.functional.softmax(att_w, dim=-1)
            attn_ouput = torch.bmm(alphas, lstm_output)
        else: # 原来的官方实现
            alphas=nn.Softmax(dim=1)(att_w)
            # print(f"alphas: {alphas.shape}")    
            attn_ouput = alphas*lstm_output # 整个seq的attn之和为1，计算前面的的时候，所有的attn都<<1，不会有问题？做的少的时候，历史作用小，做得多的时候，历史作用变大？
            # print(f"attn_ouput: {attn_ouput.shape}")
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        # print(f"attn_ouput: {attn_ouput}")
        # print(f"attn_output_cum: {attn_output_cum}")
        attn_output_cum_1=attn_output_cum-attn_ouput
        # print(f"attn_output_cum_1: {attn_output_cum_1}")
        # print(f"lstm_output: {lstm_output}")

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        # import sys
        # sys.exit()

        return final_output

    def predcurc(self, qemb, cemb, xemb, dcur, train):
        y2 = 0
        sm, c = dcur["smasks"], dcur["cseqs"]
        catemb = torch.cat([xemb, qemb], dim=-1)
        # if self.emb_type.find("cemb") != -1: # cemb已经concat到xemb里面了
        #     catemb += cemb

        if self.emb_type.find("trans") != -1:
            # print(f"catemb: {catemb.shape}")
            mask = ut_mask(seq_len = catemb.shape[1])
            qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)

        if train:
            start = 0
            cpreds = self.qclasifier(qh[:,start:,:])
            flag = sm[:,start:]==1
            y2 = self.closs(cpreds[flag], c[:,start:][flag])

        # xemb: skill+r, qh: 2*skill+r
        xemb = catemb + qh
        
        return y2, xemb
    # def predcurc(self, qemb, cemb, xemb, dcur, train):
    #     y2 = 0
    #     sm, c = dcur["smasks"], dcur["cseqs"]
    #     catemb = xemb
    #     if self.emb_type.find("trans") != -1:
    #         # print(f"catemb: {catemb.shape}")
    #         mask = ut_mask(seq_len = catemb.shape[1])
    #         qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)

    #     if train:
    #         start = 0
    #         cpreds = self.qclasifier(qh[:,start:,:])
    #         flag = sm[:,start:]==1
    #         y2 = self.closs(cpreds[flag], c[:,start:][flag])

    #     # xemb: skill+r, qh: 2*skill+r
    #     xemb = catemb + qh
        
    #     return y2, xemb

    # def predcurc(self, qemb, cemb, xemb, dcur, train): # addq
    #     y2 = 0
    #     sm, c = dcur["smasks"], dcur["cseqs"]
    #     catemb = xemb + qemb
    #     if self.emb_type.find("trans") != -1:
    #         # print(f"catemb: {catemb.shape}")
    #         mask = ut_mask(seq_len = catemb.shape[1])
    #         qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)

    #     if train:
    #         start = 0
    #         cpreds = self.qclasifier(qh[:,start:,:])
    #         flag = sm[:,start:]==1
    #         y2 = self.closs(cpreds[flag], c[:,start:][flag])

    #     # xemb: skill+r, qh: 2*skill+r
    #     xemb = catemb + qh
        
    #     return y2, xemb

    def predhis(self, h, dcur):
        sm = dcur["smasks"]

        # predict history correctness rates
        # print(f"h: {h.shape}")
        start = self.start
        rpreds = torch.sigmoid(self.hisclasifier(h)[:,start:,:]).squeeze(-1)
        rsm = sm[:,start:]
        rflag = rsm==1
        rtrues = dcur["historycorrs"][:,start:]
        # rtrues = dcur["historycorrs"][:,start:]
        # rtrues = dcur["totalcorrs"][:,start:]
        # print(f"rpreds: {rpreds.shape}, rtrues: {rtrues.shape}")
        y3 = self.hisloss(rpreds[rflag], rtrues[rflag])

        # h = self.dropout_layer(h)
        # y = torch.sigmoid(self.out_layer(h))
        return y3

    def forward(self, dcur, perturbation=None, train=False):
        question, skill, answer = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        sm = dcur["smasks"].long()
        y2, y3 = 0, 0

        emb_type = self.emb_type
        r = answer
        
        skill_embedding=self.skill_emb(skill)
        answer_embedding=self.answer_emb(answer)
        
        skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
        answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
        answer=answer.unsqueeze(2).expand_as(skill_answer)
        
        skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)
        
        # print(skill_answer_embedding)
        
        skill_answer_embedding1=skill_answer_embedding
        if  perturbation is not None:
            skill_answer_embedding += perturbation

        if emb_type == "qid":
            out,_ = self.rnn(skill_answer_embedding)
            # print(f"out: {out.shape}")
            out=self.attention_module(out)
            # print(f"after attn out: {out.shape}")
            res = self.sig(self.fc(self.dropout_layer(out)))

        elif emb_type.find("predcurc") != -1:
            cemb, xemb = skill_embedding, skill_answer_embedding
            if self.num_q > 0:
                qemb = self.question_emb(question)
            
            # predcurc(self, qemb, cemb, xemb, dcur, train)
            y2, skill_answer_embedding = self.predcurc(qemb, cemb, xemb, dcur, train)
            
            out,_ = self.rnn(skill_answer_embedding)
            # print(f"out: {out.shape}")
            out=self.attention_module(out)
            if emb_type.find("his") != -1:
                y3 = self.predhis(out, dcur)
            # print(f"after attn out: {out.shape}")
            res = self.sig(self.fc(self.dropout_layer(out)))

        # res = res[:, :-1, :]
        # pred_res = self._get_next_pred(res, skill)
        if train:
            return res, skill_answer_embedding1, y2, y3
        else:
            return res, skill_answer_embedding1

from torch.autograd import Variable

def _l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)
