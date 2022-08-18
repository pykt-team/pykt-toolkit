import os

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_

from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CDKVMN(Module):
    def __init__(self, num_c, num_q, dim_s, size_m, dropout=0.2, loss1=0.5,loss2=0.5,loss3=0.5,num_layers=1,nheads=5,start=50,
            emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "cdkvmn"
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.k_emb_layer = Embedding(self.num_c, self.dim_s)
            self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_c * 2, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.l1 = loss1
            self.l2 = loss2
            self.l3 = loss3
            num_layers = num_layers
            self.emb_size, self.hidden_size = dim_s, dim_s
            self.num_q, self.num_c = num_q, num_c
            
            if self.num_q > 0:
                self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2
            if self.emb_type.find("trans") != -1:
                self.nhead = nheads
                d_model = self.hidden_size# * 2
                encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
                encoder_norm = LayerNorm(d_model)
                self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
            elif self.emb_type.find("lstm") != -1:    
                self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            # self.qdrop = Dropout(dropout)
            self.qclasifier = Linear(self.hidden_size, self.num_c)
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size) # add concept emb
            self.closs = CrossEntropyLoss()
            # 加一个预测历史准确率的任务
            if self.emb_type.find("his") != -1:
                self.start = start
                self.hf_layer = Linear(self.hidden_size * 2, self.hidden_size)
                self.hisclasifier = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size//2), nn.ELU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_size//2, 1))
                self.hisloss = nn.MSELoss()

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def predcurc(self, qemb, cemb, xemb, dcur, train):
        y2 = 0
        sm, c, cshft = dcur["smasks"], dcur["cseqs"], dcur["shft_cseqs"]
        padsm = torch.ones(sm.shape[0], 1).to(device)
        sm = torch.cat([padsm, sm], dim=-1)
        c = torch.cat([c[:,0:1], cshft], dim=-1)
        chistory = xemb
        if self.emb_type.find("qemb") != -1 and self.num_q > 0:
            catemb = qemb + chistory
        else:
            catemb = chistory

        if self.emb_type.find("cemb") != -1: #akt本身就加了cemb
            catemb += cemb

        if self.emb_type.find("trans") != -1:
            mask = ut_mask(seq_len = catemb.shape[1])
            qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)
        else:
            qh, _ = self.qlstm(catemb)
        if train:
            start = 0
            cpreds = self.qclasifier(qh[:,start:,:])
            flag = sm[:,start:]==1
            y2 = self.closs(cpreds[flag], c[:,start:][flag])

        xemb = xemb + qh
        if self.emb_type.find("cemb") != -1: 
            xemb = xemb + cemb
        if self.emb_type.find("qemb") != -1:
            xemb = xemb+qemb
        
        return y2, xemb

    def predcurc2(self, qemb, cemb, xemb, dcur, train):
        y2 = 0
        sm, c, cshft = dcur["smasks"], dcur["cseqs"], dcur["shft_cseqs"]
        padsm = torch.ones(sm.shape[0], 1).to(device)
        sm = torch.cat([padsm, sm], dim=-1)
        c = torch.cat([c[:,0:1], cshft], dim=-1)

        catemb = cemb
        if self.num_q > 0:
            catemb += qemb

        if self.emb_type.find("trans") != -1:
            mask = ut_mask(seq_len = catemb.shape[1])
            qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)
        else:
            qh, _ = self.qlstm(catemb)
        if train:
            start = 0
            cpreds = self.qclasifier(qh[:,start:,:])
            flag = sm[:,start:]==1
            y2 = self.closs(cpreds[flag], c[:,start:][flag])

        xemb = xemb + qh# + cemb
        cemb = cemb + qh
        
        if self.emb_type.find("qemb") != -1 and self.num_q > 0:
            cemb = cemb+qemb
            xemb = xemb+qemb
        
        return y2, cemb, xemb

    def predhis(self, h, dcur):
        sm = dcur["smasks"]
        padsm = torch.ones(sm.shape[0], 1).to(device)
        sm = torch.cat([padsm, sm], dim=-1)

        # predict history correctness rates
        
        start = self.start
        rpreds = torch.sigmoid(self.hisclasifier(h)[:,start:,:]).squeeze(-1)
        rsm = sm[:,start:]
        rflag = rsm==1
        # rtrues = torch.cat([dcur["historycorrs"][:,0:1], dcur["shft_historycorrs"]], dim=-1)[:,start:]
        padr = torch.ones(h.shape[0], 1).to(device)
        rtrues = torch.cat([padr, dcur["historycorrs"]], dim=-1)[:,start:]
        # rtrues = dcur["historycorrs"][:,start:]
        # rtrues = dcur["totalcorrs"][:,start:]
        # print(f"rpreds: {rpreds.shape}, rtrues: {rtrues.shape}")
        y3 = self.hisloss(rpreds[rflag], rtrues[rflag])

        # h = self.dropout_layer(h)
        # y = torch.sigmoid(self.out_layer(h))
        return y3

    def forward(self, dcur, qtest=False, train=False):
        y2, y3 = 0, 0
        q, r = dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, rshft = dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        q, r = torch.cat([q[:,0:1], qshft], dim=-1), torch.cat([r[:,0:1], rshft], dim=-1)

        qs, qshfts = dcur["qseqs"].long(), dcur["shft_qseqs"].long()
        qs = torch.cat([qs[:,0:1], qshfts], dim=-1)

        emb_type = self.emb_type
        batch_size = q.shape[0]
        if emb_type == "qid":
            x = q + self.num_c * r
            k = self.k_emb_layer(q)
            v = self.v_emb_layer(x)
        elif emb_type.endswith("predcurc"):
            x = q + self.num_c * r
            k = self.k_emb_layer(q)
            v = self.v_emb_layer(x)
            # predict concept
            qemb = self.question_emb(qs)

            # predcurc(self, qemb, cemb, xemb, dcur, train):
            cemb = k
            if emb_type.find("noxemb") != -1:
                y2, k, v = self.predcurc2(qemb, cemb, v, dcur, train)
            else:
                y2, v = self.predcurc(qemb, cemb, v, dcur, train)
        
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        info = torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
        f = torch.tanh(self.f_layer(info))

        if emb_type.find("his") != -1:
            hf = torch.tanh(self.hf_layer(info))
            y3 = self.predhis(hf, dcur)
            
        p = self.p_layer(self.dropout_layer(f))

        p = torch.sigmoid(p)
        # print(f"p: {p.shape}")
        p = p.squeeze(-1)
        if train:
            return p, y2, y3
        else:
            if not qtest:
                return p
            else:
                return p, f