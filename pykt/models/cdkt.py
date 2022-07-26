from cmath import log
from curses.ascii import EM
import os
from tkinter import N

import pandas as pd
import torch

from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from .utils import transformer_FFN, ut_mask, pos_encode
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

device = "cpu" if not torch.cuda.is_available() else "cuda"

class CDKT(Module):
    def __init__(self, num_q, num_c, seq_len, emb_size, dropout=0.1, emb_type='qid', num_layers=1, num_attn_heads=5, l1=0.5, l2=0.5, l3=0.5, start=50, emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "cdkt"
        print(f"qnum: {num_q}, cnum: {num_c}")
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)

        if self.emb_type.endswith("pretrainddiff"): # use pretrained qemb and cemb from cc
            dpath = "/data/liuqiongqiong/kt/kaiyuan/dev/algebra2005_1024_350"
            qavgkc = pd.read_pickle(os.path.join(dpath, "algebra2005_qavgcvec.pkl"))
            qvec = pd.read_pickle(os.path.join(dpath, "algebra2005_questionvec.pkl"))
            cvec = pd.read_pickle(os.path.join(dpath, "algebra2005_conceptvec.pkl"))
            qdifficulty = pd.read_pickle(os.path.join(dpath, "algebra2005_eachqdifficulty.pkl"))
            self.pretrain_qemb = Embedding.from_pretrained(qvec)
            self.pretrain_cemb = Embedding.from_pretrained(cvec)
            self.pretrain_qavgcemb = Embedding.from_pretrained(qavgkc)
            self.pretrain_qdifficulty = Embedding.from_pretrained(qdifficulty)

            self.qlinear = Linear(qvec.shape[1], self.emb_size)
            self.clinear = Linear(cvec.shape[1], self.emb_size)
            # self.qclinear = Linear(qavgkc.shape[1], self.emb_size)
            # self.dlinear = Linear(qdifficulty.shape[1], self.emb_size)
            for param in self.pretrain_qemb.parameters():
                param.requires_grad = False
            for param in self.pretrain_cemb.parameters():
                param.requires_grad = False
            for param in self.pretrain_qavgcemb.parameters():
                param.requires_grad = False
            for param in self.pretrain_qdifficulty.parameters():
                param.requires_grad = False
            if self.emb_type.find("predcurc") != -1:
                self.l1 = l1
                self.l2 = l2
                ## learning
                self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2
                self.concept_emb = Embedding(self.num_c, self.emb_size) # add concept emb

                if self.emb_type.find("catr") != -1:
                    self.lstm_layer = LSTM(self.emb_size*2, self.hidden_size, batch_first=True)
                if self.emb_type.find("addr") != -1:
                    self.response_emb = Embedding(2, self.emb_size)
                self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)

                # self.qdrop = Dropout(dropout)
                self.qclasifier = Linear(self.hidden_size, self.num_c)
                self.closs = CrossEntropyLoss()

        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.l1 = l1
            self.l2 = l2
            if self.num_q > 0:
                self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2
            if self.emb_type.find("trans") != -1:
                self.nhead = num_attn_heads
                d_model = self.hidden_size# * 2
                encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
                encoder_norm = LayerNorm(d_model)
                self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
                if self.emb_type.find("addpos") != -1:
                    self.position_emb = Embedding(seq_len, emb_size)
            else:    
                self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.qdrop = Dropout(dropout)
            self.qclasifier = Linear(self.hidden_size, self.num_c)
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size) # add concept emb

            if self.emb_type.find("attn") != -1:
                # self.qlinear = Linear(self.hidden_size, self.hidden_size)
                # self.klinear = Linear(self.hidden_size, self.hidden_size)
                # self.vlinear = Linear(self.hidden_size, self.hidden_size)
                self.dF, self.dfenzi, self.dfenmu = dict(), dict(), dict()
                self.avgf = 0
                if self.emb_type.find("mattn") != -1:
                    self.mattn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
                    self.mattn_dropout = Dropout(dropout)
                    self.mattn_layer_norm = LayerNorm(emb_size)
                if self.emb_type.find("areaattn") != -1:
                    from .cdkt_utils import AreaAttention, MultiHeadAreaAttention 
                    area_core = AreaAttention(self.hidden_size, 4, dropout)
                    self.area_attn = MultiHeadAreaAttention(area_core, 1, self.hidden_size,
                        self.hidden_size, self.hidden_size, self.hidden_size)

            # concat response
            if self.emb_type.find("catr") != -1:
                self.lstm_layer = LSTM(self.emb_size*2, self.hidden_size, batch_first=True)
            if self.emb_type.find("addr") != -1:
                self.response_emb = Embedding(2, self.emb_size)
            if self.emb_type.find("predr") != -1:
                self.response_emb = Embedding(2, self.emb_size)
                self.rclasifier = Linear(self.hidden_size, 1)
                # self.rloss = Bina
            self.closs = CrossEntropyLoss()
            # 加一个预测历史准确率的任务
            if self.emb_type.find("his") != -1:
                self.start = start
                self.hisclasifier = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size//2), nn.ELU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_size//2, 1))
                self.hisloss = nn.MSELoss()
            if self.emb_type.find("diff") != -1:
                self.difficulty_layer = Embedding(11, self.emb_size)
            if self.emb_type.find("bkt") != -1:
                self.bkt = nn.Parameter(torch.randn(self.num_c, 2).to(device), requires_grad=True)

            # 
        if self.emb_type.endswith("addcshft"):
            self.concept_emb = Embedding(self.num_c, self.emb_size) # add concept emb
            self.lstm_layer = LSTM(self.emb_size*2, self.hidden_size, batch_first=True)
            if self.emb_type.find("attn") != -1:
                self.dF, self.dfenzi, self.dfenmu = dict(), dict(), dict()
                self.avgf = 0
                self.mattn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
                self.mattn_dropout = Dropout(dropout)
                self.mattn_layer_norm = LayerNorm(emb_size)
            self.out_layer = nn.Sequential(
                    nn.Linear(self.hidden_size,
                            self.hidden_size), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_size, 256), nn.ReLU(
                    ), nn.Dropout(dropout),
                    nn.Linear(256, 1)
                )

        if self.emb_type.endswith("seq2seq"):
            self.l1 = l1
            self.l2 = l2
            self.concept_emb = nn.Parameter(torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True)
            self.question_emb = Embedding(self.num_q, self.emb_size)
            if self.emb_type.find("addr") != -1:
                self.response_emb = Embedding(2, self.emb_size)
            
            self.nhead = 5
            d_model = self.hidden_size# * 2
            encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
            encoder_norm = LayerNorm(d_model)
            # if self.emb_type.find("addbase") != -1:
            #     self.basenet = TransformerEncoder(encoder_layer, num_layers=1, norm=encoder_norm)
            self.net = TransformerEncoder(encoder_layer, num_layers=num_layers+1, norm=encoder_norm)
            self.position_emb = Embedding(seq_len, emb_size)
            self.qnet = nn.Sequential(
                nn.Linear(d_model,
                        self.hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.emb_size), nn.ReLU(
                ), nn.Dropout(dropout)
            )
            self.qclasifier = nn.Linear(self.emb_size, self.num_c)
            self.closs = MultiLabelSoftMarginLoss()# MultiLabelMarginLoss()
            #  MultiLabelMarginLoss会导致结果不可复现
            if self.emb_type.find("transpredr") != -1:
                self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)

        if self.emb_type.endswith("mergetwo"):
            self.l1, self.l2, self.l3 = l1, l2, l3
            self.nhead = 5
            d_model = self.hidden_size# * 2
            encoder_layer1 = TransformerEncoderLayer(d_model, nhead=self.nhead)
            encoder_norm1 = LayerNorm(d_model)

            self.concept_emb = nn.Parameter(torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True)
            if self.num_q > 0:
                self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2

            if self.emb_type.find("trans") != -1:
                self.qnet1 = TransformerEncoder(encoder_layer1, num_layers=2, norm=encoder_norm1)
            else:    
                self.qnet1 = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            # self.qdrop1 = Dropout(dropout)                
            if self.emb_type.find("match") != -1: # concat(h, c) -> predict match or not?
                self.qclasifier1 = nn.Sequential(
                    nn.Linear(d_model+self.emb_size, self.hidden_size), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_size, int(self.emb_size/2)), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(int(self.emb_size/2), 2)
                )
                self.closs1 = BCELoss()
            else:
                self.qclasifier1 = Linear(self.hidden_size, self.num_c)
                if self.emb_type.find("predr") != -1:
                    # self.response_emb = Embedding(2, self.emb_size)
                    self.rclasifier1 = Linear(self.hidden_size, self.num_c)
                self.closs1 = CrossEntropyLoss()
            # seq2seq
            self.position_emb = Embedding(seq_len, emb_size)
            encoder_layer2 = TransformerEncoderLayer(d_model, nhead=self.nhead)
            encoder_norm2 = LayerNorm(d_model)
            self.base_qnet21 = TransformerEncoder(encoder_layer2, num_layers=1, norm=encoder_norm2)
            self.base_qnet22 = TransformerEncoder(encoder_layer2, num_layers=num_layers, norm=encoder_norm2)
            self.qnet2 = nn.Sequential(
                nn.Linear(d_model, self.hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.emb_size), nn.ReLU(), nn.Dropout(dropout)
            )
            self.qclasifier2 = nn.Linear(self.emb_size, self.num_c)
            self.closs2 = MultiLabelSoftMarginLoss()# MultiLabelMarginLoss()

    def get_avg_skill_emb(self, c):
        # add zero for padding
        concept_emb_cat = torch.cat(
            [torch.zeros(1, self.emb_size).to(device), 
            self.concept_emb], dim=0)
        # shift c
        related_concepts = (c+1).long()
        #[batch_size, seq_len, emb_dim]
        concept_emb_sum = concept_emb_cat[related_concepts, :].sum(
            axis=-2).to(device)

        #[batch_size, seq_len,1]
        concept_num = torch.where(related_concepts != 0, 1, 0).sum(
            axis=-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num).to(device)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def attn(self, lstm_output, cs, rs):
        bz, seqlen, dim = lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[2]
        # lstm_output = torch.rand(2,seq_len,4)
        attn_mask = torch.triu(torch.ones(seqlen,seqlen),diagonal=1).to(dtype=torch.bool).to(device)

        # q, k, v = self.qlinear(lstm_output), self.klinear(lstm_output), self.vlinear(lstm_output)
        q, k, v = lstm_output, lstm_output, lstm_output

        att_w = torch.matmul(q, k.transpose(1,2))# / math.sqrt(dim)
        # change attn weight
        if self.emb_type.find("matmulforget") != -1:
            fss = self.generate_forget(cs, rs)
            # print(f"att_w: {att_w.shape}, fss: {fss.shape}")
            att_w = att_w * fss
        #
        att_w = att_w.transpose(1,2).expand(lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[1]).clone()
        att_w = att_w.masked_fill_(attn_mask, float("-inf"))
        alphas = torch.nn.functional.softmax(att_w, dim=-1)
        if self.emb_type.find("softmaxforget") != -1:
            fss = self.generate_forget(cs, rs)
            # print(f"alphas: {alphas.shape}, fss: {fss.shape}")
            alphas = alphas * fss
        attn_ouput = torch.bmm(alphas, v)
        return attn_ouput

    def multi_head_attn(self, q, k, v):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm 
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(seq_len = k.shape[0])
        attn_emb, _ = self.mattn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.mattn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.mattn_layer_norm(q + attn_emb)
        return attn_emb

    def area_attn_fuc(self, q, k, v):
        # q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        causal_mask = ut_mask(seq_len = k.shape[1]).unsqueeze(0)
        S = self.area_attn(q, k, v, attn_mask=causal_mask)#.permute(1,0,2)
        return S

    def forward(self, dcur, train=False): ## F * xemb
        # print(f"keys: {dcur.keys()}")
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        sm = dcur["smasks"].long()
        y2, y3 = 0, 0

        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            x = c + self.num_c * r
            xemb = self.interaction_emb(x)
            
        if emb_type == "qid":
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("addcshft"):
            cshft, rshft = dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
            cemb, cshftemb = self.concept_emb(c), self.concept_emb(cshft)
            h, _ = self.lstm_layer(torch.cat([cshftemb, xemb], dim=-1))
            if emb_type.find("attn") != -1:
                # h = generate_forget(c, r) * h
                if emb_type.find("mattn") != -1:
                    h = self.multi_head_attn(cshftemb, cemb, xemb)
                else:
                    h = self.attn(h, cshft, rshft) ### 
            # h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h).squeeze(-1)) 
        elif emb_type.endswith("pretrainddiff"): # use pretrained difficulty for each question
            qemb, cemb, qavgcemb, qdiff = self.pretrain_qemb(q), self.pretrain_cemb(c), self.pretrain_qavgcemb(q), self.pretrain_qdifficulty(q)
            # qemb, cemb, qavgcemb, qdiff = self.qlinear(qemb), self.clinear(cemb), self.qclinear(qavgcemb), self.dlinear(qdiff)
            if emb_type.find("sep") != -1:
                xemb = xemb + qemb + cemb + qdiff
            elif emb_type.find("qavgc") != -1:
                xemb = xemb + qavgcemb + qdiff
            elif emb_type.find("all") != -1: # use all
                xemb = xemb + qemb + cemb + qavgcemb + qdiff
            elif emb_type.find("onlydiff") != -1:
                xemb = xemb + qdiff
            elif emb_type.find("onlycdiff") != -1:
                xemb = xemb + cemb + qdiff
            elif emb_type.find("onlyc") != -1:
                xemb = xemb + cemb
            elif emb_type.find("onlyqc") != -1:
                xemb = xemb + qemb + cemb
            if emb_type.find("predcurc") != -1:
                qemb2, cemb2 = self.question_emb(q), self.concept_emb(c)
                catemb = xemb + qemb2 + cemb2
                if emb_type.find("caddr") != -1:
                    remb = self.response_emb(r)
                    catemb += remb
                qh, _ = self.qlstm(catemb)
                cpreds = self.qclasifier(qh)
                flag = sm==1
                y2 = self.closs(cpreds[flag], c[flag])
                # predict response
                xemb = xemb + qh + cemb2# + cemb ## +cemb效果更好
                if emb_type.find("catr") != -1:
                    remb = r.float().unsqueeze(2).expand(xemb.shape[0], xemb.shape[1], xemb.shape[2])
                    xemb = torch.cat([xemb, remb], dim=-1)
                elif emb_type.find("addr") != -1:
                    remb = self.response_emb(r)
                    xemb = xemb + remb

            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("seq2seq"): # add transformer to predict multi label cs
            posemb = self.position_emb(pos_encode(xemb.shape[1]))
            if train:
                oriqs, orics, orisms = dcur["oriqs"].long(), dcur["orics"].long(), dcur["orisms"].long()#self.generate_oriqcs(q, c, sm)
                # oriqs, orics, orisms = self.generate_oriqcs(q, c, sm, is_repeat)
                # print(f"oriqs: {oriqs.shape}, orics: {orics.shape}, orisms: {orisms.shape}")
                concept_avg = self.get_avg_skill_emb(orics)
                qemb = self.question_emb(oriqs)
                # print(f"concept_avg: {concept_avg.shape}, qemb: {qemb.shape}")
                que_c_emb = concept_avg + qemb + posemb#torch.cat([concept_avg, qemb],dim=-1)

                # add mask
                # mask = self.get_attn_pad_mask(orisms)
                mask = ut_mask(seq_len = que_c_emb.shape[1])
                # if self.emb_type.find("linear") != -1:
                # que_c_emb = self.que_c_linear(que_c_emb)
                # print(f"que_c_emb: {que_c_emb.shape}, mask: {mask.shape}")
                
                # if emb_type.find("addbase") != -1:
                #     que_c_emb = self.basenet(que_c_emb.transpose(0,1), mask).transpose(0,1)
                    # que_c_emb += posemb
                qh = self.net(que_c_emb.transpose(0,1), mask).transpose(0,1)
                qh = self.qnet(qh)
            
                cpreds = torch.sigmoid(self.qclasifier(qh))
                flag = orisms==1
                masked = cpreds[flag]
                pad = torch.ones(cpreds.shape[0], cpreds.shape[1], self.num_c-10).to(device)
                pad = -1 * pad
                ytrues = torch.cat([orics, pad], dim=-1).long()[flag]
                y2 = self.closs(masked, ytrues)

            repqemb = self.question_emb(q)
            repcemb = self.concept_emb[c]
            xemb = xemb + repqemb + repcemb
            # if emb_type.find("addbase") != -1:
                # qcemb = repcemb+repqemb
            qcemb = repcemb+repqemb+posemb#torch.cat([repcemb, repqemb], dim=-1)
            # qcmask = self.get_attn_pad_mask(sm)
            qcmask = ut_mask(seq_len = qcemb.shape[1])
            # qcemb = self.basenet(qcemb.transpose(0,1), qcmask).transpose(0,1)
            # qcemb += posemb
            qcemb = self.net(qcemb.transpose(0,1), qcmask).transpose(0,1)
            qcemb = self.qnet(qcemb)
            xemb = xemb + qcemb
            
            if emb_type.find("addr") != -1:
                cremb = self.response_emb(r)
                xemb += cremb
            if emb_type.find("transpredr") != -1:
                mask = ut_mask(seq_len = xemb.shape[1])
                h = self.trans(xemb.transpose(0,1), mask).transpose(0,1)
                h = self.dropout_layer(h)
                y = torch.sigmoid(self.out_layer(h))
            else:
                h, _ = self.lstm_layer(xemb)
                h = self.dropout_layer(h)
                y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("predcurc"): # predict current question' current concept
            # predict concept
            # pad = torch.zeros(xemb.shape[0], 1, xemb.shape[2]).to(device)
            # chistory = torch.cat((pad, xemb[:,0:-1,:]), dim=1)
            chistory = xemb
            if self.num_q > 0:
                qemb = self.question_emb(q)
                catemb = qemb + chistory
            else:
                catemb = chistory
            if emb_type.find("cemb") != -1:
                cemb = self.concept_emb(c)
                catemb += cemb

            if emb_type.find("diff") != -1:
                curdiff = dcur["difficulty"]
                diffemb = self.difficulty_layer(curdiff)
                catemb += diffemb
            # cemb = self.concept_emb(c)
            # catemb = cemb
            if emb_type.find("caddr") != -1:
                remb = self.response_emb(r)
                catemb += remb
            if emb_type.find("trans") != -1:
                if emb_type.find("addpos") != -1: # 不加pos效果更好
                    posemb = self.position_emb(pos_encode(xemb.shape[1]))
                    catemb = catemb + posemb
                mask = ut_mask(seq_len = catemb.shape[1])
                qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)
            else:
                qh, _ = self.qlstm(catemb)
            if train:
                start = 0
                cpreds = self.qclasifier(qh[:,start:,:])
                flag = sm[:,start:]==1
                y2 = self.closs(cpreds[flag], c[:,start:][flag])
                # if emb_type.find("predr") != -1:
                #     rpreds = torch.sigmoid(self.rclasifier(qh))
                #     y2 = y2+self.closs(rpreds[flag], r[flag])

            # predict response
            xemb = xemb + qh + cemb
            if emb_type.find("diff") != -1:
                xemb = xemb + diffemb# + qemb
            if emb_type.find("catr") != -1:
                remb = r.float().unsqueeze(2).expand(xemb.shape[0], xemb.shape[1], xemb.shape[2])
                xemb = torch.cat([xemb, remb], dim=-1)
                # remb = torch.tensor([0]).unsqueeze(1).expand_as(xemb).to(device)
                # kc_response = torch.cat((xemb,remb), 2)
                # response_kc = torch.cat((remb,xemb), 2)
                # r = r.unsqueeze(2).expand_as(kc_response)
                # xemb = torch.where(r == 1, kc_response, response_kc)
            if emb_type.find("addr") != -1:
                remb = self.response_emb(r)
                xemb = xemb + remb

            h, _ = self.lstm_layer(xemb)

            if emb_type.find("hattn") != -1:
                h = self.generate_forget(c, r) * h
            elif emb_type.find("mattn") != -1:
                h = self.multi_head_attn(h, h, h)
            elif emb_type.find("areaattn") != -1:
                h = self.area_attn_fuc(h, h, h)
            elif emb_type.find("attn") != -1:
                h = self.attn(h, c, r)# + h

            # predict history correctness rates
            if emb_type.find("his") != -1:
                start = self.start
                rpreds = torch.sigmoid(self.hisclasifier(h)[:,start:,:]).squeeze(-1)
                rsm = sm[:,start:]
                rflag = rsm==1
                rtrues = dcur["historycorrs"][:,start:]
                # rtrues = dcur["totalcorrs"][:,start:]
                # print(f"rpreds: {rpreds.shape}, rtrues: {rtrues.shape}")
                y2 = y2+self.hisloss(rpreds[rflag], rtrues[rflag])
            if emb_type.find("predr") != -1:
                rpreds = torch.sigmoid(self.rclasifier(h).squeeze(-1))
                flag = sm==1
                y2 = y2+binary_cross_entropy(rpreds[flag].double(), r[flag].double())
                # assert False
            # predict response
            h = self.dropout_layer(h)
            # y = torch.sigmoid(self.out_layer(h))
            y = self.out_layer(h)
            # add IRT
            
            if emb_type.find("bkt") != -1:
                cshft = dcur["shft_cseqs"].long()
                ypreds = (y * one_hot(cshft.long(), self.num_c)).sum(-1)
                slipshft, guessshft = dcur["shft_slipping"], dcur["shft_guess"]
                # print(f"ypreds: {ypreds.shape}, slipshft: {slipshft.shape}, guessshft: {guessshft.shape}")
                # print(f"slipping: {slipshft}")
                # print(f"guess: {guessshft}")
                ypreds = torch.sigmoid(ypreds)
                # y = ypreds*(1-slipshft) + (1-ypreds)*guessshft)
                # print(f"self.bkt[cshft][0]: {self.bkt[cshft].shape}, {self.bkt[cshft][:,:,0].shape}")
                # TODO 根据上一次当前知识点的response来将01或者11融入模型
                y = ypreds*(1-slipshft*torch.sigmoid(self.bkt[cshft][:,:,0])) + \
                        (1-ypreds)*guessshft*torch.sigmoid(self.bkt[cshft][:,:,0]) # 0.8238
                # y = torch.sigmoid(ypreds*(1-slipshft)*self.bkt[cshft][:,:,0] + (1-ypreds)*guessshft*self.bkt[cshft][:,:,1]) # 0.8218
                # print(f"ypreds: {ypreds}")
                # print(f"yfinal: {y}")
                # print(f"ytrues: {dcur['shft_rseqs']}")
                # assert False
                # y = torch.sigmoid()
                # TODO 1-first/全局准确率 当作难度 bkt后再irt？
            else:
                y = torch.sigmoid(y)
                
            
        elif emb_type.endswith("mergetwo"):
            if self.num_q > 0:
                repqemb = self.question_emb(q)
            repcemb = self.concept_emb[c]
            # repremb = self.response_emb(r)
            # predcurc
            if emb_type.find("predcurc") != -1:
                # catemb = repqemb + repcemb + xemb
                catemb = repcemb# +xemb -> predr
                # if self.num_q > 0:
                #     catemb = catemb + repqemb
                # posemb = self.position_emb(pos_encode(xemb.shape[1]))
                # catemb += posemb
                if emb_type.find("trans") != -1:
                    # mask = self.get_attn_pad_mask(sm) # 这里不能看未来信息，看了response
                    mask = ut_mask(seq_len = catemb.shape[1])
                    qh = self.qnet1(catemb.transpose(0,1), mask).transpose(0,1)
                else:
                    qh, _ = self.qnet1(catemb)
                if train:
                    if emb_type.find("match") == -1:
                        cpreds = self.qclasifier1(qh)
                        flag = sm==1
                        y2 = self.closs1(cpreds[flag], c[flag])
                        if emb_type.find("predr") != -1:
                            rpreds = self.rclasifier1(qh)
                            y2 = y2+self.closs1(rpreds[flag], r[flag])

                    else:
                        merge = torch.cat([repcemb, qh], dim=-1)
                        cpreds = torch.sigmoid(self.qclasifier1(merge))
                        cpreds = cpreds[:,:,1]
                        # print(f"cpreds: {cpreds.shape}")
                        flag = sm==1
                        ones = torch.ones(cpreds.shape[0], cpreds.shape[1]).to(device)
                        y2 = self.closs1(cpreds[flag], ones[flag])
                        # assert False
                xemb1 = qh

            # multi label pred
            if emb_type.find("ml") != -1:
                posemb = self.position_emb(pos_encode(xemb.shape[1]))
                if train:
                    oriqs, orics, orisms = dcur["oriqs"].long(), dcur["orics"].long(), dcur["orisms"].long()#self.generate_oriqcs(q, c, sm)
                    concept_avg = self.get_avg_skill_emb(orics)
                    # que_c_emb = concept_avg + posemb
                    qemb = self.question_emb(oriqs)
                    que_c_emb = concept_avg+qemb+posemb#torch.cat([concept_avg, qemb],dim=-1)

                    # add mask
                    # mask = self.get_attn_pad_mask(orisms)
                    # print(f"que_c_emb: {que_c_emb.shape}, mask: {mask.shape}, orisms: {orisms.shape}")
                    # assert False
                    mask = ut_mask(seq_len = que_c_emb.shape[1])
                    qh = self.qnet2(self.base_qnet22(self.base_qnet21(que_c_emb.transpose(0,1), mask).transpose(0,1)))
                    cpreds = torch.sigmoid(self.qclasifier2(qh))
                    flag = orisms==1
                    masked = cpreds[flag]
                    pad = torch.ones(cpreds.shape[0], cpreds.shape[1], self.num_c-10).to(device)
                    pad = -1 * pad
                    ytrues = torch.cat([orics, pad], dim=-1).long()[flag]
                    y3 = self.closs2(masked, ytrues)
                # qcemb = repcemb+posemb
                qcemb = repcemb+repqemb+posemb#torch.cat([repcemb, repqemb], dim=-1)
                # qcmask = self.get_attn_pad_mask(sm)
                qcmask = ut_mask(seq_len = qcemb.shape[1])
                qcemb = self.qnet2(self.base_qnet22(self.base_qnet21(qcemb.transpose(0,1), qcmask).transpose(0,1)))
                xemb2 = qcemb
            
            if emb_type.find("predcurc") != -1:
                xemb = xemb + xemb1
            if emb_type.find("ml") != -1:
                xemb = xemb + xemb2 + repqemb
            xemb = xemb + repcemb
            # kt
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        if train:
            return y, y2, y3
        else:
            return y

    # 计算每个技能的遗忘率
    def calSkillF(self, cs, rs, sm):
        dr2w, dr = dict(), dict()
        concepts = set()
        for i in range(cs.shape[0]): # batch
            drs = dict()
            for j in range(cs.shape[1]): # seqlen
                curc, curr = cs[i][j].detach().cpu().item(), rs[i][j].detach().cpu().item()
                # print(f"curc: {curc}")
                if j != 0 and sm[i][j-1] != 1:
                    break
                
                if curr == 1:
                    dr.setdefault(curc, 0)
                    dr[curc] += 1
                elif curr == 0 and curc in drs and drs[curc][-1][0] == 1:
                    dr2w.setdefault(curc, 0)
                    dr2w[curc] += 1
                drs.setdefault(curc, list())
                drs[curc].append([curr, j])
                concepts.add(curc)
        print(f"dr2w: {dr2w}, dr: {dr}")
        sum = 0
        for c in dr:
            if c not in dr2w:
                self.dF[c] = 0
            else:
                self.dF[c] = dr2w[c] / dr[c]
                self.dfenzi[c] = dr2w[c]
                self.dfenmu[c] = dr[c]
                sum += dr2w[c] / dr[c]
        self.avgf = sum / len(dr)
        print(f"dF: {self.dF}, avgf: {self.avgf}")

    # forget
    def generate_forget(self, cs, rs):
        css, fss = [], []
        for i in range(cs.shape[0]): # batch
            curfs = []
            dlast = dict()
            for j in range(cs.shape[1]): # seqlen
                curc, curr = cs[i][j].detach().cpu().item(), rs[i][j].detach().cpu().item()
                # 动态forget, 看前两步历史t-2, t-1
                if self.emb_type.find("t-2t-1") != -1:
                    if curc not in dlast or len(dlast[curc]) < 2:
                        curf = 1-self.dF.get(curc, self.avgf)
                    elif curc not in self.dfenzi: # 不存在1-》0 但是可能存在1
                        curf = 1-self.dF.get(curc, self.avgf)
                    else:
                        # print(f"self.dF: {self.dF[curc]}")
                        fenzi = self.dfenzi[curc]
                        fenmu = self.dfenmu[curc]
                        if dlast[curc][-2][1] == 1:
                            if dlast[curc][-1][1] == 0:
                                fenzi = fenzi + 1
                            fenmu = fenmu + 1
                        curf = 1- fenzi / fenmu
                    curfs.append([curf])
                    dlast.setdefault(curc, [])
                    dlast[curc].append([j, curr])
                # 动态forget, 看前两步历史t-1, t
                elif self.emb_type.find("tt-1") != -1:
                    dlast.setdefault(curc, [])
                    dlast[curc].append([j, curr])
                    if len(dlast[curc]) < 2:
                        curf = 1-self.dF.get(curc, self.avgf)
                    elif curc not in self.dfenzi: # 不存在1-》0 但是可能存在1
                        curf = 1-self.dF.get(curc, self.avgf)
                    else:
                        # print(f"self.dF: {self.dF[curc]}")
                        fenzi = self.dfenzi[curc]
                        fenmu = self.dfenmu[curc]
                        if dlast[curc][-2][1] == 1:
                            if dlast[curc][-1][1] == 0:
                                fenzi = fenzi + 1
                            fenmu = fenmu + 1
                        curf = 1- fenzi / fenmu
                    curfs.append([curf])
                else:
                    # 静态forget
                    if curc not in dlast:
                        curf = 1
                    else:
                        # delta = j - dlast[curc]
                        curf = (1-self.dF.get(curc, self.avgf))#**delta
                    curfs.append([curf])
                    dlast.setdefault(curc, [])
                    dlast[curc].append([j, curr])
                    
                
            # print(f"curfs: {curfs}")
            fss.append(curfs)
            # assert False
        return torch.tensor(fss).float().to(device)


    
