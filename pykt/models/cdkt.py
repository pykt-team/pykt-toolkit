from cmath import log
from curses.ascii import EM
import os
from tkinter import N

import pandas as pd
import torch

from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, MultiLabelMarginLoss
from .utils import transformer_FFN, ut_mask, pos_encode
from torch.nn.functional import one_hot
from .cdkt_cc import generate_postives, Network, WWWNetwork

device = "cpu" if not torch.cuda.is_available() else "cuda"

class CDKT(Module):
    def __init__(self, num_q, num_c, seq_len, emb_size, dropout=0.1, emb_type='qid', num_layers=1, l1=0.5, l2=0.5, emb_path="", pretrain_dim=768):
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

        if self.emb_type.endswith("ptrainc"): # use pretrained concept embedding
            dvec = pd.read_pickle(emb_path)
            self.concept_emb = Embedding.from_pretrained(dvec)
            for param in self.concept_emb.parameters():
                param.requires_grad = True

        if self.emb_type.endswith("pretrainqc"): # use pretrained qemb and cemb from cc
            cvec = pd.read_pickle("/hw/share/liuqiongqiong/kt/kaiyuan/dev/algebra2005_cvec.pkl")
            qvec = pd.read_pickle("/hw/share/liuqiongqiong/kt/kaiyuan/dev/algebra2005_qvecnorm.pkl")
            self.pretrain_qemb = Embedding.from_pretrained(qvec)
            self.pretrain_cemb = Embedding.from_pretrained(cvec)
            self.qlinear = Linear(qvec.shape[1], self.emb_size)
            self.clinear = Linear(cvec.shape[1], self.emb_size)
            for param in self.pretrain_qemb.parameters():
                param.requires_grad = False
            for param in self.pretrain_cemb.parameters():
                param.requires_grad = False
            if self.emb_type.find("predcurc") != -1:
                self.l1 = l1
                self.l2 = l2
                self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
                self.qdrop = Dropout(dropout)
                self.qclasifier = Linear(self.hidden_size, self.num_c)

        if self.emb_type.endswith("pretrainddiff"): # use pretrained qemb and cemb from cc
            qavgkc = pd.read_pickle("/hw/share/liuqiongqiong/kt/kaiyuan/dev/algebra2005_qavgcvec.pkl")
            qvec = pd.read_pickle("/hw/share/liuqiongqiong/kt/kaiyuan/dev/algebra2005_questionvec.pkl")
            cvec = pd.read_pickle("/hw/share/liuqiongqiong/kt/kaiyuan/dev/algebra2005_conceptvec.pkl")
            qdifficulty = pd.read_pickle("/hw/share/liuqiongqiong/kt/kaiyuan/dev/algebra2005_eachqdifficulty.pkl")
            self.pretrain_qemb = Embedding.from_pretrained(qvec)
            self.pretrain_cemb = Embedding.from_pretrained(cvec)
            self.pretrain_qavgcemb = Embedding.from_pretrained(qavgkc)
            self.pretrain_qdifficulty = Embedding.from_pretrained(qdifficulty)

            self.qlinear = Linear(qvec.shape[1], self.emb_size)
            self.clinear = Linear(cvec.shape[1], self.emb_size)
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

                
        if self.emb_type.endswith("addcemb"): # xemb += cemb
            self.concept_emb = Embedding(self.num_c, self.emb_size)

        if self.emb_type.endswith("addcembr"): # concat(xemb += cemb, r)
            self.concept_emb = Embedding(self.num_c, self.emb_size)
            self.lstm_layer = LSTM(self.emb_size*3, self.hidden_size, batch_first=True)
            self.out_layer = Linear(self.hidden_size, self.num_c)

        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.l1 = l1
            self.l2 = l2
            self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2
            self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.qdrop = Dropout(dropout)
            self.qclasifier = Linear(self.hidden_size, self.num_c)
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size) # add concept emb

            # concat response
            if self.emb_type.find("catr") != -1:
                self.lstm_layer = LSTM(self.emb_size*2, self.hidden_size, batch_first=True)
            if self.emb_type.find("addr") != -1:
                self.response_emb = Embedding(2, self.emb_size)

        if self.emb_type.endswith("prednextc"): # predict next concept
            self.kc_drop = Dropout(dropout) ## 1.3
            self.kc_layer = Linear(self.hidden_size, self.num_c)
            self.layer_norm = LayerNorm(self.emb_size)
            self.kc_layer_norm = LayerNorm(self.emb_size)
            if self.emb_type.find("addpc"): 
                self.concept_emb = Embedding(self.num_c, self.emb_size)

        if self.emb_type.endswith("addcc"): # add cc loss
            self.l1 = l1
            self.l2 = l2
            if self.emb_type.find("dktxemb") != -1:
                self.interaction_emb = Embedding((self.num_c+1) * 2+1, self.emb_size)
            elif self.emb_type.find("seperate") != -1:
                self.concept_emb = Embedding(self.num_c+2, self.emb_size)
                self.response_emb = Embedding(2, self.emb_size)
            else:
                self.interaction_emb = Embedding((self.num_c+1) * 2+1, self.emb_size)
                self.concept_emb = Embedding(self.num_c+2, self.emb_size)
                self.response_emb = Embedding(2, self.emb_size)
            if self.emb_type.find("bilstm") != -1:
                self.seqmodel = LSTM(self.emb_size, self.hidden_size, batch_first=True, bidirectional=True)
                if self.emb_type.find("www") != -1:
                    # need change!
                    self.xseqmodel = LSTM(self.emb_size, self.hidden_size, batch_first=True, bidirectional=True)
                    self.net = WWWNetwork(self.seqmodel, self.xseqmodel, "lstm", self.hidden_size*2, self.emb_size, num_c, dropout)
                else:
                    self.net = Network(self.seqmodel, "lstm", self.hidden_size*2, self.emb_size, num_c, dropout)
            elif self.emb_type.find("transformer") != -1:
                self.position_embedding = Embedding(seq_len, emb_size)
                self.nhead = 5
                encoder_layer = TransformerEncoderLayer(self.hidden_size, nhead=self.nhead)
                encoder_norm = LayerNorm(self.hidden_size)
                self.seqmodel = TransformerEncoder(encoder_layer, num_layers=1, norm=encoder_norm)
                if self.emb_type.find("www") != -1:
                    xencoder_layer = TransformerEncoderLayer(self.hidden_size, nhead=self.nhead)
                    xencoder_norm = LayerNorm(self.hidden_size)
                    self.xseqmodel = TransformerEncoder(xencoder_layer, num_layers=1, norm=xencoder_norm)
                    self.net = WWWNetwork(self.seqmodel, self.xseqmodel, "transformer", self.hidden_size, self.emb_size, num_c, dropout)
                else:
                    self.net = Network(self.seqmodel, "transformer", self.hidden_size, self.emb_size, num_c, dropout)

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
            if self.emb_type.find("addbase") != -1:
                self.basenet = TransformerEncoder(encoder_layer, num_layers=1, norm=encoder_norm)
            self.net = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
            self.position_emb = Embedding(seq_len, emb_size)
            self.qnet = nn.Sequential(
                nn.Linear(d_model,
                        self.hidden_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.emb_size), nn.ReLU(
                ), nn.Dropout(dropout)
            )
            self.qclasifier = nn.Linear(self.emb_size, self.num_c)
            self.closs = MultiLabelMarginLoss()
            # if self.emb_type.find("transpredr") != -1:
            #     self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)

        if self.emb_type.endswith("predfuture") != -1:
            self.l1, self.l2 = l1, l2
            # self.futurelstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            # self.predfuture = nn.Sequential(
            #     nn.Linear(self.hidden_size, self.emb_size), nn.ReLU(
            #     ), nn.Dropout(dropout)
            # )
            self.futureclasifier = nn.Linear(self.hidden_size, self.num_c)
            self.floss = nn.MSELoss()

            # self.out_layer = nn.Sequential(
            #     nn.Linear(self.hidden_size, self.emb_size), nn.ReLU(
            #     ), nn.Dropout(dropout),
            #     nn.Linear(self.hidden_size, self.num_c)
            # )
            

        self.dF = dict()
        self.avgf = 0

    def getfseqs(self, cs, flag=False): ## F * xemb
        fss = []
        for i in range(cs.shape[0]): # batch
            curfs = []
            for j in range(cs.shape[1]): # seqlen
                curc = cs[i][j].detach().cpu().item()
                
                curf = self.dF.get(curc, self.avgf)
                if flag: # forget
                    curfs.append([curf])
                else: # left
                    curfs.append([1-curf])
            fss.append(curfs)
            # assert False
        return torch.tensor(fss).float().to(device)

    def generate_oriqcs(self, q, c, sm, is_repeat):
        dq2c = dict()
        for i in range(0, q.shape[0]):
            for j in range(0, q[i].shape[0]):
                cq, cc = q[i][j].item(), c[i][j].item()
                dq2c.setdefault(cq, set())
                dq2c[cq].add(cc)
        qs, cs, sms = [], [], []
        for i in range(0, q.shape[0]):
            
            ### TODO, 去除之前pad的-1!!
            # actualq = q[i][sm[i]==1]
            # flag = is_repeat[i][sm[i]==1]==0
            # newq = actualq[flag].tolist()
            # sms.append([1]*len(newq)+[0]*(q.shape[1]-len(newq)))
            # newq = newq + [0] * (q.shape[1]-len(newq))
            newq = q[i].tolist()
            # maxc: 10
            curcs = []
            for cq in newq:
                ccs = list(dq2c[cq]) + [-1] * (self.num_c-len(dq2c[cq]))
                curcs.append(ccs)

            qs.append(newq)
            cs.append(curcs)
        qs = torch.LongTensor(qs).to(device)
        cs = torch.LongTensor(cs).to(device)
        sms = sm#torch.BoolTensor(sms).to(device)
        # print(f"qs: {qs.shape}, cs: {cs.shape}, sm: {sm.shape}")
        # assert False
        return qs, cs, sms

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

    def forward(self, dcur, train=False): ## F * xemb
        # print(f"keys: {dcur.keys()}")
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        sm = dcur["smasks"].long()
        is_repeat = dcur["is_repeat"]
        y2 = None

        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            x = c + self.num_c * r
            xemb = self.interaction_emb(x)
            
        if emb_type == "qid":
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("predfuture"): # add pred future correct rates
            h, _ = self.lstm_layer(xemb)
            flogits = self.futureclasifier(h)
            if train:
                predrates = torch.sigmoid(flogits)
                # cal y2 loss
                predrates = (predrates * one_hot(c.long(), self.num_c)).sum(-1) # 当前知识点以后的准确率
                rates, ratessms = dcur["futurerates"].float(), dcur["futuresms"].long()
                # print(f"rates: {rates.shape}, ratessms: {ratsessms.shape}")
                floss = self.floss(predrates[ratessms==1], rates[ratessms==1])
                y2 = floss

            logits = self.out_layer(self.dropout_layer(h))
            y = torch.sigmoid(logits)
            if emb_type.find("merge") != -1:
                mlogits = flogits + logits
                # print(f"predrates: {predrates[0:1].tolist()}")
                # yy = (y * one_hot(c.long(), self.num_c)).sum(-1)
                # print(f"predratyy: {yy[0:1].tolist()}")
                # print(f"flogits: {flogits.shape}, logits: {logits.shape}, mlogits: {mlogits.shape}")
                y = torch.sigmoid(mlogits)
                # assert False
        elif emb_type.endswith("pretrainddiff"): # use pretrained difficulty for each question
            qemb, cemb, qavgcemb, qdiff = self.pretrain_qemb(q), self.pretrain_cemb(c), self.pretrain_qavgcemb(q), self.pretrain_qdifficulty(q)
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
                y2 = self.qclasifier(qh)

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
                
                if emb_type.find("addbase") != -1:
                    que_c_emb = self.basenet(que_c_emb.transpose(0,1), mask).transpose(0,1)
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
            if emb_type.find("addbase") != -1:
                # qcemb = repcemb+repqemb
                qcemb = repcemb+repqemb+posemb#torch.cat([repcemb, repqemb], dim=-1)
                # qcmask = self.get_attn_pad_mask(sm)
                qcmask = ut_mask(seq_len = qcemb.shape[1])
                qcemb = self.basenet(qcemb.transpose(0,1), qcmask).transpose(0,1)
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
            
        elif emb_type.endswith("addcc"): # add cc
            # 需要构造数据做cc任务
            if train:
                # padsm = torch.ones(c.shape[0], 1).to(device)
                # sm = torch.cat([padsm, sm], dim=-1)
                cs1, rs1, sm1 = generate_postives(c, r, sm, self.num_c)
                cs2, rs2, sm2 = generate_postives(c, r, sm, self.num_c)
                cs1, rs1, cs2, rs2 = cs1.to(device), rs1.to(device), cs2.to(device), rs2.to(device)
                sm1, sm2 = sm1.to(device), sm2.to(device)
                # pad a cls token
                if emb_type.find("dktxemb") != -1:
                    padc = torch.tensor([[(self.num_c+1)*2]] * cs1.shape[0]).to(device)
                else:
                    padc = torch.tensor([[self.num_c+1]] * cs1.shape[0]).to(device)
                padr = torch.tensor([[0]] * rs1.shape[0]).to(device)
                pads = torch.tensor([[1]] * sm1.shape[0]).to(device)
                # print(f"cs1: {cs1.shape}, padc: {padc.shape}")
                # print(f"rs1: {rs1.shape}, padr: {padr.shape}")
                cs1, rs1 = torch.cat([padc, cs1], dim=-1), torch.cat([padr, rs1], dim=-1)
                cs2, rs2 = torch.cat([padc, cs2], dim=-1), torch.cat([padr, rs2], dim=-1)
                sm1, sm2 = torch.cat([pads, sm1], dim=-1), torch.cat([pads, sm2], dim=-1)
                posemb = self.position_embedding(pos_encode(cs1.shape[1]))
                if emb_type.find("dktxemb") != -1:
                    xemb1 = self.interaction_emb(cs1 + (self.num_c+1) * rs1)
                    xemb2 = self.interaction_emb(cs2 + (self.num_c+1) * rs2)
                    xemb1, xemb2 = xemb1 + posemb, xemb2 + posemb
                elif emb_type.find("seperate") != -1:
                    cemb1, remb1 = self.concept_emb(cs1), self.response_emb(rs1)
                    cemb2, remb2 = self.concept_emb(cs2), self.response_emb(rs2)
                    xemb1, xemb2 = cemb1 + remb1 + posemb, cemb2 + remb2 + posemb
                else:
                    cemb1, remb1 = self.concept_emb(cs1), self.response_emb(rs1)
                    cemb2, remb2 = self.concept_emb(cs2), self.response_emb(rs2)
                    xemb1, xemb2 = self.interaction_emb(cs1 + (self.num_c+1) * rs1), self.interaction_emb(cs2 + (self.num_c+1) * rs2)
                    xemb1, xemb2 = xemb1 + posemb, xemb2 + posemb

                # change mask, 
                def get_attn_pad_mask(sm):
                    batch_size, l = sm.size()
                    pad_attn_mask = sm.data.eq(0).unsqueeze(1)
                    pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
                    return pad_attn_mask.repeat(self.nhead, 1, 1)
                sm1, sm2 = get_attn_pad_mask(sm1), get_attn_pad_mask(sm2)
                # print(f"xemb1: {xemb1.shape}, c: {c.shape}")
                if emb_type.find("www") == -1:
                    y2 = self.net(xemb1, xemb2, sm1, sm2) # ccloss
                else:
                    y2 = self.net(cemb1, cemb2, xemb1, xemb2, sm1, sm2)

            if emb_type.find("dktxemb") != -1:
                x = c + (self.num_c+1) * r
                xemb = self.interaction_emb(x)
            elif emb_type.find("seperate") != -1:
                cemb, remb = self.concept_emb(c), self.response_emb(r)
                xemb = cemb + remb
            else:
                x = c + (self.num_c+1) * r
                xemb = self.interaction_emb(x)
                cemb, remb = self.concept_emb(c), self.response_emb(r)
                xemb += cemb+remb
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("addcembr"):
            cemb = self.concept_emb(c)
            xemb = torch.cat([xemb, cemb], dim=-1)
            # xemb += cemb
            rpad = r.float().unsqueeze(2).expand(cemb.shape[0], cemb.shape[1], cemb.shape[2])
            xemb = torch.cat([xemb, rpad], dim=-1)
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("addcemb"):
            cemb = self.concept_emb(c)
            xemb += cemb
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("ptrainc"): # use pretrained concept emb from fasttext
            cemb = self.concept_emb(c)
            xemb += cemb
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("pretrainqc"): # pretrained qc from cc
            cemb = self.clinear(self.pretrain_cemb(c))
            qemb = self.qlinear(self.pretrain_qemb(q))
            if emb_type.find("predcurc") != -1:
                chistory = xemb
                catemb = qemb + chistory
                if emb_type.find("cemb") != -1:
                    catemb += cemb
                qh, _ = self.qlstm(catemb)
                y2 = self.qclasifier(qh)
                xemb = xemb + qh + cemb + qemb
            else:
                xemb = xemb + cemb + qemb

            # predict response
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))

        elif emb_type.endswith("hforget"): # (1-F)*h
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            fs = self.getfseqs(c)
            h = fs * h
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("xforget"): # F.exp() * xemb
            fs = self.getfseqs(c, True)
            xemb = fs.exp() * xemb
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("leftforget"): # sLeft * h
            h, _ = self.lstm_layer(xemb)
            sLeft = self.calfseqs(c) # 计算当前forget序列
            h = sLeft * h
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("predcurc"): # predict current question' current concept
            # predict concept
            qemb = self.question_emb(q)
            # pad = torch.zeros(xemb.shape[0], 1, xemb.shape[2]).to(device)
            # chistory = torch.cat((pad, xemb[:,0:-1,:]), dim=1)
            chistory = xemb
            catemb = qemb + chistory
            if emb_type.find("cemb") != -1:
                cemb = self.concept_emb(c)
                catemb += cemb
                if emb_type.find("caddr") != -1:
                    remb = self.response_emb(r)
                    catemb += remb
            qh, _ = self.qlstm(catemb)
            y2 = self.qclasifier(qh)

            # predict response
            xemb = xemb + qh + cemb
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
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))

        elif emb_type.endswith("prednextc"): # predict next concept
            h, _ = self.lstm_layer(xemb)
            # predict concept
            h2 = self.kc_drop(h)
            y2 = self.kc_layer(self.kc_layer_norm(xemb+h2))
            y2 = torch.sigmoid(y2)
            if emb_type.find("addpc") != -1: # use predicted next concept to predict next response
                predl = torch.argmax(y2, dim=-1) # use top1 predicted concept
                h3 = self.concept_emb(predl)
                h3, _ = self.lstm_layer(h3)

                h = self.dropout_layer(h)
                h = xemb + h + h3
                h = self.layer_norm(h)

                y = torch.sigmoid(self.out_layer(h))
            else:
                # predict response
                h = self.dropout_layer(h)
                y = torch.sigmoid(self.out_layer(h))

        if train:
            return y, y2
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
                sum += dr2w[c] / dr[c]
        self.avgf = sum / len(dr)
        print(f"dF: {self.dF}, avgf: {self.avgf}")

    def calfseqs(self, cs):
        css, fss = [], []
        for i in range(cs.shape[0]): # batch
            curfs = []
            dlast = dict()
            for j in range(cs.shape[1]): # seqlen
                curc = cs[i][j].detach().cpu().item()
                if curc not in dlast:
                    curf = 1
                else:
                    delta = j - dlast[curc]
                    curf = (1-self.dF.get(curc, self.avgf))**delta
                curfs.append([curf])
                dlast[curc] = j
            # print(f"curfs: {curfs}")
            fss.append(curfs)
            # assert False
        return torch.tensor(fss).float().to(device)


