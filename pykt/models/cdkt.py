import os

import pandas as pd
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm
from .utils import transformer_FFN, ut_mask

device = "cpu" if not torch.cuda.is_available() else "cuda"

class CDKT(Module):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "cdkt"
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
    
        if self.emb_type.endswith("addcemb"): # xemb += cemb
            self.concept_emb = Embedding(self.num_c, self.emb_size)

        if self.emb_type.endswith("addcembr"): # concat(xemb += cemb, r)
            self.concept_emb = Embedding(self.num_c, self.emb_size)
            self.lstm_layer = LSTM(self.emb_size*3, self.hidden_size, batch_first=True)
            self.out_layer = Linear(self.hidden_size, self.num_c)

        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2
            self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.qdrop = Dropout(dropout)
            self.qclasifier = Linear(self.hidden_size, self.num_c)

        if self.emb_type.endswith("prednextc"): # predict next concept
            self.kc_drop = Dropout(dropout) ## 1.3
            self.kc_layer = Linear(self.hidden_size, self.num_c)
            self.layer_norm = LayerNorm(self.emb_size)
            self.kc_layer_norm = LayerNorm(self.emb_size)
            if self.emb_type.find("addpc"): 
                self.concept_emb = Embedding(self.num_c, self.emb_size)

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

    def forward(self, c, r, q, train=False): ## F * xemb
        y2 = None

        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            x = c + self.num_c * r
            xemb = self.interaction_emb(x)
            
        if emb_type == "qid":
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
            pad = torch.zeros(xemb.shape[0], 1, xemb.shape[2]).to(device)
            chistory = torch.cat((pad, xemb[:,0:-1,:]), dim=1)
            qh, _ = self.qlstm(qemb+chistory)
            y2 = self.qclasifier(qh)

            # predict response
            xemb = xemb + qh
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



        

        # x = c + self.num_c * r
        # xemb = self.interaction_emb(x)
        # # xemb = fs.exp() * xemb
        # h, _ = self.lstm_layer(xemb)
        # h = self.dropout_layer(h)
        # fs = self.getfseqs(c)
        # y = self.out_layer(fs*h)
        # y = torch.sigmoid(y)
        # # y = fs * y
        # if train:
        #     return y, None
        # else:
        #     return y

    # def forward(self, c, r, q, train=False): ## 
    #     sLeft = self.calfseqs(c) # 计算当前forget序列

    #     x = c + self.num_c * r
    #     xemb = self.interaction_emb(x)

    #     h, _ = self.lstm_layer(xemb)
    #     # # print(f"sLeft: {sLeft.shape}, h: {h.shape}")
    #     # mask = ut_mask(seq_len = c.shape[1])
    #     # sLeft = sLeft.squeeze(-1).unsqueeze(1).expand(sLeft.shape[0],sLeft.shape[1],sLeft.shape[1])
    #     # sLeft = sLeft.masked_fill(mask, -1e32)
    #     # scores = torch.softmax(sLeft, dim=-1)
    #     # # print(f"scores: {scores}, h: {h.shape}")
    #     # hsum = torch.bmm(scores, h)
    #     # hcum = torch.cumsum(hsum, dim=1)-hsum

    #     # hcum = hcum+h

    #     hcum = sLeft * h

    #     h = self.dropout_layer(hcum)
    #     print(f"h: {h.shape}")
    #     y = self.out_layer(h)
    #     y = torch.sigmoid(y)     
    #     if train:   
    #         return y, None
    #     else:
    #         return y

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

    # def forward(self, c, r, q, train=False): ## 1.2
    #     # xemb
    #     x = c + self.num_c * r
    #     xemb = self.interaction_emb(x)

    #     # predict concept
    #     qemb = self.question_emb(q)
    #     pad = torch.zeros(xemb.shape[0], 1, xemb.shape[2]).to(device)
    #     chistory = torch.cat((pad, xemb[:,0:-1,:]), dim=1)
    #     qh, _ = self.qlstm(qemb+chistory)
    #     # print(f"qh: {qh.shape}")
    #     predcs = self.qclasifier(qh)

    #     # predict response
    #     xemb = xemb + qh
    #     h, _ = self.lstm_layer(xemb)
    #     h = self.dropout_layer(h)
    #     y = self.out_layer(h)
    #     y = torch.sigmoid(y)
    #     if train:
    #         return y, predcs
    #     return y
        

    # def forward(self, c, r, q, train=False): ## 1.3
    #     x = c + self.num_c * r
    #     xemb = self.interaction_emb(x)
    #     h, _ = self.lstm_layer(xemb)

    #     h = self.dropout_layer(h)
    #     y = self.out_layer(self.layer_norm(xemb+h))
    #     y = torch.sigmoid(y)

    #     if train:
    #         y2 = None
    #         h2 = self.kc_drop(h)
    #         y2 = self.kc_layer(self.kc_layer_norm(xemb+h2))
    #         y2 = torch.sigmoid(y2)

    #         # predl = torch.argmax(y2, dim=-1)
    #         # h3 = self.qemb(predl)
    #         # h3, _ = self.lstm_layer(h3)

    #         # h = self.dropout_layer(h)
    #         # h = xemb + h + h3
    #         # h = self.layer_norm(h)
    #         # y = self.out_layer(h)
    #         # y = torch.sigmoid(y)

    #         return y, y2
    #     else:
    #         return y