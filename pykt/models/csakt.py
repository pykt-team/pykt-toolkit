import torch

from torch import nn
from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CSAKT(Module):
    def __init__(self, num_q, num_c, seq_len, emb_size, num_attn_heads, dropout, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4,
            num_en=2, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "csakt"
        self.emb_type = emb_type

        self.num_q = num_q
        self.num_c = num_c
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en

        if emb_type.startswith("qid"):
            # num_c, seq_len, emb_size, num_attn_heads, dropout, emb_path="")
            self.interaction_emb = Embedding(num_c * 2, emb_size)
            self.exercise_emb = Embedding(num_c, emb_size)
            # self.P = Parameter(torch.Tensor(self.seq_len, self.emb_size))
        self.position_emb = Embedding(seq_len, emb_size)

        self.blocks = get_clones(Blocks(emb_size, num_attn_heads, dropout), self.num_en)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)

        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.l1 = loss1
            self.l2 = loss2
            self.l3 = loss3
            num_layers = num_layers
            
            if self.num_q > 0:
                self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2
            if self.emb_type.find("trans") != -1:
                self.nhead = nheads
                d_model = self.emb_size# * 2
                encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
                encoder_norm = LayerNorm(d_model)
                self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
            elif self.emb_type.find("lstm") != -1:    
                self.qlstm = LSTM(self.emb_size, self.emb_size, batch_first=True)
            # self.qdrop = Dropout(dropout)
            self.qclasifier = Linear(self.emb_size, self.num_c)
            # if self.emb_type.find("cemb") != -1:
            #     self.concept_emb = Embedding(self.num_c, self.emb_size) # add concept emb
            self.closs = CrossEntropyLoss()
            # 加一个预测历史准确率的任务
            if self.emb_type.find("his") != -1:
                self.start = start
                # self.hisclasifier = nn.Sequential(
                #     nn.Linear(self.emb_size, self.emb_size//2), nn.ELU(), nn.Dropout(dropout),
                #     nn.Linear(self.emb_size//2, 1))
                self.hisclasifier = nn.Linear(self.emb_size, 1)
                self.hisloss = nn.MSELoss()

    def base_emb(self, c, r, cshft):
        x = c + self.num_c * r
        cshftemb, xemb = self.exercise_emb(cshft), self.interaction_emb(x)
    
        posemb = self.position_emb(pos_encode(xemb.shape[1]))
        xemb = xemb + posemb
        return cshftemb, xemb

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        sm = dcur["smasks"]
        cq = torch.cat((q[:,0:1], qshft), dim=1)
        cc = torch.cat((c[:,0:1], cshft), dim=1)
        cr = torch.cat((r[:,0:1], rshft), dim=1)
        
        emb_type = self.emb_type
        y2, y3 = 0, 0
        if emb_type.startswith("qid"):
            cshftemb, xemb = self.base_emb(c, r, cshft)
        # print(f"qemb: {qemb.shape}, xemb: {xemb.shape}, qshftemb: {qshftemb.shape}")
        if emb_type == "qid":
            for i in range(self.num_en):
                xemb = self.blocks[i](cshftemb, xemb, xemb)

            p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        elif emb_type.find("predcurc") != -1:
            # predict concept
            if emb_type.find("noxemb") != -1:
                cqemb = self.question_emb(cq)
                ccemb = self.exercise_emb(cc)
                _, cxemb = self.base_emb(cc, cr, cc)
                # y2, xemb = self.predcurc(cqemb, ccemb, cxemb, dcur, train)
                y2, ccemb, cxemb = self.predcurc2(cqemb, ccemb, cxemb, dcur, train)
                cshftemb, xemb = ccemb[:,1:,:], cxemb[:,0:-1,:]
            else:
                qemb = self.question_emb(q)
                cemb = self.exercise_emb(c)
                
                if emb_type.find("notadd") != -1:
                    y2, _ = self.predcurc(qemb, cemb, xemb, dcur, train)
                else:
                    y2, xemb = self.predcurc(qemb, cemb, xemb, dcur, train)
            
            # predict response
            for i in range(self.num_en):
                xemb = self.blocks[i](cshftemb, xemb, xemb)
            if emb_type.find("after") != -1:
                y2 = self.afterpredcurc(xemb, dcur)
            if emb_type.find("his") != -1:
                y3 = self.predhis(xemb, dcur)

            p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        if train:
            return p, y2, y3
        else:
            if not qtest:
                return p
            else:
                return p, xemb

    def predcurc(self, qemb, cemb, xemb, dcur, train):
        y2 = 0
        sm, c = dcur["smasks"], dcur["cseqs"]
        chistory = xemb
        if self.num_q > 0:
            catemb = qemb + chistory
        else:
            catemb = chistory

        if self.emb_type.find("cemb") != -1:
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

        xemb = xemb + qh + cemb
        if self.emb_type.find("qemb") != -1:
            xemb = xemb+qemb
        
        return y2, xemb

    def predcurc2(self, qemb, cemb, xemb, dcur, train):
        y2 = 0
        sm, c, cshft = dcur["smasks"], dcur["cseqs"], dcur["shft_cseqs"]
        padsm = torch.ones(sm.shape[0], 1).to(device)
        sm = torch.cat([padsm, sm], dim=-1)
        c = torch.cat([c[:,0:1], cshft], dim=-1)
        chistory = cemb
        if self.num_q > 0:
            catemb = qemb + chistory
        else:
            catemb = chistory

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

        # xemb = xemb+qh
        # if self.separate_qa:
        #     xemb = xemb+cemb
        cemb = cemb + qh
        xemb = xemb+qh
        if self.emb_type.find("qemb") != -1:
            cemb = cemb+qemb
            xemb = xemb+qemb
        
        return y2, cemb, xemb

    def afterpredcurc(self, h, dcur):
        y2 = 0
        sm, c, cshft = dcur["smasks"], dcur["cseqs"], dcur["shft_cseqs"]
        # padsm = torch.ones(sm.shape[0], 1).to(device)
        # sm = torch.cat([padsm, sm], dim=-1)
        # c = torch.cat([c[:,0:1], cshft], dim=-1)
        
        start = 0
        cpreds = self.qclasifier(h[:,start:,:])
        flag = sm[:,start:]==1
        if self.emb_type.find("cshft") == -1:
            y2 = self.closs(cpreds[flag], c[:,start:][flag]) # 0.9133
        else:
            y2 = self.closs(cpreds[flag], cshft[:,start:][flag])
        
        return y2

    def predhis(self, h, dcur):
        sm = dcur["smasks"]

        # predict history correctness rates
        
        start = self.start
        rpreds = torch.sigmoid(self.hisclasifier(h)[:,start:,:]).squeeze(-1)
        rsm = sm[:,start:]
        rflag = rsm==1
        # rtrues = torch.cat([dcur["historycorrs"][:,0:1], dcur["shft_historycorrs"]], dim=-1)[:,start:]
        rtrues = dcur["historycorrs"][:,start:]
        # rtrues = dcur["historycorrs"][:,start:]
        # rtrues = dcur["totalcorrs"][:,start:]
        # print(f"rpreds: {rpreds.shape}, rtrues: {rtrues.shape}")
        y3 = self.hisloss(rpreds[rflag], rtrues[rflag])

        # h = self.dropout_layer(h)
        # y = torch.sigmoid(self.out_layer(h))
        return y3

class Blocks(Module):
    def __init__(self, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()

        self.attn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm 
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(seq_len = k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb