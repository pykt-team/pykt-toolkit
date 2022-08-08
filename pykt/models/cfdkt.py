import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from .utils import transformer_FFN, ut_mask, pos_encode
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

device = "cpu" if not torch.cuda.is_available() else "cuda"

class CFDKT(Module):
    def __init__(self, num_q, num_c, num_rgap, num_sgap, num_pcount, emb_size, dropout=0.1, 
            num_layers=1, num_attn_heads=5, l1=0.5, l2=0.5, l3=0.5, start=50, seq_len=200,
            emb_type='qid', emb_path=""):
        super().__init__()
        self.model_name = "cfdkt"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        print(f"emb_type: {self.emb_type}")

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.c_integration = CIntegration(num_rgap, num_sgap, num_pcount, emb_size)
        ntotal = num_rgap + num_sgap + num_pcount
    
        self.lstm_layer = LSTM(self.emb_size + ntotal, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size + ntotal, self.num_c)

        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.l1 = l1
            self.l2 = l2
            self.l3 = l3
            if self.num_q > 0:
                self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2
            if self.emb_type.find("trans") != -1:
                self.nhead = num_attn_heads
                d_model = self.hidden_size# * 2
                encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
                encoder_norm = LayerNorm(d_model)
                self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
            else:    
                self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.qdrop = Dropout(dropout)
            self.qclasifier = Linear(self.hidden_size, self.num_c)
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size) # add concept emb
            self.closs = CrossEntropyLoss()
            # 加一个预测历史准确率的任务
            if self.emb_type.find("his") != -1:
                self.start = start
                self.hisclasifier = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size//2), nn.ELU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_size//2, 1))
                self.hisloss = nn.MSELoss()

    def forward(self, dcur, dgaps, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        
        y2, y3 = 0, 0
        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            x = c + self.num_c * r
            xemb = self.interaction_emb(x)
            
        if emb_type == "qid":
            theta_in = self.c_integration(xemb, dgaps["rgaps"].long(), dgaps["sgaps"].long(), dgaps["pcounts"].long())
            h, _ = self.lstm_layer(theta_in)
        elif emb_type.endswith("predcurc"):
            if emb_type.find("notadd") != -1:
                y2, _ = self.predcurc(dcur, q, c, r, xemb, train)
            else:
                y2, xemb = self.predcurc(dcur, q, c, r, xemb, train)
            theta_in = self.c_integration(xemb, dgaps["rgaps"].long(), dgaps["sgaps"].long(), dgaps["pcounts"].long())

            h, _ = self.lstm_layer(theta_in)
            if emb_type.find("his") != -1:
                y3 = self.predhis(h, dcur)
        theta_out = self.c_integration(h, dgaps["shft_rgaps"].long(), dgaps["shft_sgaps"].long(), dgaps["shft_pcounts"].long())
        theta_out = self.dropout_layer(theta_out)
        y = self.out_layer(theta_out)
        y = torch.sigmoid(y)
        if train:
            return y, y2, y3
        else:
            return y

    def predcurc(self, dcur, q, c, r, xemb, train):
        sm = dcur["smasks"].long()
        emb_type = self.emb_type
        y2, y3 = 0, 0
        catemb = xemb
        if self.num_q > 0:
            qemb = self.question_emb(q)
            catemb = qemb + xemb
            
        if emb_type.find("cemb") != -1:
            cemb = self.concept_emb(c)
            catemb += cemb

        # cemb = self.concept_emb(c)
        # catemb = cemb
        if emb_type.find("trans") != -1:
            mask = ut_mask(seq_len = catemb.shape[1])
            qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)
        else:
            qh, _ = self.qlstm(catemb)
        if train:
            start = 0
            cpreds = self.qclasifier(qh[:,start:,:])
            flag = sm[:,start:]==1
            y2 = self.closs(cpreds[flag], c[:,start:][flag])


        # predict response
        xemb = xemb + qh + cemb
        if emb_type.find("qemb") != -1:
            xemb = xemb+qemb
        return y2, xemb
        
    def predhis(self, h, dcur):
        sm = dcur["smasks"].long()
        # predict history correctness rates
        start = self.start
        rpreds = torch.sigmoid(self.hisclasifier(h)[:,start:,:]).squeeze(-1)
        rsm = sm[:,start:]
        rflag = rsm==1
        rtrues = dcur["historycorrs"][:,start:]
        y3 = self.hisloss(rpreds[rflag], rtrues[rflag])
        return y3

class CIntegration(Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_dim) -> None:
        super().__init__()
        self.rgap_eye = torch.eye(num_rgap)
        self.sgap_eye = torch.eye(num_sgap)
        self.pcount_eye = torch.eye(num_pcount)

        ntotal = num_rgap + num_sgap + num_pcount
        self.cemb = Linear(ntotal, emb_dim, bias=False)
        # print(f"total: {ntotal}, self.cemb.weight: {self.cemb.weight.shape}")

    def forward(self, vt, rgap, sgap, pcount):
        rgap, sgap, pcount = self.rgap_eye[rgap].to(device), self.sgap_eye[sgap].to(device), self.pcount_eye[pcount].to(device)
        # print(f"vt: {vt.shape}, rgap: {rgap.shape}, sgap: {sgap.shape}, pcount: {pcount.shape}")
        ct = torch.cat((rgap, sgap, pcount), -1) # bz * seq_len * num_fea
        # print(f"ct: {ct.shape}, self.cemb.weight: {self.cemb.weight.shape}")
        # element-wise mul
        Cct = self.cemb(ct) # bz * seq_len * emb
        # print(f"ct: {ct.shape}, Cct: {Cct.shape}")
        theta = torch.mul(vt, Cct)
        theta = torch.cat((theta, ct), -1)
        return theta
