import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones, lt_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BAKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4,
            kq_same=1, final_fc_dim=512, num_attn_heads=8, seq_len=200, emb_type="qid", emb_path="", pretrain_dim=768, 
            use_pos=True, qmatrix=None):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "bakt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.model_type = self.model_name
        self.emb_type = emb_type
        self.use_pos = use_pos
        embed_l = d_model
        
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            if emb_type.find("aktrasch") != -1:
                self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上

        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question, embed_l)
        self.qa_embed = nn.Embedding(2, embed_l)
        if emb_type.find("aktrasch") == -1 or emb_type.find("predcurc") != -1:
            self.que_embed = nn.Embedding(self.n_pid + 1, embed_l)

        # self.position_emb = Embedding(seq_len, d_model)
        # cosine posemb更好！
        self.position_emb = CosinePositionalEmbedding(d_model=d_model, max_len=seq_len)  
        self.n_blocks = n_blocks
        self.model = get_clones(Blocks(d_model, num_attn_heads, dropout), n_blocks)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                    embed_l), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(embed_l,
                    final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 1)
        )

        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.l1 = loss1
            self.l2 = loss2
            self.l3 = loss3
            num_layers = num_layers
            self.emb_size, self.hidden_size = d_model, d_model
            self.num_q, self.num_c = n_pid, n_question
            
            # if self.num_q > 0:
            #     self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2
            if self.emb_type.find("trans") != -1:
                self.nhead = nheads
                # d_model = self.hidden_size# * 2
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
                self.hisclasifier = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size//2), nn.ELU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_size//2, 1))
                self.hisloss = nn.MSELoss()

        self.reset()

        # self.qmatrix_t = nn.Embedding.from_pretrained(qmatrix.permute(1,0), freeze=True)

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

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
        if self.num_q > 0:
            catemb = qemb + chistory
        else:
            catemb = chistory
        # if self.emb_type.find("cemb") != -1: akt本身就加了cemb
        #     catemb += cemb

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

    def changecemb(self, qemb, cemb):
        catemb = cemb
        if self.emb_type.find("qemb") != -1:
            catemb += qemb
        if self.emb_type.find("trans") != -1:
            mask = ut_mask(seq_len = catemb.shape[1])
            qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)
        else:
            qh, _ = self.qlstm(catemb)
        
        cemb = cemb + qh
        if self.emb_type.find("qemb") != -1:
            cemb = cemb+qemb
        
        return cemb

    def afterpredcurc(self, h, dcur):
        y2 = 0
        sm, c, cshft = dcur["smasks"], dcur["cseqs"], dcur["shft_cseqs"]
        padsm = torch.ones(sm.shape[0], 1).to(device)
        sm = torch.cat([padsm, sm], dim=-1)
        c = torch.cat([c[:,0:1], cshft], dim=-1)
        
        start = 1
        cpreds = self.qclasifier(h[:,start:,:])
        flag = sm[:,start:]==1
        y2 = self.closs(cpreds[flag], c[:,start:][flag])
        
        return y2

    def predhis(self, h, dcur):
        sm = dcur["smasks"]
        start = self.start
        rpreds = torch.sigmoid(self.hisclasifier(h)[:,start:,:]).squeeze(-1)
        rsm = sm[:,start:]
        rflag = rsm==1
        rtrues = dcur["historycorrs"][:,start:]
        y3 = self.hisloss(rpreds[rflag], rtrues[rflag])
        return y3

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        sm = dcur["smasks"]
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)

        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        emb_type = self.emb_type

        # Batch First
        q_embed_data = self.q_embed(q_data)
        pid_embed_data = None

        if self.n_pid > 0: # have problem id
            if emb_type.find("aktrasch") == -1:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                final_q_embed_data = q_embed_data + pid_embed_data + \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

                final_qa_embed_data = self.qa_embed(target)+q_embed_data
            else:
                qa_embed_data = self.qa_embed(target)+q_embed_data
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

                qa_embed_diff_data = self.qa_embed_diff(
                    target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
                qa_embed_data = qa_embed_data + pid_embed_data * \
                        (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
                # c_reg_loss = (pid_embed_data ** 2.).sum() * self.rashl2 # rasch部分loss
                final_q_embed_data, final_qa_embed_data = q_embed_data, qa_embed_data

            # posemb = self.position_emb(pos_encode(final_q_embed_data.shape[1]))
            qposemb = self.position_emb(final_q_embed_data)
            final_q_embed_data = final_q_embed_data + qposemb
            qaposemb = self.position_emb(final_qa_embed_data)
            final_qa_embed_data = final_qa_embed_data + qaposemb

        # print(f"final_qa_embed_data: {final_qa_embed_data}")

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        y2, y3 = 0, 0
        if emb_type in ["qid", "qidaktrasch"]:
            q, k, v = final_q_embed_data[:,1:,:], final_q_embed_data[:,0:-1,:], final_qa_embed_data[:,0:-1,:]
            for i in range(self.n_blocks):
                q = self.model[i](q, k, v)#### saint中也是作为q使用的!

            concat_q = torch.cat([q, final_q_embed_data[:,1:,:]], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
        elif emb_type.endswith("predcurc"): # predict current question' current concept
            q_embed_data, qa_embed_data = final_q_embed_data, final_qa_embed_data
            # predict concept
            qemb = self.que_embed(pid_data)

            cemb = q_embed_data
            if emb_type.find("noxemb") != -1:
                y2, q_embed_data, qa_embed_data = self.predcurc2(qemb, cemb, qa_embed_data, dcur, train)
            else:
                y2, qa_embed_data = self.predcurc(qemb, cemb, qa_embed_data, dcur, train)
            
            # q_embed_data = self.changecemb(qemb, cemb)

            # predict response
            q, k, v = q_embed_data[:,1:,:], q_embed_data[:,0:-1,:], qa_embed_data[:,0:-1,:]
            for i in range(self.n_blocks):
                q = self.model[i](q, k, v)
            # d_output = self.model(q_embed_data, qa_embed_data)
            # if emb_type.find("after") != -1:
            #     curh = self.model(q_embed_data+qemb, qa_embed_data+qemb)
            #     y2 = self.afterpredcurc(curh, dcur)
            if emb_type.find("his") != -1:
                y3 = self.predhis(q, dcur)

            concat_q = torch.cat([q, q_embed_data[:,1:,:]], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
        # padr = torch.zeros(preds.shape[0], 1).to(device)
        # preds = torch.cat([padr, preds], dim=1)
        if train:
            return preds, y2, y3
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

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
        # padv = torch.zeros(v.shape[0], 1, v.shape[2])
        # v = torch.cat([padv, v[:,0:-1,:]], dim=1)
        # print(f"before q: {q.shape}, k: {k.shape}, v: {v.shape}")
        causal_mask = ut_mask(seq_len = k.shape[0])
        # print(f"after q: {q.shape}, k: {k.shape}, v: {v.shape}")
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb

class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(1), :]  # ( 1,seq,  Feature)
