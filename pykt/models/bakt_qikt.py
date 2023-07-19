import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy
import random
# from entmax import sparsemax, entmax15, entmax_bisect, EntmaxBisect
from .loss import Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class MLP(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))

class BAKT_QIKT(nn.Module):
    def __init__(self, n_question, n_pid, 
            d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4, seq_len=200, 
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768, sparse_ratio=0.8, k_index=5, stride=1,mlp_layer_num=2,loss_c_all_lambda=0.3,loss_c_next_lambda=0.3):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "bakt_qikt"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.sparse_ratio = sparse_ratio
        self.k_index = k_index
        self.stride = stride
        self.mlp_layer_num = mlp_layer_num
        self.loss_c_all_lambda = loss_c_all_lambda
        self.loss_c_next_lambda = loss_c_next_lambda

        embed_l = d_model
        if self.n_pid > 0:
            if emb_type.find("scalar") != -1:
                # print(f"question_difficulty is scalar")
                self.difficult_param = nn.Embedding(self.n_pid+1, 1) # 题目难度
            else:
                self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上
        
        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                    self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )
        
        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.l1 = loss1
            self.l2 = loss2
            self.l3 = loss3
            num_layers = num_layers
            self.emb_size, self.hidden_size = d_model, d_model
            self.num_q, self.num_c = n_pid, n_question
            
            if self.num_q > 0:
                self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2
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
                    # nn.Linear(self.hidden_size*2, self.hidden_size), nn.ELU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_size, self.hidden_size//2), nn.ELU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_size//2, 1))
                self.hisloss = nn.MSELoss()
            
        if self.emb_type.find("qikt") != -1:
            self.out_concept_next = MLP(self.mlp_layer_num, d_model + embed_l, self.n_question, self.dropout)
            self.out_concept_all = MLP(self.mlp_layer_num, d_model, self.n_question, self.dropout)

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

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
        if self.separate_qa:
            catemb += cemb
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
        if self.separate_qa:
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
        padsm = torch.ones(sm.shape[0], 1).to(device)
        sm = torch.cat([padsm, sm], dim=-1)

        # predict history correctness rates
        
        start = self.start
        rpreds = torch.sigmoid(self.hisclasifier(h)[:,start:,:]).squeeze(-1)
        rsm = sm[:,start:]
        rflag = rsm==1
        # rtrues = torch.cat([dcur["historycorrs"][:,0:1], dcur["shft_historycorrs"]], dim=-1)[:,start:]
        padr = torch.zeros(h.shape[0], 1).to(device)
        rtrues = torch.cat([padr, dcur["historycorrs"]], dim=-1)[:,start:]
        # rtrues = dcur["historycorrs"][:,start:]
        # rtrues = dcur["totalcorrs"][:,start:]
        # print(f"rpreds: {rpreds.shape}, rtrues: {rtrues.shape}")
        y3 = self.hisloss(rpreds[rflag], rtrues[rflag])

        # h = self.dropout_layer(h)
        # y = torch.sigmoid(self.out_layer(h))
        return y3

    def forward(self, dcur, qtest=False, train=False, attn_grads=None,save_path="", save_attn_path="", save_grad_path=""):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)

        emb_type = self.emb_type
        sparse_ratio = self.sparse_ratio
        k_index = self.k_index
        stride = self.stride

        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.n_pid > 0 and emb_type.find("norasch") == -1: # have problem id
            if emb_type.find("aktrasch") == -1:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            else:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

                qa_embed_diff_data = self.qa_embed_diff(
                    target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
                qa_embed_data = qa_embed_data + pid_embed_data * \
                        (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        y2, y3 = 0, 0
        if emb_type in ["qid", "qidaktrasch", "qid_scalar", "qid_norasch"]:
            d_output, attn_weights = self.model(q_embed_data, qa_embed_data)
            self.attn_weights = attn_weights

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
        elif emb_type.find("attn") != -1:
            d_output, attn_weights = self.model(q_embed_data, qa_embed_data, emb_type, sparse_ratio, k_index, attn_grads, stride,save_path, save_attn_path, save_grad_path)
            if emb_type.find("qikt") != -1:
                    h_next = torch.cat([d_output, q_embed_data], dim=-1)
                    # print(f"h_next:{h_next.shape}")
                    y_concept_next = torch.sigmoid(self.out_concept_next(h_next))
                    # print(f"d_output:{d_output.shape}")
                    y_concept_all = torch.sigmoid(self.out_concept_all(d_output))
                    y_concept_final = (y_concept_next + y_concept_all)/2
                    # print(f"y_concept_final:{y_concept_final.shape}")
                    # y_concept_final = torch.sigmoid( y_concept_next + y_concept_all)  
            else:
                self.attn_weights = attn_weights
                concat_q = torch.cat([d_output, q_embed_data], dim=-1)
                output = self.out(concat_q).squeeze(-1)
                m = nn.Sigmoid()
                preds = m(output)

        elif emb_type.endswith("predcurc"): # predict current question' current concept
            # predict concept
            qemb = self.question_emb(pid_data)

            # predcurc(self, qemb, cemb, xemb, dcur, train):
            cemb = q_embed_data
            if emb_type.find("noxemb") != -1:
                y2, q_embed_data, qa_embed_data = self.predcurc2(qemb, cemb, qa_embed_data, dcur, train)
            else:
                y2, qa_embed_data = self.predcurc(qemb, cemb, qa_embed_data, dcur, train)
            
            # q_embed_data = self.changecemb(qemb, cemb)

            # predict response
            d_output = self.model(q_embed_data, qa_embed_data)
            # if emb_type.find("after") != -1:
            #     curh = self.model(q_embed_data+qemb, qa_embed_data+qemb)
            #     y2 = self.afterpredcurc(curh, dcur)
            if emb_type.find("his") != -1:
                y3 = self.predhis(d_output, dcur)
                concat_q = torch.cat([d_output, q_embed_data], dim=-1)
                # if emb_type.find("his") != -1:
                #     y3 = self.predhis(concat_q, dcur)
                output = self.out(concat_q).squeeze(-1)
                m = nn.Sigmoid()
                preds = m(output)
                  
                
        if emb_type.find("qikt") != -1:
                preds_all = (y_concept_all[:,1:] * one_hot(cshft.long(), self.n_question)).sum(-1)
                preds_next = (y_concept_next[:,1:] * one_hot(cshft.long(), self.n_question)).sum(-1)
                preds_final = (y_concept_final[:,1:] * one_hot(cshft.long(), self.n_question)).sum(-1)
                if train:
                    # print(f"trainiing_preds_next:{preds_next.shape}")
                    loss_c_all = binary_cross_entropy(preds_all, dcur["shft_rseqs"], dcur["smasks"])
                    loss_c_next = binary_cross_entropy(preds_next, dcur["shft_rseqs"], dcur["smasks"])#kc level loss
                    loss_kt = binary_cross_entropy(preds_final, dcur["shft_rseqs"], dcur["smasks"])
                    # total_loss = loss_c_next
                    total_loss = loss_kt  + self.loss_c_all_lambda * loss_c_all + self.loss_c_next_lambda * loss_c_next
                    return total_loss
                else:
                    # print(f"evaluating")
                    return preds_final
        else:
            if train:
                return preds, y2, y3
            else:
                if qtest:
                    return preds, concat_q
                else:
                    return preds

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type.find("bakt") != -1:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data, emb_type="qid", sparse_ratio=0.8, k_index=5, attn_grads=None, stride=1,save_path="", save_attn_path="", save_grad_path=""):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        
        for block in self.blocks_2:
            x, attn_weights = block(mask=0, query=x, key=x, values=y, apply_pos=True, emb_type=emb_type, sparse_ratio=sparse_ratio, k_index=k_index, attn_grads=attn_grads, stride=stride, save_path=save_path, save_attn_path=save_attn_path, save_grad_path=save_grad_path) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
        return x, attn_weights

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, emb_type="qid", sparse_ratio=0.8, k_index=5, attn_grads=None, stride=1,save_path="", save_attn_path="", save_grad_path=""):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2,_ = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, emb_type=emb_type, sparse_ratio=sparse_ratio, k_index=k_index, attn_grads=attn_grads, stride=stride, save_path=save_path, save_attn_path=save_attn_path, save_grad_path=save_grad_path) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2,_ = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, emb_type=emb_type, sparse_ratio=sparse_ratio, k_index=k_index, attn_grads=attn_grads, stride=stride, save_path=save_path, save_attn_path=save_attn_path, save_grad_path=save_grad_path)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query,_


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, emb_type="qid", sparse_ratio=0.8, k_index=5, attn_grads=None, stride=1,save_path="", save_attn_path="", save_grad_path=""):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores, attn_weights = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, emb_type, sparse_ratio=sparse_ratio, k_index=k_index, attn_grads=attn_grads, stride=stride,save_path=save_path, save_attn_path=save_attn_path, save_grad_path=save_grad_path)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output, attn_weights


def attention(q, k, v, d_k, mask, dropout, zero_pad, emb_type="qid", sparse_ratio=0.8, k_index=5, attn_grads=None, stride=1, save_path="", save_attn_path="", save_grad_path=""):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim

    scores = torch.matmul(q, k.transpose(-2, -1)) / \
            math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    if emb_type.find("stride") == -1 and emb_type.find("local") == -1:
        scores.masked_fill_(mask == 0, -1e32)
        if emb_type.find("sparsemax") != -1 :
            # print(f"using attn_type: sparsemax")
            scores = sparsemax(scores, dim=-1)
        elif emb_type.find("entmax15") != -1:
            # print(f"using attn_type:entmax15")
            scores = entmax15(scores, dim=-1)
        else:
            # print(f"using attn_type: std_softmax")
            scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"scores_before:{scores}")

    # 对于每一个ai，独立的生成一个【0，1】之间的随机数（uniformly sampled）
    if emb_type.find("uniform_attn") != -1:
        scores = torch.rand(bs,head,seqlen,seqlen).to(device)
        scores.masked_fill_(mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # 不改变ai的值，随机调换位置
    elif emb_type.find("random_attn") != -1:
        # print(f"running emb_type is {emb_type}")
        scores = torch.reshape(scores, (bs*head*seqlen,-1))
        total_idx = torch.tensor([]).to(device)
        for j in range(bs*head):
            index = torch.zeros(2*seqlen).to(device)
            for i in range(2,seqlen):
                tmp_idx = torch.tensor(random.sample(range(0,i),i) + [0] * (seqlen - i)).to(device)
                index = torch.cat([index,tmp_idx])
            total_idx = torch.cat([total_idx,torch.reshape(index, (seqlen,-1))]).long()
        new_scores = torch.gather(scores, -1, total_idx).reshape(bs,head,seqlen,-1)
        new_scores.masked_fill_(mask == 0, 0)
        scores = new_scores

    # 不改变ai的值，随机调换位置(加速版)
    elif emb_type.find("random_fast_attn") != -1:
        # print(f"running emb_type is {emb_type}")
        scores = torch.reshape(scores, (bs*head*seqlen,-1))
        total_idx = torch.tensor([]).to(device)
        # print(f"before sorted:{scores}")
        for j in range(head):
            index = torch.zeros(2*seqlen).to(device)
            for i in range(2,seqlen):
                tmp_sample = torch.randperm(i)
                tmp_sample = tmp_sample.unsqueeze(0).unsqueeze(0)
                tmp_idx = torch.nn.functional.pad(tmp_sample, pad = [0, (seqlen-i), 0, 0]).squeeze(0).squeeze(0).to(device)
                index = torch.cat([index,tmp_idx])
            total_idx = torch.cat([total_idx,torch.reshape(index, (seqlen,-1))]).long()
        total_idx = total_idx.reshape(head,seqlen,-1).repeat(bs,1,1).reshape(bs*head*seqlen,-1)
        new_scores = torch.gather(scores, -1, total_idx).reshape(bs,head,seqlen,-1)
        new_scores.masked_fill_(mask == 0, 0)
        scores = new_scores
        # print(f"after sorted:{scores}")

    elif emb_type.find("permute_attn") != -1:
        # print(f"running emb_type is {emb_type}")
        scores = torch.reshape(scores, (bs*head*seqlen,-1))
        # print(f"before sorted:{scores}")
        total_idx = torch.tensor([]).to(device)
        for j in range(bs*head):
            index = torch.zeros(3*seqlen).to(device)
            for i in range(3,seqlen):
                tmp_idx = torch.tensor([0] + random.sample(range(1,i),i-1) + [0] * (seqlen - i)).to(device)
                index = torch.cat([index,tmp_idx])
            total_idx = torch.cat([total_idx,torch.reshape(index, (seqlen,-1))]).long()
        new_scores = torch.gather(scores, -1, total_idx).reshape(bs,head,seqlen,-1)
        new_scores.masked_fill_(mask == 0, 0)
        scores = new_scores
        # print(f"after sorted:{scores}")

    elif emb_type.find("permute_fast_attn") != -1:
        # print(f"running emb_type is {emb_type}")
        scores = torch.reshape(scores, (bs*head*seqlen,-1))
        total_idx = torch.tensor([]).to(device)
        for j in range(head):
            index = torch.zeros(2*seqlen).to(device)
            for i in range(2,seqlen):
                tmp_sample = np.arange(1,i-1)
                random.shuffle(tmp_sample)
                tmp_sample = torch.tensor(tmp_sample).unsqueeze(0).unsqueeze(0)
                tmp_sample = torch.nn.functional.pad(tmp_sample, pad = [0, 1, 0, 0], value=i-1)
                tmp_idx = torch.nn.functional.pad(tmp_sample, pad = [1, (seqlen-i), 0, 0]).squeeze(0).squeeze(0).to(device)
                index = torch.cat([index,tmp_idx])
            total_idx = torch.cat([total_idx,torch.reshape(index, (seqlen,-1))]).long()
        total_idx = total_idx.reshape(head,seqlen,-1).repeat(bs,1,1).reshape(bs*head*seqlen,-1)
        new_scores = torch.gather(scores, -1, total_idx).reshape(bs,head,seqlen,-1)
        new_scores.masked_fill_(mask == 0, 0)
        scores = new_scores

    elif emb_type.find("sort_attn") != -1:
        # print(f"running emb_type is {emb_type}")
        # print(f"before sorted:{scores}")
        scores = torch.reshape(scores, (bs*head*seqlen,-1))
        sorted_scores,sorted_idx = torch.sort(scores,descending=True)
        new_scores = torch.gather(scores, -1, sorted_idx).reshape(bs,head,seqlen,-1)
        # new_scores.masked_fill_(mask == 0, -1e32)
        scores = new_scores
        # print(f"after sorted:{scores}")

    # sparse attention
    elif emb_type.find("sparse_attn") != -1:
        # scorted_attention
        sorted_scores,sorted_idx = torch.sort(scores,descending=True)
        sorted_scores = sorted_scores.reshape(bs*head*seqlen, seqlen)
        scores = torch.zeros(bs*head*seqlen, seqlen).to(device)
        for i,attn in enumerate(sorted_scores):
            for j in range(seqlen):
                if torch.sum(attn[:j]) >= sparse_ratio:
                    scores[i][:j] = attn[:j]
                    break
        scores = torch.where(scores == torch.tensor(0).to(device),torch.tensor(-1e32).to(device),scores).reshape(bs,head,seqlen,-1)
        scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen

    elif emb_type.find("sparseattn") != -1:
        # scorted_attention
        scores_a = scores[:, :, :k_index, :]
        scores_b = scores[:, :, k_index:, :].reshape(bs*head*(seqlen-k_index), -1)
        sorted_scores,sorted_idx = torch.sort(scores_b,descending=True)
        scores_t = sorted_scores[:,k_index-1:k_index].repeat(1,seqlen)
        scores_b = torch.where(scores_b - scores_t >= torch.tensor(0).to(device), scores_b, torch.tensor(-1e32).to(device)).reshape(bs,head,seqlen-k_index,-1)
        scores = torch.cat([scores_a, scores_b], dim=2)
        # if emb_type == "sparseattn":
        #     # print(f"top_k:softmatx")
        scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
        # else:
        #     # print(f"top_k:entmax15")
        #     scores = entmax15(scores, dim=-1)  # BS,8,seqlen,seqlen

    elif emb_type.find("grad") != -1:
        grads = np.load(save_grad_path,allow_pickle=True)["arr_0"]
        new_scores = torch.from_numpy(grads).to(device)
        new_scores.masked_fill_(mask == 0, -1e32)
        new_scores = new_scores[:bs, :, :, :]
        if emb_type.find("multiples") != -1:
            # print(f"emb_type.find != -1")
            scores = np.load(save_attn_path,allow_pickle=True)["arr_0"]
            scores = torch.from_numpy(scores).to(device)
            new_scores = new_scores * scores[:bs, :, :, :]
        scores = F.softmax(new_scores, dim=-1)  # BS,8,seqlen,seqlen

    elif emb_type.find("old_grad") != -1:
        if attn_grads is not None:
            new_scores = attn_grads.detach().cpu().numpy()
            new_scores = torch.tensor(new_scores, requires_grad=True).to(device)
            new_scores.masked_fill_(mask == 0, -1e32)
            if emb_type.find("multiples") != -1:
                # print(f"emb_type.find != -1")
                new_scores = new_scores[:bs, :, :, :] * scores
            scores = F.softmax(new_scores, dim=-1)  # BS,8,seqlen,seqlen

    elif emb_type.find("stride") != -1:
        mask_x = torch.arange(seqlen).unsqueeze(-1).to(device)
        mask_y = mask_x.permute(1,0).to(device)
        mask_z = torch.zeros(seqlen,seqlen).to(device)
        mask_q = mask_z + mask_x
        mask_k = mask_z + mask_y
        mask_c1 = mask_q > mask_k
        mask_c2 = torch.eq(torch.fmod(mask_q - mask_k, stride), 0)
        mask_c3 = torch.logical_and(mask_c1, mask_c2)
        scores.masked_fill_(mask_c3 == 0, -1e32)
        # print(f"before_scores: {scores}")
        scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen

    elif emb_type.find("local") != -1:
        mask_a = torch.tril(torch.ones(seqlen, seqlen),-1).to(device)
        mask_b = torch.triu(torch.ones(seqlen, seqlen), -k_index).to(device)
        new_mask = mask_a * mask_b
        scores.masked_fill_(new_mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)

    elif emb_type.find("accumulative") != -1:
        # print(f"running local accumulative-attn")
        scores = torch.reshape(scores, (bs*head*seqlen,-1))
        sorted_scores,sorted_idx = torch.sort(scores,descending=True)
        acc_scores = torch.cumsum(sorted_scores,dim=1)
        acc_scores_a = torch.where(acc_scores<=0.999,acc_scores,torch.tensor(0).to(device).float())
        acc_scores_b = torch.where(acc_scores>=sparse_ratio,1,0)
        idx = torch.argmax(acc_scores_b,dim=1, keepdim=True)
        new_mask = torch.zeros(bs*head*seqlen,seqlen).to(device)
        a = torch.ones(bs*head*seqlen,seqlen).to(device)
        new_mask.scatter_(1,idx,a) 
        idx_matrix = torch.arange(seqlen).repeat(bs*seqlen*head,1).to(device)
        new_mask = torch.where(idx_matrix - idx <=0,0,1).float()
        sorted_scores = new_mask * sorted_scores
        sorted_scores = torch.where(sorted_scores==0.0,torch.tensor(-1).to(device).float(),sorted_scores)
        tmp_scores, indices= torch.max(sorted_scores,dim=1)
        tmp_scores = tmp_scores.unsqueeze(-1).repeat(1,seqlen)
        new_scores = torch.where(tmp_scores-scores>=0,torch.tensor(-1e32).to(device).float(),scores).reshape((bs,head,seqlen,-1))
        scores = F.softmax(new_scores, dim=-1)
        # if emb_type == "accumulative_attn":
        #     # print(f"accumulative:softmatx")
        # scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
        # else:
        #     # print(f"accumulative:entmax15")
        #     scores = entmax15(scores, dim=-1)  # BS,8,seqlen,seqlen
    else:
        before_dropout_scores = scores

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:bs, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")

    if emb_type == "qid":
        sub_scores = torch.reshape(scores,(bs*head*seqlen,-1))
        sub_scores,sorted_idx = torch.sort(sub_scores,descending=True)
        sub_scores = sub_scores[:,:5]
        sub_scores = torch.cumsum(sub_scores,dim=1)
        sub_scores = sub_scores[:,-1].tolist()
        # with open("./2005_sub_scores_final.txt","a") as f:
        #     f.write(str(sub_scores) + "\n")
    # print(f"dropout_before:{scores}")

    # tmp_scores = scores.clone()
    # tmp_scores.masked_fill_(mask == 0, -1e32)
    # tmp_scores = tmp_scores[:,:,2:,:]
    # tmp_scores = tmp_scores.reshape(bs*head*(seqlen-2),-1)
    # cnt_zeros = torch.count_nonzero(tmp_scores==0,1)
    # print(f"cnt_zeros:{cnt_zeros.shape}")
    # with open("./2005_cnt_zerosl.txt","a") as f:
    #         f.write(str(cnt_zeros.tolist()) + "\n")

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    if emb_type != "qid":
        # print(f"output:{output}")
        return output, scores
    else:
        return output, before_dropout_scores

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


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
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)

class timeGap(nn.Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_size) -> None:
        super().__init__()
        self.rgap_eye = torch.eye(num_rgap)
        self.sgap_eye = torch.eye(num_sgap)
        self.pcount_eye = torch.eye(num_pcount)

        input_size = num_rgap + num_sgap + num_pcount

        self.time_emb = nn.Linear(input_size, emb_size, bias=False)

    def forward(self, rgap, sgap, pcount):
        rgap = self.rgap_eye[rgap].to(device)
        sgap = self.sgap_eye[sgap].to(device)
        pcount = self.pcount_eye[pcount].to(device)

        tg = torch.cat((rgap, sgap, pcount), -1)
        tg_emb = self.time_emb(tg)

        return tg_emb

