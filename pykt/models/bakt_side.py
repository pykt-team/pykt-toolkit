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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class timeGap2(nn.Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, num_prcount, emb_size) -> None:
        super().__init__()
        self.num_rgap, self.num_sgap, self.num_pcount, self.num_prcount = num_rgap, num_sgap, num_pcount, num_prcount
        if num_rgap != 0:
            self.rgap_eye = torch.eye(num_rgap)
            self.remb = nn.Linear(num_rgap, emb_size, bias=False)
        if num_sgap != 0:
            self.sgap_eye = torch.eye(num_sgap)
            self.semb = nn.Linear(num_sgap, emb_size, bias=False)
        if num_pcount != 0:
            self.pcount_eye = torch.eye(num_pcount)
            self.pemb = nn.Linear(num_pcount, emb_size, bias=False)
        if num_prcount != 0:
            self.prcount_eye = torch.eye(num_prcount)
            self.premb = nn.Linear(num_prcount, emb_size, bias=False)

        # input_size = num_rgap + num_sgap + num_pcount
        # self.temb = nn.Linear(input_size, emb_size, bias=False)

        print(f"self.num_rgap: {self.num_rgap}, self.num_sgap: {self.num_sgap}, self.num_pcount: {self.num_pcount}, self.num_prcount: {self.num_prcount}")
        
    def forward(self, rgap, sgap, pcount, prcount):
        infs = []
        if self.num_rgap != 0:
            rgap = self.rgap_eye[rgap].to(device)
            rgap = self.remb(rgap)
            infs.append(rgap)
        if self.num_sgap != 0:
            sgap = self.sgap_eye[sgap].to(device)
            sgap = self.semb(sgap)
            infs.append(sgap)
        if self.num_pcount != 0:
            pcount = self.pcount_eye[pcount].to(device)
            pcount = self.pemb(pcount)
            infs.append(pcount)
        if self.num_prcount != 0:
            prcount = self.prcount_eye[prcount].to(device)
            prcount = self.premb(prcount)
            infs.append(prcount)
        # inf = torch.cat(infs, -1)
        # temb = self.temb(inf)
        # return temb
        return infs

class BAKTSide(nn.Module):
    def __init__(self, n_question, n_pid, num_rgap, num_sgap, num_pcount, num_prcount, 
            use_rgap, use_sgap, use_pcount, use_prcount, use_pos,
            d_model, n_blocks, dropout, d_ff=256, seq_len=200, 
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "bakt_side"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        ###
        num_rgap = 0 if use_rgap == 0 else num_rgap
        num_sgap = 0 if use_sgap == 0 else num_sgap
        num_pcount = 0 if use_pcount == 0 else num_pcount
        num_prcount = 0 if use_prcount == 0 else num_prcount
        self.use_pos = use_pos
        ###
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) # 题目难度
        
        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(emb_type=emb_type, n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )
        
        # self.c_weight = nn.Linear(d_model, d_model)
        # self.t_weight = nn.Linear(d_model, d_model)
    
        if self.emb_type.startswith("qidside"):
            self.time_emb = timeGap2(num_rgap, num_sgap, num_pcount, num_prcount, d_model)
            # self.time_emb = timeGap2(num_rgap, num_sgap, num_pcount, 0, d_model)
            # if self.emb_type.endswith("concat"):
            #     fea_num = 5
            #     middle = int(d_model*fea_num/2)
            #     self.clinear = nn.Sequential(
            #             nn.Linear(d_model*fea_num, middle), nn.ReLU(), nn.Dropout(self.dropout),
            #             nn.Linear(middle, d_model)
            #         )
            # el
            if self.emb_type.endswith("gate"):
                # fea_num = 5
                # self.gates = nn.ModuleList([nn.Linear(d_model, 1) for i in range(fea_num)])
                self.gate1 = nn.Linear(d_model, 1)
                # self.gate2 = nn.Linear(d_model, 1)

                # self.out = nn.Sequential(
                #     nn.Linear(d_model, final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                #     nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(), nn.Dropout(self.dropout),
                #     nn.Linear(final_fc_dim2, 1)
                # )

        self.position_emb = CosinePositionalEmbedding(d_model=d_model, max_len=seq_len)
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

    def getside_info(self, dgaps):
        rg, sg, p, pr = dgaps["rgaps"].long(), dgaps["sgaps"].long(), dgaps["pcounts"].long(), dgaps["prcounts"].long()
        rgshft, sgshft, pshft, prshft = dgaps["shft_rgaps"].long(), dgaps["shft_sgaps"].long(), dgaps["shft_pcounts"].long(), dgaps["shft_prcounts"].long()
        # q, qshft = dcur["qseqs"].long(), dcur["shft_qseqs"].long()

        r_gaps = torch.cat((rg[:, 0:1], rgshft), dim=1)
        s_gaps = torch.cat((sg[:, 0:1], sgshft), dim=1)
        pcounts = torch.cat((p[:, 0:1], pshft), dim=1)
        prcounts = torch.cat((pr[:, 0:1], prshft), dim=1)

        # time infomation
        tembs = self.time_emb(r_gaps, s_gaps, pcounts, prcounts)

        return tembs

    def fusion_fuc(self, fusion_infs, gate_num=1):
        res = None
        if self.emb_type.endswith("add"):
            res = fusion_infs[0]
            for inf in fusion_infs[1:]:
                res = res + inf
        # elif self.emb_type.endswith("concat"):
        #     # for inf in fusion_infs:
        #     #     print(f"inf: {inf.shape}")
        #     inf = torch.cat(fusion_infs, dim=-1)
        #     res = self.clinear(inf)
        elif self.emb_type.endswith("gate"):
            # for idx in range(len(fusion_infs)):
            #     weight = self.gates[idx](fusion_infs[idx]).unsqueeze(-2)
            #     curinf = torch.matmul(weight, fusion_infs[idx].unsqueeze(-2)).squeeze()
            #     if res != None:
            #         res = res + curinf
            #     else:
            #         res = curinf
            #     # print(f"weight: {weight.shape}, curinf: {curinf.shape}")
            #     # assert False

            newinfs = []
            for inf in fusion_infs[0:]:
                curinf = inf.unsqueeze(2)
                newinfs.append(curinf)
            res = torch.cat(newinfs, dim=2)
            # bz * seqlen * m * h
            if gate_num == 1:
                weights = torch.sigmoid(self.gate1(res)).transpose(-2, -1)
            # else:
            #     weights = torch.sigmoid(self.gate2(res)).transpose(-2, -1)
            # print(f"res: {res.shape}, weights: {weights.shape}")
            res = torch.matmul(weights, res).squeeze(-2)
            # print(f"newemb: {newemb.shape}, fusion0: {fusion_infs[0].shape}")
            # assert False

        return res

    def forward(self, dcur, dgaps, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)

        emb_type = self.emb_type

        # Batch First
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        
        ###
        fusion_infs = self.getside_info(dgaps)
        fusion_infs.append(q_embed_data)
        if self.n_pid > 0: # have problem id
            pid_embed_data = self.difficult_param(pid_data)  # 题目embedding
            fusion_infs.append(pid_embed_data)
            # qa_embed_data = qa_embed_data + pid_embed_data
        # position information
        if self.use_pos:
            pemb = self.position_emb(q_embed_data).repeat(q_embed_data.shape[0], 1, 1)
            fusion_infs.append(pemb)
        
        # fusion func
        fusion_side = self.fusion_fuc(fusion_infs, gate_num=1)
        d_output = self.model(fusion_side, qa_embed_data)

        # finalemb = self.fusion_fuc([d_output] + fusion_infs[0:-1], gate_num=2) if self.use_pos else self.fusion_fuc([d_output] + fusion_infs, gate_num=2)
        # output = self.out(finalemb).squeeze(-1)

        baseinf = self.fusion_fuc(fusion_infs[0:-1]) if self.use_pos else fusion_side
        concat_q = torch.cat([d_output, baseinf], dim=-1)
        output = self.out(concat_q).squeeze(-1)

        m = nn.Sigmoid()
        preds = m(output)

        if train:
            return preds, 0, 0
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

class Architecture(nn.Module):
    def __init__(self, emb_type, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.emb_type = emb_type
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'bakt_side'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
        
        # self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    # def forward(self, fusion_fuc, fusion_infs, q_embed_data, qa_embed_data):
    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)
        # curfusion_infs = []
        # for inf in fusion_infs:
        #     curfusion_infs.append(inf.clone())
        # # print(f"first shape: {len(curfusion_infs)}")
        # curfusion_infs.append(q_embed_data)
        # curfusion_side = fusion_fuc(curfusion_infs, gate_num=1)

        # q_posemb = self.position_emb(q_embed_data)
        # q_embed_data = q_embed_data + q_posemb
        # qa_posemb = self.position_emb(qa_embed_data)
        # qa_embed_data = qa_embed_data + qa_posemb

        # qa_pos_embed = qa_embed_data
        # q_pos_embed = q_embed_data

        y = qa_embed_data
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_embed_data #curfusion_side

        # encoder
        
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # curfusion_infs = []
            # for inf in fusion_infs:
            #     curfusion_infs.append(inf.clone())
            # # print(f"second shape: {len(curfusion_infs)}")
            # curfusion_infs.append(x)
            # curfusion_side = fusion_fuc(curfusion_infs, gate_num=1)
            # x = curfusion_side
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
        # assert False
        return x

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

    def forward(self, mask, query, key, values, apply_pos=True):
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
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


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

    def forward(self, q, k, v, mask, zero_pad):

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
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


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

