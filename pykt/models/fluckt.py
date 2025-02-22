import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from math import exp
from math import log
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Optional, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)

    def forward(self, x):
        return self.conv(x)[:, :, :-(self.conv.padding[0])]  


class FrequencyLayer(nn.Module):
    def __init__(self, dropout, hidden_size, kernel_size):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.causal_conv = CausalConv1d(hidden_size, hidden_size, kernel_size)
        self.c = kernel_size // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        # causal
        input_tensor = input_tensor.permute(0, 2, 1)  # [batch, hidden, seq_len]
        low_pass = self.causal_conv(input_tensor)
        low_pass = low_pass.permute(0, 2, 1)  # [batch, seq_len, hidden]

        # high 
        high_pass = input_tensor.permute(0, 2, 1) - low_pass  # [batch, seq_len, hidden]
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor.permute(0, 2, 1))

        return hidden_states

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * x + self.bias



class FlucKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256, bar_d=128,
            kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, seq_len=200, emb_type="qid",emb_path="", pretrain_dim=768,num_buckets=32,max_distance=100,n_hat=0,max_local_shift=0.0,d_state=64,d_conv=4,expand=1,kernel_size=5):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "fluckt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model

        self.num_buckets = num_buckets
        self.max_distance = max_distance

        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上
        
        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l) # interaction emb
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object. It contains stack of attention block
        self.model = Framework(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,bar_d=bar_d,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=self.emb_type,num_buckets = self.num_buckets,max_distance = self.max_distance, n_hat=n_hat,max_local_shift=max_local_shift,d_state=d_state,d_conv=d_conv,expand=expand,kernel_size=kernel_size)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
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

    def forward(self, q_data, target, pid_data=None, qtest=False):
        emb_type = self.emb_type
        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(
                target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2 # rasch部分loss
        else:
            c_reg_loss = 0.

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)
        if not qtest:
            return preds, c_reg_loss
        else:
            return preds, c_reg_loss, concat_q


class Framework(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout,bar_d, kq_same, model_type, seq_len, emb_type,num_buckets,max_distance,n_hat,max_local_shift,d_state,d_conv,expand,kernel_size):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.emb_type = emb_type


        
        self.filter_layer = FrequencyLayer(dropout,d_model,kernel_size)
        # print(f"kernel_size:{kernel_size}, now you are in conv.")


        if model_type in {'fluckt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type,num_buckets = self.num_buckets,max_distance = self.max_distance,bar_d=bar_d)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type,num_buckets = self.num_buckets,max_distance = self.max_distance,bar_d=bar_d)
                for _ in range(n_blocks*2)
            ])




    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed


        if self.emb_type.find("conv") != -1:
            x = self.filter_layer(x)
            y = self.filter_layer(y)

        # encoder
        for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data) # yt^
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False, pdiff=pid_embed_data) # False: 没有FFN, 第一层只有self attention, 对应于xt^
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
                # print(x[0,0,:])
                flag_first = True
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same, emb_type,num_buckets,max_distance,bar_d):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type,num_buckets = self.num_buckets,max_distance = self.max_distance,bar_d=bar_d)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
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
                query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, num_buckets,max_distance,bar_d, bias=True, emb_type="qid"):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.emb_type = emb_type
        self.bar_d = bar_d


        if emb_type in ["qid_conv_kerple","qid_conv_ker_noexp"]:
            self.kernel_bias = ParallelKerpleLog(n_heads)

        if emb_type.startswith("qid"):
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
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.xavier_uniform_(self.gammas)
            self._reset_parameters()

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))  # 2*(-(8 / n))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
            else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
        
        maxpos = 1000
        attn_heads = n_heads  
        
        context_position = torch.arange(maxpos)[:, None].cuda()
        memory_position = torch.arange(maxpos)[None, :].cuda()
        relative_position = memory_position - context_position 
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(attn_heads, -1,-1)

        self.slopes = torch.Tensor(get_slopes(attn_heads)).cuda()*-1
        self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * relative_position
        self.alibi = self.alibi.view(1, attn_heads, maxpos, maxpos)


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
            # constant_(self.attnlinear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):

        bs = q.size(0)

        if self.emb_type.startswith("qid"):
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
            gammas = self.gammas
            if self.emb_type.find("pdiff") == -1:
                pdiff = None
            if self.emb_type in ["qid_conv_kerple","qid_conv_ker_noexp"]:
                scores = attention(q=q, k=k, v=v, d_k=self.d_k,mask=mask, dropout=self.dropout, zero_pad=zero_pad, gamma=gammas, pdiff=pdiff, alibi = self.alibi,emb_type=self.emb_type, bar_d=self.bar_d,kernel_bias=self.kernel_bias)
            else:
                scores = attention(q=q, k=k, v=v, d_k=self.d_k,mask=mask, dropout=self.dropout, zero_pad=zero_pad, gamma=gammas, pdiff=pdiff, alibi = self.alibi,emb_type=self.emb_type, bar_d=self.bar_d)

            # concatenate heads and put through final linear layer
            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            # # need: torch.Size([64, 1, 200]), scores: torch.Size([64, 200, 200]), v: torch.Size([64, 200, 32])
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1) # 所有v后置一位
        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad,bar_d, gamma=None, pdiff=None, alibi=None, emb_type=None, rel_pos_bias=None, rotary_pe=None,kernel_bias=None):
    """
    This is called by Multi-head atention object to find the values.
    """

    temperature = 1


    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)


    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    # ablation study
    if emb_type.find("noalib") != -1:
        scores = scores

    elif emb_type in ["qid_conv_ker_noexp"]:
        scores = kernel_bias(scores)
        # print("no exp")


    elif emb_type in ["qid_conv_kerple"]:
        seq_len = scores.size()[-1]
        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
            scores_ = scores_ * mask.float().to(device) # 结果和上一步一样
            distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
            disttotal_scores = torch.sum(
                scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
            # print(f"distotal_scores: {disttotal_scores}")
            position_effect = torch.abs(
                x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
            # bs, 8, sl, sl positive distance
            dist_scores = torch.clamp(
                (disttotal_scores-distcum_scores)*position_effect, min=0.) # score <0 时，设置为0
            dist_scores = dist_scores.sqrt().detach()
        m = nn.Softplus()
        gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
        if pdiff == None:
            total_effect = torch.clamp(torch.clamp(
                (dist_scores*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
        else:
            diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
            diff = diff.sigmoid().exp()
            total_effect = torch.clamp(torch.clamp(
                (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
        scores = scores * total_effect
        scores = kernel_bias(scores)
        # print("now")


    # longrope + noexp
    elif emb_type in ["qid_conv_noexp"]:
        scores = scores


    elif emb_type in ["qid_noexp"]:
        seq_len = scores.size()[-1]
        scores = scores+alibi[:, :, :seq_len, :seq_len]
        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
            scores_ = scores_ * mask.float().to(device) # 结果和上一步一样

    else:
        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)  # BS,heads,seqlen,seqlen
            scores_ = scores_ * mask.float().to(device) 
            distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, heads, sl, sl
            disttotal_scores = torch.sum(
                scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
            # print(f"distotal_scores: {disttotal_scores}")
            position_effect = torch.abs(
                x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
            # bs, 8, sl, sl positive distance
            dist_scores = torch.clamp(
                (disttotal_scores-distcum_scores)*position_effect, min=0.) # score <0 时，设置为0
            dist_scores = dist_scores.sqrt().detach()
        m = nn.Softplus()
        gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
        if pdiff == None:
            total_effect = torch.clamp(torch.clamp(
                (dist_scores*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
        else:
            diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
            diff = diff.sigmoid().exp()
            total_effect = torch.clamp(torch.clamp(
                (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
        scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores/temperature, dim=-1)  # BS,8,seqlen,seqlen

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output




class ParallelKerpleLog(nn.Module):
    """Kernel Bias"""
    def __init__(self, num_attention_heads):
        super().__init__()
        self.heads = num_attention_heads  # int
        self.num_heads_per_partition = self.heads  # int
        # self.pos_emb = pos_emb  # str
        self.eps = 1e-2
        
        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                    self.num_heads_per_partition,
                    dtype=torch.float32,
                )[:, None, None] * scale)
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                    self.num_heads_per_partition,
                    dtype=torch.float32,
                )[:, None, None] * scale)
        
        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')
        self.cached_matrix = None
        self.cached_seq_len = None
    
    def stats(self):
        def get_stats(name, obj):
            return {
                name + '_mean': obj.mean().detach().cpu(),
                name + '_std': obj.std().detach().cpu(),
                name + '_max': obj.max().detach().cpu(),
                name + '_min': obj.min().detach().cpu()
            }
        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd
    
    def forward(self, x):
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p * torch.log(1 + self.bias_a * diff)  # log kernel
        
        if seq_len_q != seq_len_k:  
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            
            if not isinstance(bias, float):
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])
        return x + bias