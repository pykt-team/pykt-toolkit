import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .disentangled_attention import DisentangledSelfAttention,build_relative_position
import random
from .utils import change_attn_scores
from .masked_area_attn import MultiHeadAreaAttention,AreaAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class BAKT(nn.Module):
    def __init__(self, n_question, n_pid, 
            d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4, seq_len=200, 
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768, sparse_ratio=0.8, k_index=5, stride=1,max_area_width=3):
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
        self.max_area_width = max_area_width

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
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len,emb_type=self.emb_type,max_area_width=max_area_width)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
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

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)


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
        if emb_type in ["qid", "qidaktrasch", "qid_scalar", "qid_norasch","qid_aa"]:
            d_output, attn_weights = self.model(q_embed_data, qa_embed_data)
            self.attn_weights = attn_weights

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)
        elif emb_type.find("attn") != -1:
            d_output, attn_weights = self.model(q_embed_data, qa_embed_data, emb_type, sparse_ratio, k_index, attn_grads, stride,save_path, save_attn_path, save_grad_path)
            self.attn_weights = attn_weights

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)


        if train:
            return preds, y2, y3
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len,emb_type,max_area_width=3):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'bakt'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same,emb_type=emb_type,block_index=block_index,max_area_width=max_area_width)
                for block_index in range(n_blocks)
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
                 d_ff, n_heads, dropout,  kq_same, emb_type,block_index,c=10000,max_area_width=3):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.emb_type = emb_type
        if self.emb_type in ['qid_aa']:
            area_attn_core = AreaAttention(
                d_model=d_model,
                max_area_width=max_area_width,
                dropout_rate=dropout,
            )
            self.masked_attn_head = MultiHeadAreaAttention(
                area_attention=area_attn_core,
                n_heads=n_heads,
                d_model=d_model,
                kq_same = kq_same,
                dropout=dropout,
                d_feature=d_feature
            )
        else:
            self.masked_attn_head = MultiHeadAttention(
                d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type,block_index=block_index)

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
            if self.emb_type in ['qid_aa']:
                query2,_ = self.masked_attn_head(
                    query, key, values, mask=src_mask,zero_pad=True)
            else:
                query2,_ = self.masked_attn_head(
                    query, key, values, mask=src_mask, zero_pad=True, emb_type=emb_type, sparse_ratio=sparse_ratio, k_index=k_index, attn_grads=attn_grads, stride=stride, save_path=save_path, save_attn_path=save_attn_path, save_grad_path=save_grad_path) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            if self.emb_type in ['qid_aa']:
                query2,_ = self.masked_attn_head(
                    query, key, values, mask=src_mask,zero_pad=False)
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
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True,emb_type="qid",pos_model=None,block_index=0,max_relative_positions=-1,position_buckets=-1):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        if emb_type in ["qid_disentangled_sparse_attn","qid_disentangled_sparseattn"]:
            self.attn = DisentangledSelfAttention(num_attention_heads=n_heads,hidden_size=d_model,hidden_dropout_prob=dropout,attention_probs_dropout_prob=dropout)
            self.max_relative_positions = max_relative_positions
            self.position_buckets = position_buckets
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)

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
            
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), bucket_size = self.position_buckets, max_position=self.max_relative_positions)
        return relative_pos
    
    def forward(self, q, k, v, mask, zero_pad, emb_type="qid", sparse_ratio=0.8, k_index=5, attn_grads=None, stride=1,save_path="", save_attn_path="", save_grad_path=""):
        bs = q.size(0)
        if emb_type in ['qid_disentangled_sparse_attn','qid_disentangled_sparseattn']:
            relative_pos = self.get_rel_pos(q, query_states=None, relative_pos=None)# get relative position 
            scores, attn_weights = self.attn(q,k,v,mask,
                                             zero_pad=zero_pad,
                                             relative_pos=relative_pos,
                                             emb_type=emb_type,
                                             sparse_ratio=sparse_ratio,
                                             k_index=k_index)
            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)
            # concat = attn_result['hidden_states']
            # attn_weights = attn_result['attention_no_softmax']
        else:
            

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
                            mask, self.dropout, zero_pad, emb_type, sparse_ratio=sparse_ratio, k_index=k_index, attn_grads=attn_grads, stride=stride,save_path=save_path, save_attn_path=save_attn_path, save_grad_path=save_grad_path)#scores shape is torch.Size([64, 8, 200, 32])
            # print(f"scores shape is {scores.shape}")#
            
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
    #print(f"q: {q.shape}, keys: {k.shape}, masks: {mask.shape}, vals: {v.shape}")
    #q: torch.Size([64, 8, 200, 32]), keys: torch.Size([64, 8, 200, 32]), masks: torch.Size([1, 1, 200, 200]), vals: torch.Size([64, 8, 200, 32])
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
            math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
     
    scores = change_attn_scores(scores, emb_type, k_index, device,mask)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:bs, :, 1:, :]], dim=2)
       
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, scores


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
