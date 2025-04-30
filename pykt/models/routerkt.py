import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class RouterKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256, 
            kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768, balance_loss_weight=0.001, **kwargs):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "routerkt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.balance_loss_weight = balance_loss_weight
        embed_l = d_model


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
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type)

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
        
        # Calculate balance loss from MoH attention layers
        balance_loss = self.model.get_balance_loss()
        
        if not qtest:
            return preds, c_reg_loss + self.balance_loss_weight * balance_loss
        else:
            return preds, c_reg_loss + self.balance_loss_weight * balance_loss, concat_q


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'routerkt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
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

    def get_balance_loss(self):
        balance_loss = 0
        for block in self.blocks_1:
            balance_loss += block.masked_attn_head.get_balance_loss()
        for block in self.blocks_2:
            balance_loss += block.masked_attn_head.get_balance_loss()
        return balance_loss

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same, emb_type):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MoHAttention(
            d_model, d_feature, n_heads, n_shared_heads=1, n_selected_heads=2, dropout=dropout, kq_same=kq_same)

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


class MoHAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, n_shared_heads, 
                 n_selected_heads, dropout, kq_same,
                 seq_len=200, routing_mode="dynamic"):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.h_shared = n_shared_heads
        self.h_selected = n_selected_heads
        self.kq_same = kq_same
        self.routing_mode = routing_mode
        
        # Linear layers for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        if not kq_same:
            self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Routing networks
        if routing_mode == "dynamic":
            self.wg = nn.Linear(d_model, n_heads - n_shared_heads, bias=False)  # Router for dynamic heads
            
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
        # Track routing statistics for load balancing
        self.register_buffer('head_selections', torch.zeros(n_heads - n_shared_heads))
        self.register_buffer('head_routing_probs', torch.zeros(n_heads - n_shared_heads))
        
    def get_balance_loss(self):
        # Calculate load balance loss for dynamic heads
        f = self.head_selections / (self.head_selections.sum() + 1e-5)
        P = self.head_routing_probs / (self.head_routing_probs.sum() + 1e-5)
        balance_loss = (f * P).sum()
        return balance_loss
        
    def forward(self, q, k, v, mask=None, zero_pad=False, pdiff=None):
        bs = q.size(0)
        seq_len = q.size(1)
        
        # Linear projections
        q = self.q_linear(q)  # [bs, seq_len, d_model]
        if self.kq_same:
            k = q
        else:
            k = self.k_linear(k)
        v = self.v_linear(v)
        
        # Reshape for attention computation
        q = q.view(bs, -1, self.h, self.d_k).transpose(1, 2)  # [bs, h, seq_len, d_k]
        k = k.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [bs, h, seq_len, seq_len]
        
            
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # First position zero padding
        if zero_pad:
            pad_zero = torch.zeros(bs, self.h, 1, scores.size(-1)).to(q.device)
            scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
        
        # Calculate routing scores for dynamic heads
        q_for_routing = q.permute(0, 2, 1, 3).reshape(bs * seq_len, self.h * self.d_k)  # [bs*seq_len, h*d_k]
        
        # Handle dynamic heads routing
        if self.routing_mode == "dynamic":
            # Use learned routing weights
            logits = self.wg(q_for_routing)  # [bs*seq_len, n_dynamic_heads]
            gates = F.softmax(logits, dim=1)  # [bs*seq_len, n_dynamic_heads]
        else:  # query_norm mode
            # Calculate L2 norms for dynamic heads
            q_for_routing = q.permute(0, 2, 1, 3).reshape(-1, self.h, self.d_k)
            logits = torch.stack([
                torch.norm(q_for_routing[:, i, :], p=2, dim=1) 
                for i in range(self.h_shared, self.h)
            ], dim=1)  # [bs*seq_len, n_dynamic_heads]
            
            # Normalize logits
            logits_std = logits.std(dim=1, keepdim=True)
            logits_norm = logits / (logits_std / 1)
            gates = F.softmax(logits_norm, dim=1)  # [bs*seq_len, n_dynamic_heads]
        
        # Select top-k heads
        _, indices = torch.topk(gates, k=self.h_selected, dim=1)
        dynamic_mask = torch.zeros_like(gates).scatter_(1, indices, 1.0)
        
        self.dynamic_scores = gates * dynamic_mask
        
        # Update routing statistics
        self.head_routing_probs = gates.mean(dim=0)
        self.head_selections = dynamic_mask.sum(dim=0)
        
        # Handle shared heads routing
        # All shared heads have equal weight of 1.0
        self.shared_scores = torch.ones(bs, seq_len, self.h_shared).to(q.device)
        
        dynamic_scores_reshaped = self.dynamic_scores.view(bs, seq_len, -1)
        routing_mask = torch.zeros(bs, seq_len, self.h).to(q.device)
        routing_mask[:, :, :self.h_shared] = 1.0  # Shared heads always active
        routing_mask[:, :, self.h_shared:] = dynamic_scores_reshaped  # Add dynamic head weights
        
        # Reshape routing mask to match attention dimensions [bs, h, seq_len, 1]
        routing_mask = routing_mask.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        
        # Apply attention
        attn = self.dropout(torch.softmax(scores, dim=-1))
        
        # Save attention maps for visualization
        self.attention_maps = attn.detach().clone()  # [bs, h, seq_len, seq_len]
        
        context = torch.matmul(attn, v)  # [bs, h, seq_len, d_k]
        
        # Apply routing mask
        context = context * routing_mask
        
        # Combine heads
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        return self.out(context)