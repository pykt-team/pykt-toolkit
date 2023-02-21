import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
import math
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_

class AreaAttention(nn.Module):
    def __init__(self,d_model,  max_area_width, dropout_rate=0.2):
        super(AreaAttention, self).__init__()
        self.max_area_width = max_area_width
        self.dropout = nn.Dropout(p=dropout_rate)
        self.area_temperature = np.power(d_model, 0.5)
        

    def forward(self, q, k, v,d_k, mask,zero_pad):
        # print(f"q: {q.shape}, keys: {k.shape}, masks: {mask.shape}, vals: {v.shape}")
        #q: torch.Size([64, 8, 200, 32]), keys: torch.Size([64, 8, 200, 32]), masks: torch.Size([1, 1, 200, 200]), vals: torch.Size([64, 8, 200, 32])
        
        k = self._compute_values(k, "mean")
        mask = self._compute_masks(mask,zero_pad)
        v = self._compute_values(v, "sum")
        # print(f"q: {q.shape}, keys: {k.shape}, masks: {mask.shape}, vals: {v.shape}")
        bs = k.shape[0]

        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            math.sqrt(d_k)
        # print(f"scores shape is {scores.shape}")
        scores.masked_fill_(mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def _compute_masks(self, attn_mask,zero_pad):
        # print(f"ori mask: {attn_mask.shape}")
        # print(f"attn_mask is {attn_mask}")
        masks = []
        for i in range(0, self.max_area_width):
            if i == 0:
                # new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
                # new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                new_attn_mask = attn_mask
            else:
                new_attn_mask = attn_mask[:,:,:,i:]
            if zero_pad:
                bs, head,seqlen,seqlen = new_attn_mask.shape
                pad_zero = torch.zeros(bs, head, 1, seqlen).to(attn_mask.device)
                new_attn_mask = torch.cat([pad_zero, new_attn_mask[:, :, 1:, :]], dim=2)
            # print(new_attn_mask.shape)
            masks.append(new_attn_mask)
        # print(masks)
        masks = torch.cat(masks,dim=-1)
        return masks

    def _compute_values(self, val, merge_type="mean"):
        vals = []
        last = None
        for i in range(0, self.max_area_width):
            if i == 0:
                until_curv_sum, paded_curv = val, val
                # print(v)
            else:
                v1 = val[:,:,i:,:]
                v2 = last[:,:,:-1,:]
                until_curv_sum = v1 + v2
                if merge_type == "sum":
                    curv = until_curv_sum#[64, 8, 199, 256]
                else:
                    curv = until_curv_sum / (i + 1)
                # pad_val = torch.tensor([0] * curv.shape[-1]).unsqueeze(0).repeat(curv.shape[0],curv.shape[1], i, 1).to(val.device)#[64, 1, 256]
                # print(f"pad_val shape is {pad_val.shape},curv shape is {curv.shape}")
                # paded_curv = torch.cat([pad_val, curv], dim=2)
                # print(paded_curv)
                paded_curv = curv
                
            last = until_curv_sum
            vals.append(paded_curv)
        allvals = torch.cat(vals, dim=2)
        return allvals


class MultiHeadAreaAttention(nn.Module):
    """ Multi-Head version of Area Attention. """

    # def __init__(self, area_attention: AreaAttention, n_heads: int, d_model: int,
    def __init__(self, area_attention, n_heads,d_feature,d_model,dropout,kq_same, bias=True):
        """
        Initializes the Multi-Head Area Attention module.
        :param area_attention: initialized single head Area Attention module
        :param n_heads: number of heads
        :param d_model: input size of keys and queries
        """
        super(MultiHeadAreaAttention, self).__init__()
        self.d_k = d_feature
        self.d_model = d_model
        self.h = n_heads
        self.area_attention = area_attention
        self.n_heads = n_heads
        self.kq_same = kq_same
        
        self.v_linear = nn.Linear(d_model, d_model,bias=bias)
        self.k_linear = nn.Linear(d_model, d_model,bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model,bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model,bias=bias)
        
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

    def forward(self, q, k, v, mask,zero_pad):
        """
        Forward pass of the Multi-Head Area Attention module.
        :param q: queries Tensor with shape (bs, num_queries, d_model)
        :param k: keys Tensor with shape (bs, num_keys_values, d_model)
        :param v: values Tensor with shape (bs, num_keys_values, d_model)
        :returns a Tensor with shape (bs, num_queries, d_model)
        """
        # print(f"raw q shape is {q.shape}")
        # print(f"query head weight: {self.q_linear.weight}")
        bs, num_queries, _ = q.size()#[64, 200, 256]
        # print(f"new q shape is {self.q_linear(q).shape}")
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        
        if self.kq_same is False:
            q = self.q_linear(q)
        else:
            q = self.k_linear(q)
        q = q.view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # print(f"after q: {q.shape}, k: {k.shape}")
        # print(f"q: {q}")
        scores = self.area_attention(q=q, k=k, v=v,d_k=self.d_k,mask=mask,zero_pad = zero_pad)
        # output = attention
        # print(f"attn: {attention.shape}") # 
        # scores = attention.view(bs, self.n_heads, num_queries, self.d_model)
        # print(f"scores shape is {scores.shape}")#
        concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)
        # print(f"concat shape is {concat.shape}")#
        output = self.out_proj(concat)
        # print(f"output: {output.shape}")
        # import sys
        # sys.exit()
        return output,_