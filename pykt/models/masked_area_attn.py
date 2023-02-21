import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
import math

class AreaAttention(nn.Module):
    def __init__(self,key_query_size,  max_area_width, dropout_rate=0.2):
        super(AreaAttention, self).__init__()
        self.max_area_width = max_area_width
        self.dropout = nn.Dropout(p=dropout_rate)
        self.area_temperature = np.power(key_query_size, 0.5)
        

    def forward(self, q, key, val, attn_mask,zero_pad):
        # print(f"q: {q}")
        # print(f"key: {key}")
        # print(f"val: {val}")
        keys = self._compute_values(key, "mean")
        masks =self. _compute_masks(attn_mask,zero_pad)
        allvals = self._compute_values(val, "sum")
        n_head = q.shape[1]
        d_k = q.shape[-1]
        # print(f"q: {q.shape}, keys: {keys.shape}, masks: {masks.shape}, vals: {allvals.shape}")
        # q: torch.Size([64, 8, 200, 256]), keys: torch.Size([64, 8, 400, 256]), masks: torch.Size([1, 1, 200, 400]), vals: torch.Size([64, 8, 400, 256])

        # attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1))
        q = q.view(-1, q.size(-2), q.size(-1))
        keys = keys.view(-1, keys.size(-2), keys.size(-1)).transpose(1,2)
        # print(f"line30 q shape is {q.shape}, keys.shape is {keys.shape}")
        ws = torch.bmm(q, keys)/ math.sqrt(d_k)
            #torch.Size([512, 200, 400]) 512 = 64*8
        # print(f"ws: {ws.shape}")
        ws = ws.view(-1,n_head, ws.size(-2), ws.size(-1))#torch.Size([64, 8, 200, 400])
        # print(f"ws: {ws.shape}")
        
        # masks = torch.repeat_interleave(masks, repeats=key.shape[0], dim=0)
        # ws+=masks
        # ws = ws / self.area_temperature
        
        ws.masked_fill_(masks == 0, -1e32)
        ws = F.softmax(ws, dim=-1)  # BS,8,seqlen,seqlen
        
        
        # if zero_pad:
        #     bs, head,seqlen,seqlen = ws.shape
        #     pad_zero = torch.zeros(bs, head, 1, seqlen).to(attn_mask.device)
        #     new_attn_mask = torch.cat([pad_zero, new_attn_mask[:, :, 1:, :]], dim=2)
            
        all_weights = softmax(ws, dim=-1)
        all_weights = self.dropout(all_weights)
       
        # print(f"all_weights.shape is: {all_weights.shape}")
        # print(f"all_weights is: {all_weights}")
        
            

        # print(f"all_weights: {all_weights}")
        # print(f"allvals: {allvals}")
        finalemb = torch.matmul(all_weights, allvals)
        return finalemb

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

    # def __init__(self, area_attention: AreaAttention, num_heads: int, key_query_size: int,
    def __init__(self, area_attention: None, num_heads, key_query_size,
                 key_query_size_hidden, value_size, value_size_hidden):
        """
        Initializes the Multi-Head Area Attention module.
        :param area_attention: initialized single head Area Attention module
        :param num_heads: number of heads
        :param key_query_size: input size of keys and queries
        :param key_query_size_hidden: hidden size of keys and queries
        :param value_size: input size of values
        :param value_size_hidden: hidden size of values
        """
        super(MultiHeadAreaAttention, self).__init__()
        self.area_attention = area_attention
        self.num_heads = num_heads
        self.key_query_size = key_query_size//self.num_heads
        self.key_query_size_hidden = key_query_size_hidden//self.num_heads
        self.value_size = value_size//self.num_heads
        self.value_size_hidden = value_size_hidden//self.num_heads

        self.query_projection = nn.Linear(key_query_size, key_query_size_hidden)
        self.key_projection = nn.Linear(key_query_size, key_query_size_hidden)
        self.value_projection = nn.Linear(value_size, value_size_hidden)
        self.output_projection = nn.Linear(value_size_hidden, value_size)

    def forward(self, q, k, v, mask,zero_pad):
        """
        Forward pass of the Multi-Head Area Attention module.
        :param q: queries Tensor with shape (batch_size, num_queries, key_query_size)
        :param k: keys Tensor with shape (batch_size, num_keys_values, key_query_size)
        :param v: values Tensor with shape (batch_size, num_keys_values, value_size)
        :returns a Tensor with shape (batch_size, num_queries, value_size)
        """
        # print(f"raw q shape is {q.shape}")
        # print(f"query head weight: {self.query_projection.weight}")
        batch_size, num_queries, _ = q.size()#[64, 200, 256]
        num_keys_values = k.size(1)
        # print(f"new q shape is {self.query_projection(q).shape}")
        q = self.query_projection(q)#[64, 200, 2048]
        q = q.view(batch_size, num_queries, self.num_heads, self.key_query_size_hidden)#[64, 200, 8, 256]
        q = q.permute(0, 2, 1, 3).contiguous()#[64, 8, 200, 256]
        # q = q.flatten(0, 1)#[512, 200, 256]
        k = self.key_projection(k).view(batch_size, num_keys_values, self.num_heads, self.key_query_size_hidden).permute(0, 2, 1, 3).contiguous()#.flatten(0, 1)
        v = self.value_projection(v).view(batch_size, num_keys_values, self.num_heads, self.value_size_hidden).permute(0, 2, 1, 3).contiguous()#.flatten(0, 1)
        # print(f"after q: {q.shape}, k: {k.shape}")
        # print(f"q: {q}")
        attention = self.area_attention(q, k, v, mask,zero_pad = zero_pad)
        # output = attention
        # print(f"attn: {attention.shape}") # 
        attention = attention.view(batch_size, self.num_heads, num_queries, self.value_size_hidden)
        attention = attention.permute(0, 2, 1, 3).contiguous().flatten(-2, -1)
        output = self.output_projection(attention)
        # print(f"output: {output.shape}")
        # import sys
        # sys.exit()
        return output,_