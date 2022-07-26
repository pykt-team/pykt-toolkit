import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import softmax
import numpy as np
import pandas as pd

device = "cpu" if not torch.cuda.is_available() else "cuda"

# area attention
class AreaAttention(nn.Module):
    def __init__(self,key_query_size,  max_area_width, dropout_rate=0.2):
        super(AreaAttention, self).__init__()
        self.max_area_width = max_area_width
        self.dropout = nn.Dropout(p=dropout_rate)
        self.area_temperature = np.power(key_query_size, 0.5)

    def forward(self, q, key, val, attn_mask):
        # print(f"q: {q}")
        # print(f"key: {key}")
        # print(f"val: {val}")
        keys = self._compute_values(key, "mean")
        masks =self. _compute_masks(attn_mask)
        allvals = self._compute_values(val, "sum")
        # print(f"q: {q.shape}, keys: {keys.shape}, masks: {masks.shape}, vals: {allvals.shape}")
        ws = torch.bmm(q, keys.transpose(1,2))
        masks = torch.repeat_interleave(masks, repeats=key.shape[0], dim=0)
        ws+=masks
        ws = ws / self.area_temperature
        all_weights = softmax(ws, dim=-1)
        all_weights = self.dropout(all_weights)

        # print(f"all_weights: {all_weights}")
        # print(f"allvals: {allvals}")
        finalemb = torch.matmul(all_weights, allvals)
        return finalemb

    def _compute_masks(self, attn_mask):
        # print(f"ori mask: {attn_mask.shape}")
        masks = []
        for i in range(0, self.max_area_width):
            if i == 0:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            else:
                new_attn_mask = torch.zeros_like(attn_mask[:,i:,i:], dtype=torch.float)
                new_attn_mask.masked_fill_(attn_mask[:,i:,i:], float("-inf"))
                
                for j in range(0, i):
                    p1d = (1,0,1,0)
                    new_attn_mask = F.pad(new_attn_mask, p1d, "constant", float("-inf")) 
            # print(new_attn_mask.shape)
            masks.append(new_attn_mask)
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
                v1 = val[:,i:,:]
                v2 = last[:,:-1,:]
                until_curv_sum = v1 + v2
                if merge_type == "sum":
                    curv = until_curv_sum
                else:
                    curv = until_curv_sum / (i + 1)
                pad_val = torch.tensor([0] * curv.shape[-1]).unsqueeze(0).repeat(curv.shape[0], i, 1).to(device)
                paded_curv = torch.cat([pad_val, curv], dim=1)
                # print(paded_curv)
                
            last = until_curv_sum
            vals.append(paded_curv)
        allvals = torch.cat(vals, dim=1)
        return allvals

# def init_weights(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         nn.init.kaiming_normal_(m.weight)

class MultiHeadAreaAttention(nn.Module):
    """ Multi-Head version of Area Attention. """

    # def __init__(self, area_attention: AreaAttention, num_heads: int, key_query_size: int,
    def __init__(self, area_attention: None, num_heads: int, key_query_size: int,
                 key_query_size_hidden: int, value_size: int, value_size_hidden: int):
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
        self.key_query_size = key_query_size
        self.key_query_size_hidden = key_query_size_hidden
        self.value_size = value_size
        self.value_size_hidden = value_size_hidden

        self.query_projection = nn.Linear(key_query_size, num_heads * key_query_size_hidden)
        self.key_projection = nn.Linear(key_query_size, num_heads * key_query_size_hidden)
        self.value_projection = nn.Linear(value_size, num_heads * value_size_hidden)
        self.output_projection = nn.Linear(num_heads * value_size_hidden, value_size)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Multi-Head Area Attention module.
        :param q: queries Tensor with shape (batch_size, num_queries, key_query_size)
        :param k: keys Tensor with shape (batch_size, num_keys_values, key_query_size)
        :param v: values Tensor with shape (batch_size, num_keys_values, value_size)
        :returns a Tensor with shape (batch_size, num_queries, value_size)
        """
        # print(f"query head weight: {self.query_projection.weight}")
        batch_size, num_queries, _ = q.size()
        num_keys_values = k.size(1)
        q = self.query_projection(q).view(batch_size, num_queries, self.num_heads, self.key_query_size_hidden).permute(0, 2, 1, 3).contiguous().flatten(0, 1)
        k = self.key_projection(k).view(batch_size, num_keys_values, self.num_heads, self.key_query_size_hidden).permute(0, 2, 1, 3).contiguous().flatten(0, 1)
        v = self.value_projection(v).view(batch_size, num_keys_values, self.num_heads, self.value_size_hidden).permute(0, 2, 1, 3).contiguous().flatten(0, 1)
        # print(f"after q: {q.shape}, k: {k.shape}")
        # print(f"q: {q}")
        attention = self.area_attention(q, k, v, attn_mask)
        # output = attention
        # print(f"attn: {attention.shape}") # 
        attention = attention.view(batch_size, self.num_heads, num_queries, self.value_size_hidden)
        attention = attention.permute(0, 2, 1, 3).contiguous().flatten(-2, -1)
        output = self.output_projection(attention)
        # print(f"output: {output.shape}")
        # import sys
        # sys.exit()
        return output