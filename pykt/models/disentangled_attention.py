# The codes copy from https://github.com/microsoft/DeBERTa/tree/master/DeBERTa

"""
  Disentangled SelfAttention module
"""

import pdb
import math
import torch
import functools
import numpy as np
from torch import nn
from .disentangled_attention_ops import *
from functools import lru_cache


def make_log_bucket_position(relative_pos, bucket_size, max_position):
  sign = np.sign(relative_pos)
  mid = bucket_size//2
  abs_pos = np.where((relative_pos<mid) & (relative_pos > -mid), mid-1, np.abs(relative_pos))
  log_pos = np.ceil(np.log(abs_pos/mid)/np.log((max_position-1)/mid) * (mid-1)) + mid
  bucket_pos = np.where(abs_pos<=mid, relative_pos, log_pos*sign).astype(np.int)
  return bucket_pos

@lru_cache(maxsize=128)
def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
  q_ids = np.arange(0, query_size)
  k_ids = np.arange(0, key_size)
  rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0],1))
  if bucket_size>0 and max_position > 0:
    rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
  rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
  rel_pos_ids = rel_pos_ids[:query_size, :]
  rel_pos_ids = rel_pos_ids.unsqueeze(0)
  return rel_pos_ids

def test_log_bucket():
  x=np.arange(-511,511)
  y=make_log_bucket_position(x, 128, 512)
  pdb.set_trace()

class DisentangledSelfAttention(nn.Module):
    """Disentangled self-attention module"""
    def __init__(self, num_attention_heads,hidden_size,max_position_embeddings=200,share_att_key=False,pos_att_type="c2p|p2c",relative_attention=True,position_buckets=-1,max_relative_positions=-1,hidden_dropout_prob=0.1,attention_probs_dropout_prob=0.1):
        """_summary_

        Args:
            num_attention_heads (_type_): number of attention heads
            hidden_size (_type_): hidden size of the model
            max_position_embeddings (int, optional): maximum position embedding. Defaults to 200.
            share_att_key (bool, optional): _description_. Defaults to False.
            pos_att_type (str, optional): _description_. Defaults to "c2p|p2c".
            relative_attention (bool, optional): _description_. Defaults to True.
            position_buckets (int, optional): _description_. Defaults to -1.
            max_relative_positions (int, optional): _description_. Defaults to -1.
            hidden_dropout_prob (float, optional): _description_. Defaults to 0.1.
            attention_probs_dropout_prob (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        self.num_attention_heads = num_attention_heads
        _attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_head_size = _attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)

        self.share_att_key = share_att_key
        self.pos_att_type = [x.strip() for x in pos_att_type.lower().split('|')] # c2p|p2c
        self.relative_attention = relative_attention
        

        if self.relative_attention:
            self.position_buckets = position_buckets
            self.max_relative_positions = max_relative_positions
            if self.max_relative_positions <1:
                self.max_relative_positions = max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets>0:
                self.pos_ebd_size = self.position_buckets
                # For backward compitable
            self.rel_embeddings = nn.Embedding(self.pos_ebd_size*2, hidden_size)
            self.pos_dropout = StableDropout(hidden_dropout_prob)

            if (not self.share_att_key):
                if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)
                if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = StableDropout(attention_probs_dropout_prob)
        self._register_load_state_dict_pre_hook(self._pre_load_hook)

    def transpose_for_scores(self, x, attention_heads):
        # Create a new shape for x by concatenating the first elements of x.size() with attention_heads and -1 as the last two elements
        # Split raw input x into multi heads
        new_x_shape = x.size()[:-1] + (attention_heads, -1)# B*seq*head*dim
        # Reshape x to the new shape defined in new_x_shape
        x = x.view(*new_x_shape)
        #Permute the dimensions of x to (0, 2, 1, 3) and return the contiguous version of the reshaped x
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    

    def forward(self, q,k,v, attention_mask, return_att=False, query_states=None, relative_pos=None,zero_pad=False):
        """_summary_

        Args:
            q (_type_): B*seq*dim
            k (_type_): B*seq*dim
            v (_type_): B*seq*dim
            attention_mask (_type_): 1*1*seq*seq
            return_att (bool, optional): _description_. Defaults to False.
            query_states (_type_, optional): _description_. Defaults to None.
            relative_pos (_type_, optional): _description_. Defaults to None.
            zero_pad (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        query_layer = self.transpose_for_scores(self.query_proj(q), self.num_attention_heads).float()# B*seq*head_dim
        # print("line 125 query_layer",query_layer.size())
        key_layer = self.transpose_for_scores(self.key_proj(k), self.num_attention_heads).float()
        value_layer = self.transpose_for_scores(self.value_proj(v), self.num_attention_heads)
        
        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        scale = 1/math.sqrt(query_layer.size(-1)*scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2)*scale)# content to content attention, B*seq*seq
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(self.rel_embeddings.weight)
            rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = (attention_scores + rel_att)
        attention_scores = (attention_scores - attention_scores.max(dim=-1, keepdim=True).values.detach()).to(q)
        attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1))#B*num_head*seq*seq
        
        # bxhxlxd
        rmask = ~(attention_mask.bool())
        output_no_softmax = attention_scores.masked_fill(rmask, float('-inf'))#mask True的地方置-inf
        # print("output_no_softmax[0] is",output_no_softmax[0])
        # print("rmask[0] is ",rmask[0])
        _attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(_attention_probs)#B*num_head*seq*seq
 
        if zero_pad:
            # print(f"raw attention_probs is {attention_probs}")
            attention_probs[:,:,0,:] = 0# 第一行score置0,不参与attention，softmax后其实就是0
            # print(f"attention_probs shape is {attention_probs.shape}")
            # print(f"attention_probs is {attention_probs}")
           
        context_layer = torch.bmm(attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer)
        context_layer = context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1)).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)# B*seq*dim
        # print("line 163 context_layer",context_layer.size())
        # print(f"hidden_states shape is {context_layer.shape}")
        return {
            'hidden_states': context_layer,#计算完的特征
            'attention_probs': _attention_probs,#经过softmax的attention，且mask
            'attention_no_softmax': output_no_softmax,#未经过softmax的attention，且mask
            'attention_logits': attention_scores#计算完的原始attention score，未mask
            }

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        """Get relative attention score
        Args:
            query_layer (_type_): query layer
            key_layer (_type_): key layer
            relative_pos (_type_): relative position encoding
            rel_embeddings (_type_): relative position embedding
            scale_factor (_type_): scale factor for attention score
        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions)
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(0) #.repeat(query_layer.size(0)//self.num_attention_heads, 1, 1)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
        else:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            scale = 1/math.sqrt(pos_key_layer.size(-1)*scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2).to(query_layer)*scale)
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]))
            score += c2p_att

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            scale = 1/math.sqrt(pos_query_layer.size(-1)*scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions).to(query_layer.device)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2).to(key_layer)*scale)
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)])).transpose(-1,-2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))))
            score += p2c_att

        # position->position
        if 'p2p' in self.pos_att_type:
            pos_query = pos_query_layer[:,:,att_span:,:]
            p2p_att = torch.matmul(pos_query, pos_key_layer.transpose(-1, -2))
            p2p_att = p2p_att.expand(query_layer.size()[:2] + p2p_att.size()[2:])
            if query_layer.size(-2) != key_layer.size(-2):
                p2p_att = torch.gather(p2p_att, dim=-2, index=pos_index.expand(query_layer.size()[:2] + (pos_index.size(-2), p2p_att.size(-1))))
            p2p_att = torch.gather(p2p_att, dim=-1, index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]))
            score += p2p_att

        return score

    def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs):
        self_state = self.state_dict()
        if ((prefix + 'query_proj.weight') not in state_dict) and ((prefix + 'in_proj.weight') in state_dict):
          v1_proj = state_dict[prefix+'in_proj.weight']
          v1_proj = v1_proj.unsqueeze(0).reshape(self.num_attention_heads, -1, v1_proj.size(-1))
          q,k,v=v1_proj.chunk(3, dim=1)
          state_dict[prefix + 'query_proj.weight'] = q.reshape(-1, v1_proj.size(-1))
          state_dict[prefix + 'key_proj.weight'] = k.reshape(-1, v1_proj.size(-1))
          state_dict[prefix + 'key_proj.bias'] = self_state['key_proj.bias']
          state_dict[prefix + 'value_proj.weight'] = v.reshape(-1, v1_proj.size(-1))
          v1_query_bias = state_dict[prefix + 'q_bias']
          state_dict[prefix + 'query_proj.bias'] = v1_query_bias
          v1_value_bias = state_dict[prefix +'v_bias']
          state_dict[prefix + 'value_proj.bias'] = v1_value_bias

          v1_pos_key_proj = state_dict[prefix + 'pos_proj.weight']
          state_dict[prefix + 'pos_key_proj.weight'] = v1_pos_key_proj
          v1_pos_query_proj = state_dict[prefix + 'pos_q_proj.weight']
          state_dict[prefix + 'pos_query_proj.weight'] = v1_pos_query_proj
          v1_pos_query_proj_bias = state_dict[prefix + 'pos_q_proj.bias']
          state_dict[prefix + 'pos_query_proj.bias'] = v1_pos_query_proj_bias
          state_dict[prefix + 'pos_key_proj.bias'] = self_state['pos_key_proj.bias']

          del state_dict[prefix + 'in_proj.weight']
          del state_dict[prefix + 'q_bias']
          del state_dict[prefix + 'v_bias']
          del state_dict[prefix + 'pos_proj.weight']
          del state_dict[prefix + 'pos_q_proj.weight']
          del state_dict[prefix + 'pos_q_proj.bias']
