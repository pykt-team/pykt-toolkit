"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from .que_base_model import QueBaseModel,QueEmb
from pykt.utils import debug_print


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.emb_size % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.emb_size, 3 * config.emb_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.emb_size, config.emb_size, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.emb_size = config.emb_size
        self.dropout = config.dropout

     
        self.mask = torch.tril(torch.ones(config.seq_len-1, config.seq_len-1)).view(1, 1, config.seq_len-1, config.seq_len-1)
        
    def forward(self, q,k,v):
        B, T, C = q.size() # batch size, sequence length, embedding dimensionality (emb_size)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.emb_size, 4 * config.emb_size, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.emb_size, config.emb_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.emb_size, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.emb_size, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, q,k,v):
        q,k,v = self.ln_1(q),self.ln_1(k),self.ln_1(v)
        x = q + self.attn(q,k,v)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.emb_type = config.emb_type
        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(config.seq_len, config.emb_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.emb_size, bias=config.bias),
        ))
        
        self.que_emb_list = nn.ModuleList([QueEmb(num_q=dataset_config['num_q'],num_c=dataset_config['num_c'],emb_size=config.emb_size,emb_type=config.emb_type,model_name=config.model_name,device=config.device,
                             emb_path=config.emb_path,pretrain_dim=config.pretrain_dim) for dataset_config in self.config.dataconfig_list])
        
        self.emb_pooling = nn.Linear(config.emb_size*2, config.emb_size)
        
        self.r_emb = nn.Embedding(2, config.emb_size)

        self.out_layer = nn.Sequential(
            nn.Linear(config.emb_size*2, config.emb_size, bias=config.bias),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.emb_size, 1, bias=config.bias)
            )
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, q, c ,r,data=None):
        pos = torch.arange(0, self.config.seq_len, dtype=torch.long, device=q.device).unsqueeze(0) # shape (1, t)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, emb_size)

        # Get the embeddings
        que_emb_index = self.config.source_list.index(data['source'][0])
        raw_que_emb = self.que_emb_list[que_emb_index](q,c)
        
        q_emb_full = self.emb_pooling(raw_que_emb) + pos_emb

        q_shift = q_emb_full[:,1:]
        q_emb = q_emb_full[:,:-1]
        
        r_emb = self.r_emb(r)[:,:-1]
        inter_emb = q_emb + r_emb

        
        # forward the GPT model itself
        x = q_emb
        for block in self.transformer.h:
            x = block(x,x,inter_emb)
        x = self.transformer.ln_f(x)
        # print(f"x shape: {x.shape}")
        x = torch.cat([q_shift,x],dim=-1)
        logits = self.out_layer(x).squeeze(-1)

        y = torch.sigmoid(logits)
        # print(f"y shape: {y.shape},logits shape: {logits.shape}")

        outputs = {"y":y,"logits":logits}
        return outputs

@dataclass
class GPTConfig:
    n_layer: int = 2
    n_head: int = 8
    dropout: float = 0.0
    bias: bool = True
    seq_len: int = 200
    num_q: int=0
    num_c: int=0
    emb_size: int=256
    emb_type: str = 'qid'
    emb_path: str = ""
    pretrain_dim: int = 768
    device: str = 'cpu'
    seed: int = 0
    model_name: str = "gpt_mt"
    dataconfig_list: list = None  # Added attribute
    source_list: list = None      # Added attribute


class GPTMT(QueBaseModel):
    def __init__(self, num_q, num_c, emb_size,seq_len=200,dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768,device='cpu',seed=0,n_head=8,n_layer=8,dataconfig_list=None,source_list=None):
        self.config = GPTConfig(n_layer=n_layer,
                                n_head=n_head,
                                dropout=dropout,
                                bias=True,
                                seq_len=seq_len,
                                num_q=num_q,
                                num_c=num_c,
                                emb_size=emb_size,
                                emb_type=emb_type,
                                emb_path=emb_path,
                                pretrain_dim=pretrain_dim,
                                device=device,
                                seed=seed,
                                dataconfig_list=dataconfig_list,
                                source_list=source_list
                                )

        model_name = self.config.model_name
        debug_print(f"emb_type is {emb_type}",fuc_name=model_name)

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = GPTNet(self.config)
       
        self.model = self.model.to(device)
        self.emb_type = self.config.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
       
    def train_one_step(self,data,process=True,return_all=False):
        outputs,data_new = self.predict_one_step(data,return_details=True,process=process)
        loss = self.get_loss(outputs['y'],data_new['rshft'],data_new['sm'])
        return outputs['y'],loss#y_question没用

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        data_new = self.batch_to_device(data,process=process)
        outputs = self.model(data_new['cq'].long(),data_new['cc'],data_new['cr'].long(),data=data_new)
        if return_details:
            return outputs,data_new
        else:
            return outputs['y']