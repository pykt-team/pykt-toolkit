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
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn import metrics
import random
from scipy.special import softmax

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
        
        self.que_emb_list = nn.ModuleList([QueEmb(num_q=dataset_config['num_q'],
                                                  num_c=dataset_config['num_c'],
                                                  emb_size=config.emb_size,
                                                  emb_type=config.emb_type,
                                                  model_name=config.model_name,
                                                  device=config.device,
                                                  emb_path=config.emb_path,
                                                  pretrain_dim=config.pretrain_dim) for dataset_config in self.config.dataconfig_list])
        
        self.emb_pooling = nn.Linear(config.emb_size*2, config.emb_size)
        
        self.r_emb = nn.Embedding(2, config.emb_size)

        if config.share_output:
            print("Share output layer")
            self.out_layer = nn.Sequential(
                nn.Linear(config.emb_size*2, config.emb_size, bias=config.bias),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.emb_size, 1, bias=config.bias)
                )
        else:
            print("Not share output layer")
            self.out_layer = nn.ModuleList([nn.Sequential(
                nn.Linear(config.emb_size*2, config.emb_size, bias=config.bias),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.emb_size, 1, bias=config.bias)
                ) for _ in self.config.dataconfig_list])
        
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
        que_emb_index = self.config.source_list.index(data['source'])
        
        # print(f"que_emb_index: {que_emb_index},source: {data['source'][0]}")
        # print(f"q shape: {q.shape},c shape: {c.shape},r shape: {r.shape},q is {q}")
        # print(f"data['source'] is {data['source']}")
        
        raw_que_emb = self.que_emb_list[que_emb_index](q,c)
        # print(f"raw_que_emb shape: {raw_que_emb.shape},pos_emb shape: {pos_emb.shape}")
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
        if self.config.share_output:
            logits = self.out_layer(x).squeeze(-1)
        else:
            logits = self.out_layer[que_emb_index](x).squeeze(-1)
        
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
    return_dict: bool = False
    share_output: bool = False
    model_name: str = "gpt_mt"
    source_weight_t: float = 1
    dataconfig_list: list = None  # Added attribute
    source_list: list = None      # Added attribute


class GPTMT(QueBaseModel):
    def __init__(self, num_q, num_c, emb_size,seq_len=200,dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768,device='cpu',seed=0,n_head=8,n_layer=8,dataconfig_list=None,source_list=None,return_dict=False,source_weight_t=1,share_output=False):
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
                                source_list=source_list,
                                return_dict=return_dict,
                                source_weight_t = source_weight_t,
                                share_output=share_output
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
        
    def train(self,train_dataset_list, valid_dataset_list,source_list,batch_size=16,valid_batch_size=None,num_epochs=32, test_loader=None, test_window_loader=None,save_dir="tmp",save_model=False,patient=10,shuffle=True,process=True):
        self.save_dir = save_dir
        os.makedirs(self.save_dir,exist_ok=True)

        if valid_batch_size is None:
            valid_batch_size = batch_size

        data_list = []
        for train_dataset,valid_dataset,source in zip(train_dataset_list,valid_dataset_list,source_list):
            train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=shuffle)
            for data in train_loader:
                data['source'] = source
                data_list.append(data)
        random.shuffle(data_list)
            # data_loder_list.append(train_loader)
        
        max_auc, best_epoch = 0, -1
        train_step = 0
        weights = np.ones(len(source_list))
        all_auc_dict = {}
        all_acc_dict = {}
        for i in range(1, num_epochs + 1):
            loss_list = []
            for data in data_list:
                # data['source'] = source
                train_step += 1
                self.model.train()
                y,loss = self.train_one_step(data,process=process)
                self.opt.zero_grad()
                loss = (loss * weights[source_list.index(data['source'])])
                loss.backward()#compute gradients 
                self.opt.step()#update model’s parameters
                loss_list.append(loss.detach().cpu().numpy())

            tmp_auc_list = []
            tmp_acc_list = []
                  
            for valid_dataset,source in zip(valid_dataset_list,source_list):
                if source not in all_auc_dict:
                    all_auc_dict[source] = []
                    all_acc_dict[source] = []
                
                eval_result = self.evaluate(valid_dataset,batch_size=valid_batch_size,source=source)
                auc, acc = eval_result['auc'],eval_result['acc']
                all_auc_dict[source].append(auc)
                all_acc_dict[source].append(acc)
                tmp_auc_list.append(auc)
                tmp_acc_list.append(acc)
                # 
                # print(f"{source}'s eval_result is {eval_result}")
            weights = 1 - np.array(tmp_auc_list)
            weights = softmax(weights/self.config.source_weight_t)*len(tmp_auc_list)
            print(f"weights is {weights}, source_weight_t is {self.config.source_weight_t}")

            auc = np.mean(tmp_auc_list)
            acc = np.mean(tmp_acc_list)
            loss_mean = np.mean(loss_list)
            if auc > max_auc+1e-3:
                if save_model:
                    self._save_model()
                max_auc = auc
                best_epoch = i
                testauc, testacc = -1, -1
                window_testauc, window_testacc = -1, -1
                validauc, validacc = auc, acc
            print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {self.model.emb_type}, model: {self.model.model_name}, save_dir: {self.save_dir}")
            print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")
            
            model_result_data = {}
            for source in source_list:
                best_epoch = all_auc_dict[source].index(max(all_auc_dict[source]))
                source_auc = all_auc_dict[source][best_epoch]
                source_acc = all_acc_dict[source][best_epoch]
                print(f"            {source} best epoch: {best_epoch}, best auc: {source_auc:.4}, best acc: {source_acc:.4}")
                model_result_data['best_epoch_'+source] = best_epoch
                model_result_data['best_auc_'+source] = source_auc
                model_result_data['best_acc_'+source] = source_acc

            if i - best_epoch >= patient:
                break
        if self.config.return_dict:
            model_result = {"best_epoch":best_epoch,"best_auc":max_auc,
                            "testauc":testauc,"testacc":testacc,"window_testauc":window_testauc,"window_testacc":window_testacc,"validauc":validauc,"validacc":validacc}
            model_result.update(model_result_data)
            return model_result
        else:
            return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch

    def evaluate(self,dataset,batch_size,acc_threshold=0.5,source=None):
        ps,ts = self.predict(dataset,batch_size=batch_size,source=source)
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= acc_threshold else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        eval_result = {"auc":auc,"acc":acc}
        return eval_result
    
    def predict(self,dataset,batch_size,return_ts=False,process=True,source=None):
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_scores = []
            for data in test_loader:
                new_data = self.batch_to_device(data,process=process)
                data['source'] = source
                y = self.predict_one_step(data)
                y = torch.masked_select(y, new_data['sm']).detach().cpu()
                t = torch.masked_select(new_data['rshft'], new_data['sm']).detach().cpu()
                y_trues.append(t.numpy())
                y_scores.append(y.numpy())
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        return ps,ts