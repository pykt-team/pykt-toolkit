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
from enum import IntEnum

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

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
        emb_size = config.emb_size
        dropout = config.dropout
        assert config.emb_size % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # output projection
        
        self.c_proj = nn.Linear(emb_size, emb_size, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = config.n_head

     
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
        emb_size = config.emb_size
        dropout = config.dropout
        self.c_fc    = nn.Linear(emb_size, 4 * emb_size, bias=config.bias)
        self.c_proj  = nn.Linear(4 * emb_size, emb_size, bias=config.bias)
        self.dropout = nn.Dropout(dropout)

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


class OutputLayer(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))

class MultiLabelClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sigmoid = nn.Sigmoid()
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.mlp = OutputLayer(config.mlp_layer_num, config.emb_size*2, config.num_c, config.dropout)
        self.num_c = config.num_c
  
    def forward(self, x, label=None):
        # x: [batch_size, seq_len, emb_size]
        num_c = self.num_c
        y = self.mlp(x)
        loss = None
        if label is not None:
            input_one_hot = F.one_hot(
                torch.where(label != -1, label, 0), num_c)
            mask = label != -1
            # 将mask扩展到与input_one_hot相同的形状
            mask_expanded = mask.unsqueeze(-1).expand_as(input_one_hot)
            # input_one_hot*mask_flatten
            label = (input_one_hot*mask_expanded).sum(axis=2).double()

            y_flatten = y.view(-1, num_c)
            label_flatten = label.view(-1, num_c)
            keep_index = label_flatten.sum(axis=-1) != 0
            
            y_flatten = y_flatten[keep_index]
            label_flatten = label_flatten[keep_index]
            loss = self.loss_fct(y_flatten, label_flatten)
            
        y = self.sigmoid(y)
        return y, loss
    


class GPTNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.emb_type = config.emb_type
        emb_size = config.emb_size
        mlp_layer_num = config.mlp_layer_num
        dropout = config.dropout

        self.transformer = nn.ModuleDict(dict(
            wpe = CosinePositionalEmbedding(d_model=emb_size, max_len=config.seq_len),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(emb_size, bias=config.bias),
        ))


        self.que_emb_list = nn.ModuleList([QueEmb(num_q=dataset_config['num_q'],
                                                  num_c=dataset_config['num_c'],
                                                  emb_size=emb_size,
                                                  emb_type=config.emb_type,
                                                  model_name=config.model_name,
                                                  device=config.device,
                                                  emb_path=config.emb_path,
                                                  pretrain_dim=config.pretrain_dim) for dataset_config in self.config.dataconfig_list])
        
        self.emb_pooling = nn.Linear(emb_size*2, emb_size)
        self.concept_lstm_layer = nn.LSTM(emb_size*2, emb_size, batch_first=True)
        
        self.r_emb = nn.Embedding(2, emb_size)

       
        if config.share_output:
            print("Share output layer")
            self.out_layer_q_next = OutputLayer(mlp_layer_num, emb_size*2, 1, dropout)
            if "c_pred_next" in self.config.aux_tasks:
                self.out_layer_c_next = OutputLayer(mlp_layer_num, emb_size*2, 1, dropout)
            if "c_pred_all" in self.config.aux_tasks:
                 self.out_layer_c_all = OutputLayer(mlp_layer_num, emb_size, config.num_c, dropout)
        else:
            print("Not share output layer")
            self.out_layer_q_next = nn.ModuleList([OutputLayer(mlp_layer_num, emb_size*2,1, dropout) for _ in self.config.dataconfig_list])
            if "c_pred_next" in self.config.aux_tasks:
                self.out_layer_c_next = nn.ModuleList([OutputLayer(mlp_layer_num, emb_size*2, 1, dropout)  for _ in self.config.dataconfig_list])
            if "c_pred_all" in self.config.aux_tasks:
                self.out_layer_c_all = nn.ModuleList([OutputLayer(mlp_layer_num, emb_size, config.num_c, dropout)  for _ in self.config.dataconfig_list])
            
        if "pred_c" in self.config.aux_tasks:
            self.concept_classifier = MultiLabelClassifier(self.config)
            
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def get_avg_fusion_concepts(self,y_concept,cshft):
        """获取知识点 fusion 的预测结果
        """
        max_num_concept = cshft.shape[-1]
        concept_mask = torch.where(cshft.long()==-1,False,True)
        concept_index = F.one_hot(torch.where(cshft!=-1,cshft,0),self.config.num_c)
        concept_sum = (y_concept.unsqueeze(2).repeat(1,1,max_num_concept,1)*concept_index).sum(-1)
        concept_sum = concept_sum*concept_mask#remove mask
        y_concept = concept_sum.sum(-1)/torch.where(concept_mask.sum(-1)!=0,concept_mask.sum(-1),1)
        return y_concept

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
    
    def get_attention(self, q_emb, q_shift, inter_emb, que_emb_index, concat_q = True):
        # forward the GPT model itself
        x = q_emb
        for block in self.transformer.h:
            x = block(x,x,inter_emb)
        x = self.transformer.ln_f(x)
        if concat_q:
            x = torch.cat([q_shift,x],dim=-1)
        return x

    def get_output(self, x, que_emb_index,output_type='q_pred_next',data=None):
        if output_type == 'q_pred_next':
            if self.config.share_output:
                logits = self.out_layer_q_next(x).squeeze(-1)
            else:
                logits = self.out_layer_q_next[que_emb_index](x).squeeze(-1)
            y = torch.sigmoid(logits)
        elif output_type == 'c_pred_next':
            if self.config.share_output:
                logits = self.out_layer_c_next(x).squeeze(-1)
            else:
                logits = self.out_layer_c_next[que_emb_index](x).squeeze(-1)
            y = torch.sigmoid(logits)
        elif output_type == 'c_pred_all':
            if self.config.share_output:
                logits = self.out_layer_c_all(x).squeeze(-1)
            else:
                logits = self.out_layer_c_all[que_emb_index](x).squeeze(-1)
            # self.get_avg_fusion_concepts(y_concept_next,data['cshft'])
            y = torch.sigmoid(logits)
            # print(f"y shape is {y.shape}, logits shape is {logits.shape}, data['cshft'] shape {data['cshft'].shape}")
            y = self.get_avg_fusion_concepts(y,data['cshft'])
       
        else:
            raise NotImplementedError

        
        return logits,y


    def forward(self, q, c ,r,data=None):
        pos_emb = self.transformer.wpe(q) # position embeddings of shape (1, t, emb_size)

        # Get the embeddings
        que_emb_index = self.config.source_list.index(data['source'])
        if self.emb_type in ['qid']:
            raw_emb_qc = self.que_emb_list[que_emb_index](q,c)
        else:
            _,_,raw_emb_qc,raw_emb_q,raw_emb_c = self.que_emb_list[que_emb_index](q,c,r)#[bs,emb_size*4],[bs,emb_size*2],[bs,emb_size*1],[bs,emb_size*1]
            emb_c_all = (raw_emb_c + pos_emb)[:,:-1]
        q_emb_all = self.emb_pooling(raw_emb_qc) + pos_emb
        q_shift = q_emb_all[:,1:]
        q_emb = q_emb_all[:,:-1]
        r_emb = self.r_emb(r)[:,:-1]

        inter_emb = q_emb + r_emb

        h_qc = self.get_attention(q_emb,q_shift,inter_emb,que_emb_index)
        logits_qc,y_qc = self.get_output(h_qc,que_emb_index,output_type='q_pred_next',data=data)
    
        
        num_y = 1
        total_y = y_qc
        if "c_pred_next" in self.config.aux_tasks:
            h_c = self.get_attention(emb_c_all,q_shift,inter_emb,que_emb_index)
            logits_c,y_c = self.get_output(h_c,que_emb_index,output_type='c_pred_next',data=data)
            total_y = total_y + y_c
            num_y += 1
        
        if "c_pred_all" in self.config.aux_tasks:
            h_c = self.get_attention(emb_c_all,q_shift,inter_emb,que_emb_index,concat_q=False)
            logits_c_all,y_c_all = self.get_output(h_c,que_emb_index,output_type='c_pred_all',data=data)
            total_y = total_y + y_c_all
            num_y += 1

        y = total_y/num_y
        outputs = {"y":y,"logits":logits_qc}
        if "pred_c" in self.config.aux_tasks:
            y_pred_c,y_perd_c_loss = self.concept_classifier(h_qc,data['c'])
            outputs['y_perd_c_loss'] = y_perd_c_loss
            outputs['y_pred_c'] = y_pred_c
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
    emb_type: str = 'iekt'
    emb_path: str = ""
    pretrain_dim: int = 768
    device: str = 'cpu'
    seed: int = 0
    return_dict: bool = False
    share_output: bool = False
    model_name: str = "gpt_mt"
    source_weight_t: float = 1
    mlp_layer_num: int = 1
    dataconfig_list: list = None  # Added attribute
    source_list: list = None      # Added attribute
    aux_tasks: list = None 



class GPTMT(QueBaseModel):
    def __init__(self, num_q, num_c, emb_size,seq_len=200,dropout=0.1, emb_type='iekt', emb_path="", pretrain_dim=768,device='cpu',seed=0,n_head=8,n_layer=8,dataconfig_list=None,source_list=None,return_dict=False,source_weight_t=1,share_output=False,aux_tasks=[],mlp_layer_num=1,model_name="gpt_mt"):
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
                                share_output=share_output,
                                aux_tasks=aux_tasks,
                                mlp_layer_num = mlp_layer_num
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
        if "pred_c" in self.config.aux_tasks:
            loss += outputs['y_perd_c_loss']
        return outputs['y'],loss#y_question没用

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        data_new = self.batch_to_device(data,process=process)
        outputs = self.model(data_new['cq'].long(),data_new['cc'],data_new['cr'].long(),data=data_new)
        if return_details:
            return outputs,data_new
        else:
            return outputs['y']
        
    def evaluate_pred_c(self,y_list,data_list):
        all_label_index = np.arange(1, self.config.num_c+1, 1)
        correct_num = 0
        error_num = 0
        for y,data in zip(y_list,data_list):
            for b_i in range(y.shape[0]):
                for s_i in range(y.shape[1]):
                    pred_i = y[b_i, s_i].detach().cpu().numpy()
                    label_i = data['c'][b_i, s_i].detach().cpu().numpy()
                    y_pred = all_label_index[pred_i > 0.5].tolist()
                    y_label = [x for x in label_i.tolist() if x != -1]
                    if y_pred == y_label:
                        correct_num += 1
                    else:
                        error_num += 1
        acc = correct_num/(correct_num+error_num)
        print(f"acc is {acc:.4f},correct_num={correct_num},error_num={error_num}")
        return acc
    
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
                
            model_result_data = {}
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

                # Get the best epoch
                source_best_epoch = all_auc_dict[source].index(max(all_auc_dict[source]))
                source_auc = all_auc_dict[source][source_best_epoch]
                source_acc = all_acc_dict[source][source_best_epoch]
                
                model_result_data['best_epoch_'+source] = source_best_epoch+1
                model_result_data['best_auc_'+source] = source_auc
                model_result_data['best_acc_'+source] = source_acc
                print(f"            {source}'s eval AUC is {auc:.4f}, ACC is {acc:.4f}, best epoch: {source_best_epoch+1}, best auc: {source_auc:.4}, best acc: {source_acc:.4}")
                # 
                # print(f"{source}'s eval_result is {eval_result}")
            weights = 1 - np.array(tmp_auc_list)
            weights = softmax(weights/self.config.source_weight_t)*len(tmp_auc_list)
            print(f"            weights is {weights}, source_weight_t is {self.config.source_weight_t}")

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
            
            if i - best_epoch >= patient:
                break
        if self.config.return_dict:
            model_result = {"best_epoch":best_epoch,"best_auc":max_auc,
                            "testauc":testauc,"testacc":testacc,"window_testauc":window_testauc,"window_testacc":window_testacc,"validauc":validauc,"validacc":validacc}
            model_result.update(model_result_data)
            return model_result
        else:
            return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch

    def evaluate(self,dataset,batch_size,acc_threshold=0.5,source=None,return_details=True):
        ps,ts,result_list,data_list = self.predict(dataset,batch_size=batch_size,source=source,return_details=return_details)
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= acc_threshold else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        eval_result = {"auc":auc,"acc":acc}
        
        if "pred_c" in self.config.aux_tasks:
            y_list = [x['y_pred_c'] for x in result_list]
            pred_c_acc = self.evaluate_pred_c(y_list, data_list)
            eval_result['pred_c_acc'] = pred_c_acc
        return eval_result
    
    def predict(self,dataset,batch_size,return_ts=False,process=True,source=None,return_details=False):
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
        self.model.eval()
        if return_details:
            result_list,data_list = [],[]
        with torch.no_grad():
            y_trues = []
            y_scores = []
            for data in test_loader:
                new_data = self.batch_to_device(data,process=process)
                data['source'] = source
                y = self.predict_one_step(data,return_details)
                if return_details:
                    outputs,data_new = y
                    y = outputs['y']
                    result_list.append(outputs)
                    data_list.append(new_data)
                y = torch.masked_select(y, new_data['sm']).detach().cpu()
                t = torch.masked_select(new_data['rshft'], new_data['sm']).detach().cpu()
                y_trues.append(t.numpy())
                y_scores.append(y.numpy())
               
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        if return_details:
            return ps,ts,result_list,data_list
        else:
            return ps,ts