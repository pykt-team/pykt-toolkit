import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from .pretrain_utils import RobertaEncode  

device = "cpu" if not torch.cuda.is_available() else "cuda"  


class GateEarlyFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model, 1)

    def forward(self, fusion_infs):
        newinfs = []
        for inf in fusion_infs[0:]:
            curinf = inf.unsqueeze(2)
            newinfs.append(curinf)
        res = torch.cat(newinfs, dim=2)
        # bz * seqlen * m * h
        weights = torch.sigmoid(self.gate(res)).transpose(-2, -1)
        # print(f"res: {res.shape}, weights: {weights.shape}")
        res = torch.matmul(weights, res).squeeze(-2)
        # print(f"newemb: {newemb.shape}, fusion0: {fusion_infs[0].shape}")
        # assert False
        return res

class QuestionEncoder(Module):
    def __init__(self, num_q, emb_type, emb_size, dropout, emb_paths, pretrain_dim=768) -> None:
        super().__init__()
        
        self.emb_type = emb_type
        
        self.id_encoder = Embedding(num_q, emb_size)
        self.content_encoder = RobertaEncode(emb_size, dropout, emb_paths['que_embs'][0], emb_paths['que_embs'][1])
        self.analysis_encoder = RobertaEncode(emb_size, dropout, emb_paths['ana_embs'][0], emb_paths['ana_embs'][1])
        self.type_encoder = Embedding(2, emb_size)
        
    def forward(self, qs, types):
        qid_emb = self.id_encoder(qs)
        cont_emb = self.content_encoder(qs)
        ana_emb = self.analysis_encoder(qs)
        type_emb = self.type_encoder(types)

        return [qid_emb, cont_emb, ana_emb, type_emb]

    
class KCRouteEncoder(Module):
    def __init__(self, num_level, num_c, num_croutes, emb_type, emb_size, dropout, 
            emb_path, pretrain_dim=768, model_name="akt_peiyou") -> None:
        super().__init__()
        self.num_c = num_c
        self.num_croutes = num_croutes
        self.emb_size = emb_size
        self.num_level = num_level
        self.emb_type = emb_type
        self.model_name = model_name

        print(f"emb_path: {emb_path}, num_c: {num_c}, num_croutes: {num_croutes}")
        
        if self.emb_type.find("rc") != -1:
            if self.emb_type.find("rcadd") != -1:
                self.rc_content_emb = RobertaEncode(emb_size, dropout, emb_path, pretrain_dim, 2)
            self.rc_cid_emb = nn.Parameter(torch.randn(self.num_croutes, self.emb_size).to(device), requires_grad=True)#concept embeding
            self.rc_weight = nn.Parameter(torch.randn(num_level).to(device), requires_grad=True)
        # elif self.emb_type.endswith("onlycid"): # don't equal to akt peiyou
        #     self.cid_emb = nn.Embedding(self.num_c, self.emb_size)
        if self.emb_type.find("tail") != -1:
            self.tailc_content_emb = RobertaEncode(emb_size, dropout, emb_path, pretrain_dim, 0)
            self.tailc_cid_emb = nn.Embedding(self.num_c+1, self.emb_size) # +1是因为有一个-1
        
    def forward(self, croutes, tailcs): # cs: batch_size, sequence_len, num_level
        # if self.emb_type.endswith("onlycid"):
        #     return self.cid_emb(tailcs)
        # print(f"croutes: {croutes.shape}, tailcs: {tailcs.shape}")
        # print(f"model_name: {self.model_name}, emb_type: {self.emb_type}")
        ## tali c
        if self.emb_type.find("tail") != -1:
            tailc_cid_emb = self.tailc_cid_emb(tailcs)
            tailc_cont_emb = self.tailc_content_emb(tailcs)
        
        if self.emb_type.find("rc") != -1:
            # add zero for padding -1
            ## rc route
            concept_emb_cat = torch.cat([torch.zeros(2, self.emb_size).to(device), self.rc_cid_emb], dim=0)
            # shift c
            related_concepts = (croutes+2).long() # 0->-2->kc route pad, 1->-1->sequence pad
            # print(f"concept_emb_cat: {concept_emb_cat.shape}")
            # print(f"related_concepts: {related_concepts}")
            rc_cid_emb = concept_emb_cat[related_concepts, :]
            if self.emb_type.find("rcadd") != -1:
                rc_cont_emb = self.rc_content_emb(croutes+2) ###
            # assert False
        
            # weights
            indexs = torch.from_numpy(np.arange(0, self.num_level)).unsqueeze(0).expand(related_concepts.shape[0], related_concepts.shape[1], self.num_level).to(device)
            is_avail = torch.where(related_concepts != 0, 1, 0)
            indexs = torch.where(is_avail == 1, indexs, -1)      
            new_weights = torch.where(is_avail > 0, self.rc_weight[indexs], torch.tensor(float("-inf"), dtype=torch.float32).to(device)) 
            # print(new_weights.shape)
            # assert False
            alphas = torch.softmax(new_weights, dim=-1).unsqueeze(-2)
            # alphas = torch.tensor(alphas, dtype=torch.float32)
            # print(alphas.shape, cemb.shape)
            rcid_emb = torch.matmul(alphas, rc_cid_emb).squeeze(-2)
            if self.emb_type.find("rcadd") != -1:
                rccont_emb = torch.matmul(alphas, rc_cont_emb).squeeze(-2)
        
        if self.emb_type.find("tail") != -1 and self.emb_type.find("rcadd") != -1: # all
            return [tailc_cid_emb, tailc_cont_emb, rcid_emb, rccont_emb]
        elif self.emb_type.find("tail") != -1: # tail id & cont
            return [tailc_cid_emb, tailc_cont_emb]
        elif self.emb_type.find("rcadd") != -1: # rc id & cont
            return [rcid_emb, rccont_emb]
        elif self.emb_type.find("rcid") != -1: # only rc id
            return  [rcid_emb]

    def get_avg_skill_emb(self, c, concept_emb):
        # add zero for padding
        # concept_emb_cat = torch.cat(
        #     [torch.zeros(1, self.emb_size).to(self.device), 
        #     concept_emb], dim=0)
        # # shift c

        related_concepts = (c+1).long()
        # #[batch_size, seq_len, emb_dim]
        concept_emb_sum = concept_emb.sum(axis=-2)
        # concept_emb_sum = concept_emb_cat[related_concepts, :].sum(
        #     axis=-2)

        #[batch_size, seq_len,1]
        concept_num = torch.where(related_concepts != 0, 1, 0).sum(
            axis=-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg
    
def mean_max_mul(embs):
    onedim = []
    for emb in embs:
        onedim.append(emb.reshape(-1).unsqueeze(0))
    merge = torch.cat(onedim, dim=0)
    
    mean = torch.mean(merge, dim=0).reshape_as(embs[0])
    max = torch.max(merge, dim=0)[0].reshape_as(embs[0])
    mul = torch.mul(embs[0], embs[1])
    for k in range(2, len(embs)):
        mul = torch.mul(mul, embs[k])
    res = torch.cat([mean, max, mul], dim=-1)
    return res

class QueEmbPeiyou(nn.Module):
    def __init__(self, num_q, num_croutes, num_c, emb_size, dropout, emb_type='qid', 
            num_level=10, emb_paths={}, pretrain_dim=768):
        super().__init__()
        self.num_q = num_q
        self.num_c = num_c if emb_type.find("rc") == -1 else num_croutes
        self.emb_size = emb_size
        self.num_level = num_level
        #get emb type
        # tmp_emb_type = f"{model_name}-{emb_type}"
        # emb_type = emb_type_map.get(tmp_emb_type,tmp_emb_type.replace(f"{model_name}-",""))
        print(f"emb_type is {emb_type}, self.num_c: {self.num_c}")

        self.emb_type = emb_type
        self.emb_paths = emb_paths
        self.pretrain_dim = pretrain_dim

        ## iekt: qc_merge
        self.concept_emb = KCRouteEncoder(num_level, self.num_c, emb_type, emb_size, dropout, 
                emb_paths["kc_embs"][0], emb_paths["kc_embs"][1], model_name="iekt_peiyou")
        self.que_emb = QuestionEncoder(num_q, emb_type, emb_size, dropout, emb_paths, pretrain_dim)

        self.que_c_linear = nn.Linear(2*self.emb_size,self.emb_size)

        self.output_emb_dim = emb_size

    def forward(self, q, qtypes, qcroutes, c, r=None):
        emb_type = self.emb_type
        ## iekt: qc_merge
        
        croutes = qcroutes.reshape(qcroutes.shape[0], -1, self.num_level)
        # print(f"qcroutes: {qcroutes.shape}, croutes: {croutes.shape}")
        if self.emb_type.find("tail") == -1:
            cembs = self.concept_emb(croutes, c).reshape(qcroutes.shape[0], qcroutes.shape[1], qcroutes.shape[2], -1)#[batch,max_len-1,emb_size]
            concept_avg = self.get_avg_skill_emb(qcroutes, cembs)
            # print(f"concept_avg: {concept_avg.shape}")
        else:
            concept_avg = self.concept_emb(croutes, c)
        que_emb = self.que_emb(q, qtypes)#[batch,max_len-1,emb_size]
        # print(f"que_emb shape is {que_emb.shape}")
        # print(f"concept_avg is {concept_avg.shape}")
        que_c_emb = torch.cat([concept_avg, que_emb],dim=-1)#[batch,max_len-1,2*emb_size]
    
        # print("qc_merge")
        xemb = que_c_emb.squeeze(1)

        return xemb

    def get_avg_skill_emb(self, qcroutes, cembs):
        qc_sum = cembs.sum(axis=-2)
        is_avail = torch.where(qcroutes.sum(axis=-1) != self.num_level, 1, 0)
        qc_num = torch.where(is_avail != 0, 1, 0).sum(axis=-1).unsqueeze(-1)
        qc_num = torch.where(qc_num == 0, 1, qc_num)
        concept_avg = qc_sum / qc_num
        return concept_avg

from .que_base_model import QueBaseModel
class QueBaseModelPeiyou(QueBaseModel):
    def __init__(self, model_name, emb_type, emb_path, pretrain_dim):
        super().__init__(model_name, emb_type, emb_path, pretrain_dim, device)

    def batch_to_device(self,data,process=True):
        if not process:
            return data
        dcur = data
        # q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]
        # qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
        # m, sm = dcur["masks"], dcur["smasks"]
        data_new = {}
        data_new['cq'] = torch.cat((dcur["qseqs"][:,0:1], dcur["shft_qseqs"]), dim=1)
        data_new['cqtypes'] = torch.cat((dcur["qtypes"][:,0:1], dcur["shft_qtypes"]), dim=1)
        data_new['cc'] = torch.cat((dcur["cseqs"][:,0:1],  dcur["shft_cseqs"]), dim=1)
        data_new['cqcroutes'] = torch.cat((dcur["qcroutes"][:,0:1,:], dcur["shft_qcroutes"]), dim=1)
        # print(f"cqcrooutes: {data_new['cqcroutes'].shape}, cc: {data_new['cc'].shape}")
        # assert False
        data_new['cr'] = torch.cat((dcur["rseqs"][:,0:1], dcur["shft_rseqs"]), dim=1)
        data_new['ct'] = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        data_new['q'] = dcur["qseqs"]
        data_new['qtypes'] = dcur["qtypes"]
        data_new['c'] = dcur["cseqs"]
        data_new['qcroutes'] = dcur["qcroutes"]
        data_new['r'] = dcur["rseqs"]
        data_new['t'] = dcur["tseqs"]
        data_new['qshft'] = dcur["shft_qseqs"]
        data_new['qtypesshft'] = dcur["shft_qtypes"]
        data_new['cshft'] = dcur["shft_cseqs"]
        data_new['qcroutesshft'] = dcur["shft_qcroutes"]
        data_new['rshft'] = dcur["shft_rseqs"]
        data_new['tshft'] = dcur["shft_tseqs"]
        data_new['m'] = dcur["masks"]
        data_new['sm'] = dcur["smasks"]
        return data_new