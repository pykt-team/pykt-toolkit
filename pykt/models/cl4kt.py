import math
import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Embedding,
    Linear,
    Dropout,
    ModuleList,
    Sequential,
)
import torch.nn.functional as F
from torch.nn.modules.activation import GELU
from .cl4kt_modules import CL4KTTransformerLayer,CosinePositionalEmbedding

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class CL4KT(Module):
    def __init__(self, num_skills, num_questions, seq_len,emb_type,**kwargs):
        
        super(CL4KT, self).__init__()
        self.model_name = "cl4kt"
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.seq_len = seq_len
        self.emb_type = emb_type
        self.args = kwargs
        
        # Set hyperparameters.
        self.hidden_size = self.args["hidden_size"]
        self.num_blocks = self.args["num_blocks"]
        self.num_attn_heads = self.args["num_attn_heads"]
        self.kq_same = self.args["kq_same"]
        self.final_fc_dim = self.args["final_fc_dim"]
        self.final_fc_dim2 = self.args["final_fc_dim2"]
        self.d_ff = self.args["d_ff"]
        self.l2 = self.args.get("l2",1e-5)
        self.dropout = self.args["dropout"]
        self.reg_cl = self.args["reg_cl"]
        self.negative_prob = self.args["negative_prob"]
        self.hard_negative_weight = self.args["hard_negative_weight"]
        
        self.cl_emb_use_pos = self.args.get("cl_emb_use_pos",0) == 1

        # Define question and interaction embeddings.
        self.question_embed = Embedding(
            self.num_skills + 2, self.hidden_size, padding_idx=0
        )

        self.interaction_embed = Embedding(
            2 * (self.num_skills + 2), self.hidden_size, padding_idx=0
        )

    
        print(f"emb_type: {self.emb_type}")
        if self.emb_type in ["simplekt"]:
            if self.num_questions > 0:
                self.difficult_param = nn.Embedding(self.num_questions+2, self.hidden_size) # 
                self.q_embed_diff = nn.Embedding(self.num_skills+2, self.hidden_size) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed = nn.Embedding(2, self.hidden_size)
                
    
        # Define similarity measure and transformers.
        self.sim = Similarity(temp=self.args["temp"])

        self.question_encoder = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                    emb_type = self.emb_type
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.interaction_encoder = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                    emb_type = self.emb_type
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.knoweldge_retriever = ModuleList(
            [
                CL4KTTransformerLayer(
                    d_model=self.hidden_size,
                    d_feature=self.hidden_size // self.num_attn_heads,
                    d_ff=self.d_ff,
                    n_heads=self.num_attn_heads,
                    dropout=self.dropout,
                    kq_same=self.kq_same,
                    emb_type = self.emb_type
                )
                for _ in range(self.num_blocks)
            ]
        )

        # Define output layer.
        
        
        if self.emb_type in ["simplekt"]: 
            self.position_emb = CosinePositionalEmbedding(d_model=self.hidden_size, max_len=seq_len)
            self.out = nn.Sequential(
                nn.Linear(self.hidden_size + self.hidden_size, self.final_fc_dim), 
                nn.ReLU(), 
                nn.Dropout(self.dropout),
                nn.Linear(self.final_fc_dim, self.final_fc_dim2), 
                nn.ReLU(), 
                nn.Dropout(self.dropout),
                nn.Linear(self.final_fc_dim2, 1)
            )
        else:
            self.out = Sequential(
            Linear(2 * self.hidden_size, self.final_fc_dim),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim, self.final_fc_dim // 2),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim // 2, 1))
        # Define loss functions.
        self.cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.loss_fn = nn.BCELoss(reduction="mean")
        if self.emb_type in ["simplekt"]:
            self.reset()
        
    def reset(self):
        # This is a function for initializing parameters that sets all elements to a specified constant value (in this case, 0).
        for p in self.parameters():
            if p.size(0) == self.num_questions+2 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.)


    def base_emb(self,q,c,r):
        q_embed_data = self.question_embed(c)
        qa_embed_data = q_embed_data + self.qa_embed(r.long())
        
        if self.emb_type in ["simplekt"]:# add rasch        
            if self.num_questions > 0:
                q_embed_diff_data = self.q_embed_diff(c)  
                pid_embed_data = self.difficult_param(q)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data  
                
            # add position embedding
            q_pos_embed_data = q_embed_data + self.position_emb(q_embed_data)
            qa_pos_embed_data = qa_embed_data + self.position_emb(qa_embed_data)
        else:
            q_pos_embed_data = None
            qa_pos_embed_data = None
                
        return q_embed_data, qa_embed_data,q_pos_embed_data,qa_pos_embed_data
        
    def forward(self, batch):
        if self.training:
            q_i, q_j, q = batch["skills"]  # augmented q_i, augmented q_j and original q
            q_id_i, q_id_j, q_id = batch["questions"]  # augmented q_id_i, augmented q_id_j and original q_id
            r_i, r_j, r, neg_r = batch["responses"]  # augmented r_i, augmented r_j and original r
            attention_mask_i,attention_mask_j,attention_mask = batch["attention_mask"]  # augmented attention_mask_i, augmented attention_mask_j and original attention_mask
            
            if self.emb_type in ["simplekt"]:
                q_i_embed_data, qa_i_embed_data,q_i_pos_embed_data,qa_i_pos_embed_data = self.base_emb(q_id_i,q_i, r_i)
                q_j_embed_data, qa_j_embed_data,q_j_pos_embed_data,qa_j_pos_embed_data = self.base_emb(q_id_j,q_j, r_j)
            else:
                q_i_embed_data = self.question_embed(q_i)
                q_j_embed_data = self.question_embed(q_j)
                qa_i_embed_data = self.get_interaction_embed(q_i, r_i)
                qa_j_embed_data = self.get_interaction_embed(q_j, r_j)
                
                
            if self.negative_prob > 0:
                if self.emb_type in ["simplekt"]:
                    _,inter_k_embed,_,_ = self.base_emb(q_id, q, neg_r)
                else:
                    inter_k_embed = self.get_interaction_embed(q, neg_r)

            
            if self.emb_type in ["simplekt"] and self.cl_emb_use_pos:
                ques_i_score, ques_j_score = q_i_pos_embed_data, q_j_pos_embed_data 
                inter_i_score, inter_j_score = qa_i_pos_embed_data, qa_j_pos_embed_data
            else:
                ques_i_score, ques_j_score = q_i_embed_data, q_j_embed_data #[batch_size, seq_len, hidden_size]
                inter_i_score, inter_j_score = qa_i_embed_data, qa_j_embed_data

            # Apply transformers to question and interaction embeddings. Bidirectional attention.
            # mask=2 means bidirectional attention of BERT
            for block in self.question_encoder:
                ques_i_score, _ = block(
                    mask=2,
                    query=ques_i_score,
                    key=ques_i_score,
                    values=ques_i_score,
                    apply_pos=self.cl_emb_use_pos,
                )
                ques_j_score, _ = block(
                    mask=2,
                    query=ques_j_score,
                    key=ques_j_score,
                    values=ques_j_score,
                    apply_pos=self.cl_emb_use_pos,
                )

            for block in self.interaction_encoder:
                inter_i_score, _ = block(
                    mask=2,
                    query=inter_i_score,
                    key=inter_i_score,
                    values=inter_i_score,
                    apply_pos=self.cl_emb_use_pos,
                )
                inter_j_score, _ = block(
                    mask=2,
                    query=inter_j_score,
                    key=inter_j_score,
                    values=inter_j_score,
                    apply_pos=self.cl_emb_use_pos,
                )
                if self.negative_prob > 0:
                    inter_k_score, _ = block(
                        mask=2,
                        query=inter_k_embed,
                        key=inter_k_embed,
                        values=inter_k_embed,
                        apply_pos=self.cl_emb_use_pos,
                    )
            # Calculate pooled scores for question and interaction embeddings, pooling one sequence to one vector
            pooled_ques_i_score = (ques_i_score * attention_mask_i.unsqueeze(-1)).sum(
                1
            ) / attention_mask_i.sum(-1).unsqueeze(-1) #[batch_size, hidden_size]
            
            pooled_ques_j_score = (ques_j_score * attention_mask_j.unsqueeze(-1)).sum(
                1
            ) / attention_mask_j.sum(-1).unsqueeze(-1)

            # Calculate cosine similarity between pooled question embeddings.
            ques_cos_sim = self.sim(
                pooled_ques_i_score.unsqueeze(1), pooled_ques_j_score.unsqueeze(0)
            )
            
            # Calculate loss for cosine similarity between pooled question embeddings.
            ques_labels = torch.arange(ques_cos_sim.size(0)).long().to(q_i.device)
            # print(ques_cos_sim.size(), ques_labels.size())
            # print(ques_labels)
            question_cl_loss = self.cl_loss_fn(ques_cos_sim, ques_labels)
         
            # Calculate pooled scores for interaction embeddings.
            pooled_inter_i_score = (inter_i_score * attention_mask_i.unsqueeze(-1)).sum(
                1
            ) / attention_mask_i.sum(-1).unsqueeze(-1)
            pooled_inter_j_score = (inter_j_score * attention_mask_j.unsqueeze(-1)).sum(
                1
            ) / attention_mask_j.sum(-1).unsqueeze(-1)


            # Calculate cosine similarity between pooled interaction embeddings.
            inter_cos_sim = self.sim(
                pooled_inter_i_score.unsqueeze(1), pooled_inter_j_score.unsqueeze(0)
            )#[batch_size, batch_size]

            if self.negative_prob > 0:
                # Calculate pooled scores for hard negative interaction embeddings.
                pooled_inter_k_score = (
                    inter_k_score * attention_mask.unsqueeze(-1)
                ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                
                # Calculate cosine similarity between pooled interaction embeddings and hard negative
                neg_inter_cos_sim = self.sim(
                    pooled_inter_i_score.unsqueeze(1), pooled_inter_k_score.unsqueeze(0)
                )
                inter_cos_sim = torch.cat([inter_cos_sim, neg_inter_cos_sim], 1)#[batch_size, batch_size*2]
            # print(f"inter_cos_sim is {inter_cos_sim.shape}")
            inter_labels = torch.arange(inter_cos_sim.size(0)).long().to(q_i.device)# [batch_size]
            # print(f"inter_labels is {inter_labels.shape}")
            if self.negative_prob > 0:
                weights = torch.tensor(
                    [
                        [0.0] * (inter_cos_sim.size(-1) - neg_inter_cos_sim.size(-1))#batch_size
                        + [0.0] * i
                        + [self.hard_negative_weight]
                        + [0.0] * (neg_inter_cos_sim.size(-1) - i - 1)
                        for i in range(neg_inter_cos_sim.size(-1))
                    ]
                ).to(q_i.device)# [batch_size, batch_size*2]
                # print(f"weights are {weights.shape}")
                inter_cos_sim = inter_cos_sim + weights
            """ 
            tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.0000, 0.0000, 0.0000],
                    #[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                    #[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.0000],
                    #[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000]])
            """
            interaction_cl_loss = self.cl_loss_fn(inter_cos_sim, inter_labels)
        else:
   
            q = batch["skills"]  # augmented q_i, augmented q_j and original q
            q_id = batch["questions"]
            r = batch["responses"]  # augmented r_i, augmented r_j and original r
            attention_mask = batch["attention_mask"]
   
        # print(f"q is {q.shape}, q_id is {q_id.shape}, r is {r.shape}, attention_mask is {attention_mask.shape}")
        # Generate final prediction
        if self.emb_type in ["simplekt"]:
            q_embed_data, qa_embed_data,q_pos_embed_data,qa_pos_embed_data = self.base_emb(q_id,q,r)
            x, y = q_pos_embed_data, qa_pos_embed_data

        else:
            q_embed_data = self.question_embed(q)
            qa_embed_data = self.get_interaction_embed(q, r)
            x, y = q_embed_data, qa_embed_data
            

        # if self.emb_type not in ["simplekt"]:
        for block in self.question_encoder:
            x, _ = block(mask=1, query=x, key=x, values=x, apply_pos=True)

        for block in self.interaction_encoder:
            y, _ = block(mask=1, query=y, key=y, values=y, apply_pos=True)

        for block in self.knoweldge_retriever:
            x, attn = block(mask=0, query=x, key=x, values=y, apply_pos=True)

        retrieved_knowledge = torch.cat([x, q_embed_data], dim=-1)

        output = torch.sigmoid(self.out(retrieved_knowledge)).squeeze(-1)
        if self.training:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "cl_loss": question_cl_loss + interaction_cl_loss,
                "attn": attn,
            }
            # print(f"out_dict is {out_dict['cl_loss']}")
        else:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "attn": attn,
                "x": x,
            }

        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        cl_loss = torch.mean(out_dict["cl_loss"])  # torch.mean() for multi-gpu FIXME
        mask = true > -1

        loss = self.loss_fn(pred[mask], true[mask]) + self.reg_cl * cl_loss

        return loss, len(pred[mask]), true[mask].sum().item()

    def get_interaction_embed(self, skills, responses):
        
        masked_responses = responses * (responses > -1).long()
        interactions = (skills + self.num_skills * masked_responses).long()
        # print(f"interactions are {interactions},{interactions.dtype}")
        return self.interaction_embed(interactions)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp