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
from .cl4kt_modules import CL4KTTransformerLayer

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
        self.d_ff = self.args["d_ff"]
        self.l2 = self.args.get("l2",0)
        self.dropout = self.args["dropout"]
        self.reg_cl = self.args["reg_cl"]
        self.negative_prob = self.args["negative_prob"]
        self.hard_negative_weight = self.args["hard_negative_weight"]

        # Define question and interaction embeddings.
        self.question_embed = Embedding(
            self.num_skills + 2, self.hidden_size, padding_idx=0
        )
        self.interaction_embed = Embedding(
            2 * (self.num_skills + 2), self.hidden_size, padding_idx=0
        )
        
        
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
                )
                for _ in range(self.num_blocks)
            ]
        )

        # Define output layer.
        self.out = Sequential(
            Linear(2 * self.hidden_size, self.final_fc_dim),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim, self.final_fc_dim // 2),
            GELU(),
            Dropout(self.dropout),
            Linear(self.final_fc_dim // 2, 1),
        )
        
        # Define loss functions.
        self.cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.loss_fn = nn.BCELoss(reduction="mean")

    def forward(self, batch):
        if self.training:
            q_i, q_j, q = batch["skills"]  # augmented q_i, augmented q_j and original q
            r_i, r_j, r, neg_r = batch["responses"]  # augmented r_i, augmented r_j and original r
            attention_mask_i, attention_mask_j, attention_mask = batch["attention_mask"]

            ques_i_embed = self.question_embed(q_i)
            ques_j_embed = self.question_embed(q_j)
            inter_i_embed = self.get_interaction_embed(q_i, r_i)
            inter_j_embed = self.get_interaction_embed(q_j, r_j)
            if self.negative_prob > 0:
                # inter_k_embed = self.get_negative_interaction_embed(q, r) # hard negative
                inter_k_embed = self.get_interaction_embed(q, neg_r)

            # mask=2 means bidirectional attention of BERT
            ques_i_score, ques_j_score = ques_i_embed, ques_j_embed
            inter_i_score, inter_j_score = inter_i_embed, inter_j_embed

            # Apply transformers to question and interaction embeddings. Bidirectional attention.
            for block in self.question_encoder:
                ques_i_score, _ = block(
                    mask=2,
                    query=ques_i_score,
                    key=ques_i_score,
                    values=ques_i_score,
                    apply_pos=False,
                )
                ques_j_score, _ = block(
                    mask=2,
                    query=ques_j_score,
                    key=ques_j_score,
                    values=ques_j_score,
                    apply_pos=False,
                )

            for block in self.interaction_encoder:
                inter_i_score, _ = block(
                    mask=2,
                    query=inter_i_score,
                    key=inter_i_score,
                    values=inter_i_score,
                    apply_pos=False,
                )
                inter_j_score, _ = block(
                    mask=2,
                    query=inter_j_score,
                    key=inter_j_score,
                    values=inter_j_score,
                    apply_pos=False,
                )
                if self.negative_prob > 0:
                    inter_k_score, _ = block(
                        mask=2,
                        query=inter_k_embed,
                        key=inter_k_embed,
                        values=inter_k_embed,
                        apply_pos=False,
                    )

            # Calculate pooled scores for question and interaction embeddings.
            pooled_ques_i_score = (ques_i_score * attention_mask_i.unsqueeze(-1)).sum(
                1
            ) / attention_mask_i.sum(-1).unsqueeze(-1)
            pooled_ques_j_score = (ques_j_score * attention_mask_j.unsqueeze(-1)).sum(
                1
            ) / attention_mask_j.sum(-1).unsqueeze(-1)

            # Calculate cosine similarity between pooled question embeddings.
            ques_cos_sim = self.sim(
                pooled_ques_i_score.unsqueeze(1), pooled_ques_j_score.unsqueeze(0)
            )
            
            # Calculate loss for cosine similarity between pooled question embeddings.
            ques_labels = torch.arange(ques_cos_sim.size(0)).long().to(q_i.device)
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
            )

            if self.negative_prob > 0:
                # Calculate pooled scores for hard negative interaction embeddings.
                pooled_inter_k_score = (
                    inter_k_score * attention_mask.unsqueeze(-1)
                ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                
                # Calculate cosine similarity between pooled interaction embeddings and hard negative
                neg_inter_cos_sim = self.sim(
                    pooled_inter_i_score.unsqueeze(1), pooled_inter_k_score.unsqueeze(0)
                )
                inter_cos_sim = torch.cat([inter_cos_sim, neg_inter_cos_sim], 1)

            inter_labels = torch.arange(inter_cos_sim.size(0)).long().to(q_i.device)

            if self.negative_prob > 0:
                weights = torch.tensor(
                    [
                        [0.0] * (inter_cos_sim.size(-1) - neg_inter_cos_sim.size(-1))
                        + [0.0] * i
                        + [self.hard_negative_weight]
                        + [0.0] * (neg_inter_cos_sim.size(-1) - i - 1)
                        for i in range(neg_inter_cos_sim.size(-1))
                    ]
                ).to(q_i.device)
                inter_cos_sim = inter_cos_sim + weights

            interaction_cl_loss = self.cl_loss_fn(inter_cos_sim, inter_labels)
        else:
            q = batch["skills"]  # augmented q_i, augmented q_j and original q
            r = batch["responses"]  # augmented r_i, augmented r_j and original r

            attention_mask = batch["attention_mask"]

        q_embed = self.question_embed(q)
        i_embed = self.get_interaction_embed(q, r)

        x, y = q_embed, i_embed
        for block in self.question_encoder:
            x, _ = block(mask=1, query=x, key=x, values=x, apply_pos=True)

        for block in self.interaction_encoder:
            y, _ = block(mask=1, query=y, key=y, values=y, apply_pos=True)

        for block in self.knoweldge_retriever:
            x, attn = block(mask=0, query=x, key=x, values=y, apply_pos=True)

        retrieved_knowledge = torch.cat([x, q_embed], dim=-1)

        output = torch.sigmoid(self.out(retrieved_knowledge)).squeeze()

        if self.training:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "cl_loss": question_cl_loss + interaction_cl_loss,
                "attn": attn,
            }
        else:
            out_dict = {
                "pred": output[:, 1:],
                "true": r[:, 1:].float(),
                "attn": attn,
                "x": x,
            }

        return out_dict

    def alignment_and_uniformity(self, out_dict):
        return (
            out_dict["question_alignment"],
            out_dict["interaction_alignment"],
            out_dict["question_uniformity"],
            out_dict["interaction_uniformity"],
        )

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


# https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], ("unrecognized pooling type %s" % self.pooler_type)

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


# ref: https://github.com/SsnL/align_uniform
def align_loss(x, y, alpha=2):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    x = F.normalize(x, dim=1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
