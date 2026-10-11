import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


def get_kc_embedding(last_kc, kc_emb, padding_idx=-1):
    batch_size, seq_len, max_concepts = last_kc.shape
    dim = kc_emb.size(1)
    mask = (last_kc != padding_idx)
    last_kc = last_kc.clamp(min=0)
    flat_kc = last_kc.view(-1)
    flat_emb = F.embedding(flat_kc, kc_emb)
    emb = flat_emb.view(batch_size, seq_len, max_concepts, dim)
    mask = mask.unsqueeze(-1).expand(-1, -1, -1, dim)
    masked_emb = emb * mask.float()
    concept_counts = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
    return (masked_emb.sum(dim=2) / concept_counts.squeeze(2)).squeeze(0)


class Enbedding_layer(nn.Module):
    def __init__(self, skill_max, pro_max, d, dataset_name):
        super().__init__()
        self.pro_embed = nn.Parameter(torch.ones((pro_max, d)))
        self.ans_embed = nn.Embedding(2, d)
        concept_map = torch.tensor(
            np.load(f"../data/{dataset_name}/question_concept_map.npy")
        ).unsqueeze(0)
        concept_embedding = nn.Embedding(skill_max, d)
        nn.init.xavier_uniform_(concept_embedding.weight)
        self.pro_embed = nn.Parameter(
            get_kc_embedding(concept_map, concept_embedding.weight) * self.pro_embed
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, last_pro, last_ans, next_pro):
        last_pro_embed = F.embedding(last_pro, self.pro_embed)
        next_pro_embed = F.embedding(next_pro, self.pro_embed)
        ans_embed = self.ans_embed(last_ans)
        return last_pro_embed + ans_embed, next_pro_embed


class Adaptive_fusion(nn.Module):
    def __init__(self, emb_size, dropout=0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(3 * emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size),
            nn.Sigmoid()
        )

    def forward(self, x_t, h_situational_adaptability, h_knowledge_accumulation):
        g_t = self.gate_net(torch.cat([x_t, h_situational_adaptability, h_knowledge_accumulation], dim=-1))
        h_fused = g_t * h_situational_adaptability + (1 - g_t) * h_knowledge_accumulation
        return h_fused, g_t


class CDFKT(nn.Module):
    """
    CDFKT: Cognitive Decouple-then-Fuse Knowledge Tracing.

    Decouples the student knowledge state into knowledge accumulation,
    captured by a Mamba-based sequence encoder, and situational
    adaptability, modeled by a sequence-step decay temporal encoder, then
    re-integrates the two signals with a feature-wise adaptive fusion gate.

    Requires the extra dependencies mamba-ssm and causal-conv1d.
    """

    def __init__(self,
                 num_q,
                 num_c,
                 emb_size,
                 dropout=0.1,
                 emb_type='qid',
                 base_intensity=0.1,
                 alpha_init_scale=0.01,
                 min_half_life=1.0,
                 max_half_life=64.0,
                 sa_layerscale_init=1.0,
                 sa_dropout=0.0,
                 sa_gate_bias=2.0,
                 fusion_dropout=0.1,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 device=None,
                 **kwargs):
        super().__init__()
        self.model_name = "cdfkt"
        self.num_c = num_c
        self.num_q = num_q
        self.emb_size = emb_size
        self.emb_type = emb_type
        self.dropout = dropout
        self.device = self._resolve_device(device)

        dataset_name = self._resolve_dataset_name()
        self.embedding_layer = Enbedding_layer(
            self.num_c,
            self.num_q,
            self.emb_size,
            dataset_name
        ).to(self.device)

        self.mu = nn.Parameter(
            torch.full((self.emb_size,), base_intensity, device=self.device)
        )
        alpha_init = torch.randn(self.emb_size, self.emb_size, device=self.device) * alpha_init_scale
        self.alpha = nn.Parameter(alpha_init)
        min_hl = max(float(min_half_life), 1e-3)
        max_hl = max(float(max_half_life), min_hl)
        half_lives = torch.logspace(
            math.log10(min_hl), math.log10(max_hl), self.emb_size, device=self.device
        )
        self.beta = nn.Parameter(math.log(2.0) / half_lives)

        self.sa_gate_proj = nn.Linear(self.emb_size, self.emb_size, device=self.device)
        nn.init.zeros_(self.sa_gate_proj.weight)
        nn.init.constant_(self.sa_gate_proj.bias, float(sa_gate_bias))
        self.sa_layernorm = nn.LayerNorm(self.emb_size, device=self.device)

        self.sa_layerscale = nn.Parameter(
            torch.full((self.emb_size,), float(sa_layerscale_init), device=self.device)
        )
        self.sa_drop = nn.Dropout(sa_dropout) if sa_dropout > 0 else nn.Identity()

        self.knowledge_accumulation_encoder = Mamba(
            d_model=self.emb_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        ).to(self.device)

        self.fusion_layer = Adaptive_fusion(self.emb_size, fusion_dropout).to(self.device)

        self.out_layer = nn.Sequential(
            nn.Linear(2 * self.emb_size, self.emb_size, device=self.device),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.emb_size, 1, device=self.device)
        ).to(self.device)

    def _resolve_device(self, device):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if device is None:
            return torch.device("cuda")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str) and device.startswith("cuda"):
            return torch.device(device)
        return torch.device(f"cuda:{device}")

    def _resolve_dataset_name(self):
        mapping = {
            "EdNet_large": "ednet5w",
            "NIPS34": "nips_task34",
            "XES3G5M": "peiyou",
            "AS09": "assist2009",
        }
        for key, dataset_name in mapping.items():
            if key in self.emb_type:
                return dataset_name
        return None

    def _excitation_input(self, xemb):
        contributions = torch.matmul(xemb, self.alpha.transpose(0, 1))  # [B, L, D]
        return contributions * torch.sigmoid(self.sa_gate_proj(xemb))

    def _causal_decay_conv(self, contributions, kernel):
        """T[t] = sum_{s<=t} kernel[t - s] * contributions[s], per channel."""
        _, seq_len, emb_size = contributions.shape
        x = contributions.transpose(1, 2)          # [B, D, L]
        x = F.pad(x, (seq_len - 1, 0))             # causal left padding only
        T_all = F.conv1d(x, kernel.flip(-1).unsqueeze(1), groups=emb_size)  # [B, D, L]
        return T_all.transpose(1, 2)

    def _decay_kernel(self, seq_len, dtype, device):
        time_indices = torch.arange(seq_len, device=device, dtype=dtype)
        beta = self.beta.clamp(min=1e-6)
        return torch.exp(-beta.unsqueeze(1) * time_indices.unsqueeze(0))  # [D, L]

    def situational_adaptability_encoder(self, xemb):
        seq_len = xemb.shape[1]
        contributions = self._excitation_input(xemb)
        kernel = self._decay_kernel(seq_len, xemb.dtype, xemb.device)
        T_all = self.sa_layernorm(self._causal_decay_conv(contributions, kernel))
        return self.sa_drop(T_all) * self.sa_layerscale

    def forward(self, last_pro, last_ans, next_pro, smask=None):
        last_pro = last_pro.to(self.device)
        last_ans = last_ans.to(self.device)
        next_pro = next_pro.to(self.device)
        xemb, next_xemb = self.embedding_layer(last_pro, last_ans, next_pro)
        xemb = xemb.to(self.device)
        next_xemb = next_xemb.to(self.device)

        T_all = self.situational_adaptability_encoder(xemb)
        h_situational_adaptability = xemb + self.mu.unsqueeze(0).unsqueeze(0) + T_all
        h_knowledge_accumulation = self.knowledge_accumulation_encoder(xemb)
        h_fused, _ = self.fusion_layer(xemb, h_situational_adaptability, h_knowledge_accumulation)
        return torch.sigmoid(
            self.out_layer(torch.cat([h_fused, next_xemb], dim=-1))
        ).squeeze(-1)
