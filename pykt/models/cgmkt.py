import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from .QueGraphLearing import Questions_Embedding
from .SkillGraphLearning import Concepts_Embedding

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dataset_from_emb_type(emb_type):
    if emb_type.find("as09") != -1:
        return "assist2009"
    if emb_type.find("ni34") != -1:
        return "nips_task34"
    if emb_type.find("al05") != -1:
        return "algebra2005"
    if emb_type.find("bd06") != -1:
        return "bridge2algebra2006"
    if emb_type.find("py") != -1:
        return "peiyou"
    raise ValueError(f"Unsupported CGMKT emb_type: {emb_type}")


def _load_sparse_tensor(path):
    matrix = torch.load(path, weights_only=False).to(device)
    if not matrix.is_sparse:
        matrix = matrix.to_sparse()
    return matrix


def _to_dense(matrix):
    return matrix.to_dense() if matrix.is_sparse else matrix


def _as_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.tensor(obj.tolist())
    return torch.tensor(obj)


def _normalize_membership(raw, num_c, num_clusters):
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()

    if isinstance(raw, dict):
        for key in ["Q_hat", "q_hat", "membership", "memberships", "labels", "cluster_labels"]:
            if key in raw:
                raw = raw[key]
                break

    q_hat = _as_tensor(raw).float()
    if q_hat.dim() == 1:
        labels = q_hat.long()
        if labels.numel() != num_c:
            raise ValueError(
                f"CGMKT cluster label length mismatch: expected {num_c}, got {labels.numel()}"
            )
        if labels.min().item() < 0 or labels.max().item() >= num_clusters:
            raise ValueError(
                f"CGMKT cluster labels must be in [0, {num_clusters - 1}]"
            )
        q_hat = F.one_hot(labels, num_classes=num_clusters).float()
    elif q_hat.dim() == 2:
        if q_hat.shape == (num_clusters, num_c):
            q_hat = q_hat.t()
        if q_hat.shape != (num_c, num_clusters):
            raise ValueError(
                "CGMKT membership shape mismatch: expected "
                f"({num_c}, {num_clusters}), got {tuple(q_hat.shape)}"
            )
    else:
        raise ValueError(f"CGMKT membership tensor must be 1D or 2D, got {q_hat.dim()}D")

    q_hat = q_hat.clamp(min=0)
    row_sum = q_hat.sum(dim=-1, keepdim=True)
    if torch.any(row_sum <= 0):
        raise ValueError("CGMKT membership contains at least one all-zero KC row")
    return q_hat / row_sum


def _load_membership_tensor(data_dir, num_c, num_clusters):
    candidates = [
        f"kc_group_membership_{num_clusters}.pt",
        f"kc_group_membership_{num_clusters}.npy",
        f"kc_sbm_membership_{num_clusters}.pt",
        f"kc_sbm_membership_{num_clusters}.npy",
        f"kc_sbm_{num_clusters}_membership.pt",
        f"kc_sbm_{num_clusters}_membership.npy",
        f"q_hat_sbm_{num_clusters}.pt",
        f"q_hat_sbm_{num_clusters}.npy",
        f"sbm_q_hat_{num_clusters}.pt",
        f"sbm_q_hat_{num_clusters}.npy",
        f"kc_cluster_labels_{num_clusters}.pt",
        f"kc_cluster_labels_{num_clusters}.npy",
        f"kc_sbm_{num_clusters}_labels.pt",
        f"kc_sbm_{num_clusters}_labels.npy",
        f"kc_cluster_{num_clusters}.pt",
        f"kc_cluster_{num_clusters}.npy",
    ]

    checked_paths = []
    for name in candidates:
        path = os.path.join(data_dir, name)
        checked_paths.append(path)
        if not os.path.exists(path):
            continue
        if path.endswith(".npy"):
            raw = np.load(path, allow_pickle=True)
        else:
            raw = torch.load(path)
        return _normalize_membership(raw, num_c, num_clusters), path

    return None, checked_paths


def _simple_kmeans(x, num_clusters, num_iters=30):
    num_nodes = x.size(0)
    if num_nodes < num_clusters:
        raise ValueError(f"Cannot cluster {num_nodes} nodes into {num_clusters} groups")
    init_idx = torch.linspace(0, num_nodes - 1, steps=num_clusters, device=x.device).long()
    centers = x[init_idx].clone()
    labels = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
    for _ in range(num_iters):
        distances = torch.cdist(x, centers)
        labels = distances.argmin(dim=1)
        new_centers = centers.clone()
        for cluster_id in range(num_clusters):
            mask = labels == cluster_id
            if torch.any(mask):
                new_centers[cluster_id] = x[mask].mean(dim=0)
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


def _infer_membership_from_adjacency(matrix_kc, num_c, num_clusters):
    adj = _to_dense(matrix_kc).float().cpu()
    if adj.shape != (num_c, num_c):
        raise ValueError(
            "CGMKT fallback membership expects KC adjacency shape "
            f"({num_c}, {num_c}), got {tuple(adj.shape)}"
        )
    adj = (adj + adj.t()) / 2.0
    degree = adj.sum(dim=-1)
    inv_sqrt = torch.rsqrt(degree.clamp(min=1e-8))
    norm_adj = inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)
    _, eigvecs = torch.linalg.eigh(norm_adj)
    features = eigvecs[:, -num_clusters:]
    features = F.normalize(features, p=2, dim=-1)
    labels = _simple_kmeans(features, num_clusters)
    return F.one_hot(labels, num_classes=num_clusters).float()


def _row_normalize(matrix):
    row_sum = matrix.sum(dim=-1, keepdim=True)
    normalized = matrix / (row_sum + 1e-8)
    zero_rows = row_sum.squeeze(-1) <= 1e-8
    if torch.any(zero_rows) and matrix.size(0) > 1:
        fallback = torch.ones_like(matrix)
        fallback = fallback - torch.eye(matrix.size(0), device=matrix.device)
        fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp(min=1.0)
        normalized = torch.where(zero_rows.unsqueeze(-1), fallback, normalized)
    return normalized


def _transition_from_block(block):
    block = block.float().clamp(min=0)
    block = block - torch.diag_embed(torch.diagonal(block))
    return _row_normalize(block)


def _load_group_block_tensor(data_dir, num_clusters):
    candidates = [
        f"kc_group_block_{num_clusters}.pt",
        f"kc_group_block_{num_clusters}.npy",
        f"sbm_block_{num_clusters}.pt",
        f"sbm_block_{num_clusters}.npy",
        f"kc_sbm_block_{num_clusters}.pt",
        f"kc_sbm_block_{num_clusters}.npy",
        f"kc_group_B_{num_clusters}.pt",
        f"kc_group_B_{num_clusters}.npy",
    ]

    checked_paths = []
    for name in candidates:
        path = os.path.join(data_dir, name)
        checked_paths.append(path)
        if not os.path.exists(path):
            continue
        if path.endswith(".npy"):
            raw = np.load(path, allow_pickle=True)
        else:
            raw = torch.load(path)
        if isinstance(raw, np.ndarray) and raw.shape == ():
            raw = raw.item()

        if isinstance(raw, dict):
            for key in ["B", "block", "block_matrix", "group_block", "group_transition"]:
                if key in raw:
                    raw = raw[key]
                    break

        block = _as_tensor(raw).float()
        if block.shape != (num_clusters, num_clusters):
            raise ValueError(
                "CGMKT group block shape mismatch: expected "
                f"({num_clusters}, {num_clusters}), got {tuple(block.shape)}"
            )
        return block, path

    return None, checked_paths


def _aggregate_block_from_adjacency(matrix_kc, q_hat):
    adj = _to_dense(matrix_kc).float().cpu()
    q_hat_cpu = q_hat.float().cpu()
    block = q_hat_cpu.t().matmul(adj).matmul(q_hat_cpu)
    counts = q_hat_cpu.sum(dim=0).clamp(min=1.0)
    block = block / counts.unsqueeze(1) / counts.unsqueeze(0)
    return block


class CGMKT(Module):
    """
    CGMKT: Cognition-driven Dual-Graph Fusion with Group-level Mastery
    for Knowledge Tracing.

    The model keeps MoEKT's SBM-based dual graph embeddings, replaces the
    expert/router pool with one GRU, and adds a student-level group mastery
    state that is used before encoding and updated after observing r_t.
    """

    def __init__(
        self,
        num_q,
        num_c,
        emb_size,
        dropout=0.1,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        dropout_qk=0.2,
        dropout_kk=0.2,
        num_clusters=5,
        num_gcn_layers=1,
        mastery_update_hidden=64,
        mastery_step=0.1,
        modulation_type="gate",
        spread_type="sbm",
        spread_rate_init=0.1,
        spread_rate_max=1.0,
        mastery_bound=5.0,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "cgmkt"
        self.num_c = num_c
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.dropout = dropout
        self.num_clusters = num_clusters
        self.mastery_step = mastery_step
        self.mastery_bound = mastery_bound
        self.modulation_type = modulation_type
        self.spread_type = spread_type
        self.spread_rate_max = spread_rate_max

        dataset_name = _dataset_from_emb_type(emb_type)
        data_dir = os.path.join("..", "data", dataset_name)
        sbm_dir = os.path.join(data_dir, "sbm")

        self.matrix = _load_sparse_tensor(os.path.join(data_dir, "ques_skill_gcn_adj.pt"))
        self.matrix_kc = _load_sparse_tensor(
            os.path.join(sbm_dir, f"kc_kc_sbm_{num_clusters}_gcn_adj.pt")
        )
        kc_graph_source = f"sbm/kc_kc_sbm_{num_clusters}_gcn_adj.pt"

        q_hat, q_hat_source = _load_membership_tensor(sbm_dir, num_c, num_clusters)
        if q_hat is None:
            q_hat = _infer_membership_from_adjacency(self.matrix_kc, num_c, num_clusters)
            q_hat_source = f"inferred from {kc_graph_source}"
        q_hat = q_hat.to(device)
        self.register_buffer("Q_hat", q_hat)

        if spread_type == "sbm":
            block, block_source = _load_group_block_tensor(sbm_dir, num_clusters)
            if block is None:
                block = _aggregate_block_from_adjacency(self.matrix_kc, q_hat)
                block_source = f"aggregated from {kc_graph_source}"
            block = block.to(device)
            group_transition = _transition_from_block(block)
        elif spread_type == "random":
            block = torch.rand(num_clusters, num_clusters, device=device)
            group_transition = _transition_from_block(block)
            block_source = "random"
        elif spread_type == "hard":
            group_transition = torch.zeros(num_clusters, num_clusters, device=device)
            block_source = "hard"
        else:
            raise ValueError(f"Unknown spread_type: {spread_type}")
        self.register_buffer("group_transition", group_transition)
        if spread_rate_max <= 0:
            raise ValueError("spread_rate_max must be positive")
        beta_ratio = spread_rate_init / spread_rate_max
        beta_ratio = min(max(beta_ratio, 1e-6), 1.0 - 1e-6)
        beta_logit = torch.logit(torch.tensor(beta_ratio, dtype=torch.float32))
        self.raw_spread_rate = nn.Parameter(beta_logit.to(device))

        d = self.emb_size
        self.qe = Questions_Embedding(
            self.num_c, self.num_q, d, dropout_qk, dataset_name, num_gcn_layers
        ).to(device)
        self.ce = Concepts_Embedding(
            self.num_c, d, dropout_kk, dataset_name, num_gcn_layers
        ).to(device)

        if modulation_type == "gate":
            self.modulator = nn.Sequential(
                nn.Linear(num_clusters + d, d),
                nn.Sigmoid(),
            ).to(device)
        elif modulation_type == "film":
            self.modulator_gamma = nn.Linear(num_clusters, d).to(device)
            self.modulator_beta = nn.Linear(num_clusters, d).to(device)
        elif modulation_type in ("scalar", "none"):
            self.modulator = None
        else:
            raise ValueError(f"Unknown modulation_type: {modulation_type}")

        self.gru = nn.GRU(d, d, batch_first=True).to(device)

        self.update_mlp = nn.Sequential(
            nn.Linear(d + 1 + num_clusters, mastery_update_hidden),
            nn.ReLU(),
            nn.Linear(mastery_update_hidden, num_clusters),
        ).to(device)

        out_in = 2 * d + num_clusters
        self.out_layer = nn.Sequential(
            nn.Linear(out_in, d),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(d, 1),
        ).to(device)

        self._initialize_weights()

    def _initialize_weights(self):
        def init_linear(m):
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain("relu")
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.modulation_type == "gate":
            nn.init.zeros_(self.modulator[0].weight)
            nn.init.zeros_(self.modulator[0].bias)
        elif self.modulation_type == "film":
            nn.init.zeros_(self.modulator_gamma.weight)
            nn.init.zeros_(self.modulator_gamma.bias)
            nn.init.zeros_(self.modulator_beta.weight)
            nn.init.zeros_(self.modulator_beta.bias)
        self.update_mlp.apply(init_linear)
        self.out_layer.apply(init_linear)

    def _concept_group_membership(self, concept_ids):
        """
        Average SBM group memberships over possibly multi-concept exercises.

        concept_ids is [B, T, max_concepts] in the question-level loader and may
        be [B, T] in concept-level loaders. Padding concept id is -1.
        """
        if concept_ids.dim() == 2:
            concept_ids = concept_ids.unsqueeze(-1)
        valid = concept_ids >= 0
        safe_ids = concept_ids.clamp(min=0)
        memberships = F.embedding(safe_ids, self.Q_hat)
        memberships = memberships * valid.unsqueeze(-1).float()
        counts = valid.sum(dim=2, keepdim=True).clamp(min=1).float()
        return memberships.sum(dim=2) / counts

    def _modulate(self, x_t, m_prev, q_cur):
        if self.modulation_type == "none":
            return x_t
        m_q = m_prev * q_cur
        if self.modulation_type == "gate":
            gate = self.modulator(torch.cat([m_q, x_t], dim=-1))
            return x_t * (2.0 * gate)
        if self.modulation_type == "film":
            gamma = 1.0 + torch.tanh(self.modulator_gamma(m_q))
            beta = self.modulator_beta(m_q)
            return gamma * x_t + beta
        if self.modulation_type == "scalar":
            score = m_q.sum(dim=-1, keepdim=True)
            return (1.0 + torch.tanh(score)) * x_t
        raise ValueError(f"Unknown modulation_type: {self.modulation_type}")

    def _spread_rate(self):
        return self.spread_rate_max * torch.sigmoid(self.raw_spread_rate)

    def _spread_mask(self, q_cur):
        if self.spread_type == "hard":
            return q_cur
        beta = self._spread_rate()
        spread = q_cur + beta * torch.matmul(q_cur, self.group_transition)
        return spread / (spread.sum(dim=-1, keepdim=True) + 1e-8)

    def _update_mastery(self, m_prev, h_t, r_t, spread):
        context = m_prev * spread
        update_input = torch.cat([h_t, r_t.unsqueeze(-1).float(), context], dim=-1)
        delta = torch.tanh(self.update_mlp(update_input))
        m_new = m_prev + self.mastery_step * spread * delta
        return m_new.clamp(-self.mastery_bound, self.mastery_bound)

    def forward(self, last_pro, last_ans, last_skill, next_pro, next_skill):
        last_pro = last_pro.to(device)
        last_ans = last_ans.to(device)
        last_skill = last_skill.to(device)
        next_pro = next_pro.to(device)
        next_skill = next_skill.to(device)

        xemb_pro, next_xemb_pro = self.qe(last_pro, last_ans.long(), next_pro, self.matrix)
        xemb_kc, next_xemb_kc = self.ce(
            last_ans.long(), last_skill.long(), next_skill.long(), self.matrix_kc
        )

        x_t = xemb_pro + xemb_kc
        next_x = next_xemb_pro + next_xemb_kc

        q_cur = self._concept_group_membership(last_skill.long())
        q_next = self._concept_group_membership(next_skill.long())
        valid_steps = (last_skill >= 0).any(dim=-1) if last_skill.dim() == 3 else last_skill >= 0

        batch_size, seq_len, dim = x_t.shape
        m_prev = torch.zeros(batch_size, self.num_clusters, device=x_t.device)
        h_gru = torch.zeros(1, batch_size, dim, device=x_t.device)
        h_seq = []
        m_seq = []

        for t in range(seq_len):
            spread = self._spread_mask(q_cur[:, t, :])
            x_mod = self._modulate(x_t[:, t, :], m_prev, q_cur[:, t, :])
            h_out, h_gru = self.gru(x_mod.unsqueeze(1), h_gru)
            h_step = h_out.squeeze(1)
            m_new = self._update_mastery(m_prev, h_step, last_ans[:, t], spread)
            m_new = torch.where(valid_steps[:, t].unsqueeze(-1), m_new, m_prev)
            h_seq.append(h_step)
            m_seq.append(m_new)
            m_prev = m_new

        h_t = torch.stack(h_seq, dim=1)
        mastery_readout = torch.sigmoid(torch.stack(m_seq, dim=1))
        masked_mastery = mastery_readout * q_next
        y = torch.sigmoid(
            self.out_layer(torch.cat([h_t, next_x, masked_mastery], dim=-1))
        ).squeeze(-1)

        return y, torch.zeros((), device=y.device)
