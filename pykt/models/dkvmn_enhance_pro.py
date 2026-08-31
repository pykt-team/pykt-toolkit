import torch
from torch.nn import Dropout, Embedding, Linear, Module, Parameter
from torch.nn.init import kaiming_normal_

from .opera_utils import OperaQuestionEmbeddingMixin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DKVMN_Enhance_PRO(OperaQuestionEmbeddingMixin, Module):
    def __init__(self, num_q, dim_s, size_m, dropout=0.2, emb_type="qid", emb_path="", pretrain_dim=768, data_config=None):
        super().__init__()
        self.model_name = "dkvmn_enhance_pro"
        self.num_q = num_q
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type

        if data_config is None:
            raise ValueError(f"{self.model_name} requires data_config with pro_emb_path.")

        self.setup_question_embeddings(data_config, self.dim_s, self.num_q, device)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))
        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.ans_emb = Embedding(2, self.dim_s)
        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)
        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, q, r, qtest=False):
        batch_size = q.shape[0]
        k = self.get_question_embedding(q)
        v = k + self.ans_emb(r)

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
        Mv = [Mvt]
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))
        for et, at, wt in zip(e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)
        f = torch.tanh(self.f_layer(torch.cat([(w.unsqueeze(-1) * Mv[:, :-1]).sum(-2), k], dim=-1)))
        p = torch.sigmoid(self.p_layer(self.dropout_layer(f))).squeeze(-1)
        if not qtest:
            return p
        return p, f
