import torch
from torch.nn import Dropout, Embedding, LayerNorm, Linear, Module, MultiheadAttention

from .opera_utils import OperaQuestionEmbeddingMixin
from .utils import get_clones, pos_encode, transformer_FFN, ut_mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAKT_Enhance_PRO(OperaQuestionEmbeddingMixin, Module):
    def __init__(self, num_q, seq_len, emb_size, num_attn_heads, dropout, num_en=2, emb_type="qid", emb_path="", pretrain_dim=768, data_config=None):
        super().__init__()
        self.model_name = "sakt_enhance_pro"
        self.emb_type = emb_type
        self.num_q = num_q
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en

        if data_config is None:
            raise ValueError(f"{self.model_name} requires data_config with pro_emb_path.")

        self.setup_question_embeddings(data_config, self.emb_size, self.num_q, device)
        self.ans_emb = Embedding(2, self.emb_size)
        self.position_emb = Embedding(seq_len, emb_size)
        self.blocks = get_clones(Blocks(emb_size, num_attn_heads, dropout), self.num_en)
        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)

    def base_emb(self, q, r, qry):
        qshftemb = self.get_question_embedding(qry)
        xemb = self.get_question_embedding(q) + self.ans_emb(r)
        posemb = self.position_emb(pos_encode(xemb.shape[1]))
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, q, r, c, qry, cshift, qtest=False):
        qshftemb, xemb = self.base_emb(q, r, qry)
        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        if not qtest:
            return p
        return p, xemb


class Blocks(Module):
    def __init__(self, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()
        self.attn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)
        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        causal_mask = ut_mask(seq_len=k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)
        attn_emb = self.attn_layer_norm(q + attn_emb)
        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb
