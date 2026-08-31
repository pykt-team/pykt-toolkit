import torch
from torch.nn import Dropout, Embedding, LSTM, Linear, Module

from .opera_utils import OperaQuestionEmbeddingMixin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DKT_Enhance_Pro(OperaQuestionEmbeddingMixin, Module):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, emb_type="qid", emb_path="", pretrain_dim=768, data_config=None):
        super().__init__()
        self.model_name = "dkt_enhance_pro"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.dropout = dropout

        if data_config is None:
            raise ValueError(f"{self.model_name} requires data_config with pro_emb_path.")

        self.setup_question_embeddings(data_config, self.emb_size, self.num_q, device)
        self.ans_emb = Embedding(2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = torch.nn.Sequential(
            Linear(2 * self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            Dropout(p=self.dropout),
            Linear(self.hidden_size, 1),
        )

    def forward(self, last_pro, last_ans, last_skill, next_pro, next_skill, perb=None):
        last_pro_embedding = self.get_question_embedding(last_pro)
        last_ans_embedding = self.ans_emb(last_ans)
        next_pro_embedding = self.get_question_embedding(next_pro)

        xemb = last_pro_embedding + last_ans_embedding
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = torch.sigmoid(self.out_layer(torch.cat([h, next_pro_embedding], dim=-1))).squeeze(-1)
        return y
