import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout

class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', pretrained_emb_path="", emb_path=None):
        # emb_path is a dummy param to deal with the current data_config design. Will need to remove. Ignore for now
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

            
        if emb_type.startswith("qid"):
            if pretrained_emb_path.endswith('.pt'):
                emb_w = torch.load(pretrained_emb_path)
                self.emb_size = emb_w.shape[-1]
                self.interaction_emb = Embedding.from_pretrained(emb_w, freeze=True)
            else:
                self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
                
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        

    def forward(self, q, r):
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y