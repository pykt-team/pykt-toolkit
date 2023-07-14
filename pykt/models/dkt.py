import os

import numpy as np
import torch
import pickle

from torch.nn import Module, Embedding, LSTM, Linear, Dropout


class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768, temp_filename="gradient.pkl"):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self._temp_filename = temp_filename

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)

    # def set_temp_filename(self, fname):
    #     self._temp_filename = fname   

    def backward_gradient_reporting_template(self, grad_input, filename):
        """
        Wrap this in a lambda function providing the filename when registering it as a backwards hook
        :param input:
        :param filename:
        :return:
        """
        tensors_to_cat = [grad_input[j].view(1, -1) for j in range(len(grad_input))]
        with open(filename, 'ab') as f:
            pickle.dump(torch.cat(tensors_to_cat, dim=0).cpu(), f)

    def forward(self, q, r):
        # print(f"q.shape is {q.shape}")
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        # print(f"xemb.shape is {xemb.shape}")
        h, _ = self.lstm_layer(xemb)
        self.h = h
        # print(f"h: {h.shape}")
        # h.register_hook(lambda grad: self.backward_gradient_reporting_template(grad, self._temp_filename))
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y