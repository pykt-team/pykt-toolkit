import torch
import torch.nn as nn
from .stosa_util import Encoder, LayerNorm, DistSAEncoder, DistMeanSAEncoder, wasserstein_distance
import numpy as np
from random import choice

class StosaKT(nn.Module):
    def __init__(self, n_question, args ,emb_type="qid", emb_path=""):
        super(StosaKT, self).__init__()

        self.model_name = "stosakt"
        self.emb_type = emb_type
        self.item_mean_embeddings = nn.Embedding(n_question, args.hidden_size, padding_idx=0)
        self.item_response_mean_embeddings = nn.Embedding(n_question*2+1, args.hidden_size, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(n_question, args.hidden_size, padding_idx=0)
        self.item_response_cov_embeddings = nn.Embedding(n_question*2+1, args.hidden_size, padding_idx=0)

        self.position_mean_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.position_response_mean_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.position_cov_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.position_response_cov_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        # self.user_margins = nn.Embedding(args.num_users, 1)
        self.out = nn.Linear(1,1)
        self.out_neg = nn.Linear(1,1)
        self.item_encoder = DistSAEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.n_question = n_question
        self.apply(self.init_weights)


    def add_position_mean_embedding(self, sequence, target):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_mean_embeddings(position_ids)
        
        item_embeddings = self.item_mean_embeddings(sequence)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(sequence_emb)

        qa_data = sequence + self.n_question * target
        item_response_embeddings = self.item_response_mean_embeddings(qa_data)
        position_response_embeddings = self.position_response_mean_embeddings(position_ids)
        sequence_response_emb = item_response_embeddings + position_response_embeddings
        sequence_response_emb = self.LayerNorm(sequence_response_emb)
        sequence_response_emb = self.dropout(sequence_response_emb)
        elu_act = torch.nn.ELU()
        sequence_response_emb = elu_act(sequence_response_emb)

        return sequence_emb, sequence_response_emb

    def add_position_cov_embedding(self, sequence, target):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_cov_embeddings(position_ids)

        item_embeddings = self.item_cov_embeddings(sequence)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(self.dropout(sequence_emb)) + 1

        qa_data = sequence + self.n_question * target
        item_response_embeddings = self.item_response_cov_embeddings(sequence)
        position_response_embeddings = self.position_response_cov_embeddings(position_ids)
        sequence_response_emb = item_response_embeddings + position_response_embeddings
        sequence_response_emb = self.LayerNorm(sequence_response_emb)
        elu_act = torch.nn.ELU()
        sequence_response_emb = elu_act(self.dropout(sequence_response_emb)) + 1

        return sequence_emb, sequence_response_emb

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        input_ids = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)

        attention_mask = (input_ids >= 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=0) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().cuda()

        # if self.args.cuda_condition:
        #     subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * (-2 ** 32 + 1)
        # print(f"extended_attention_mask:{extended_attention_mask}")

        mean_sequence_emb, mean_sequence_response_emb = self.add_position_mean_embedding(input_ids, target)
        cov_sequence_emb, cov_sequence_response_emb = self.add_position_cov_embedding(input_ids, target)

        item_encoded_layers = self.item_encoder(mean_sequence_emb,
                                                mean_sequence_response_emb,
                                                cov_sequence_emb,
                                                cov_sequence_response_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        mean_sequence_output, cov_sequence_output, att_scores = item_encoded_layers[-1]

        pvn_loss = 0
        if self.emb_type.find("qid") != -1:
            pos_logits = wasserstein_distance(mean_sequence_output, cov_sequence_output, mean_sequence_emb, cov_sequence_emb).unsqueeze(2)
            # print(f"pos_logits:{pos_logits.shape}")
            # print(f"preds:{torch.min(pos_logits)}")
            pos_logits_ = self.out(pos_logits)
            # print(f"preds:{torch.min(pos_logits_)}")
            m = nn.Sigmoid()
            preds = m(pos_logits_).squeeze(-1)
            preds = preds[:,1:]
            # print(f"preds:{preds.shape}")
        # print(f"preds:{torch.min(preds)}")
            if self.emb_type.find("qid_pvn") != -1:
                neg_ids = []
                total_c = torch.arange(0,self.n_question)
                for i in range(cshft.size(0)):
                    pos_sample = cshft[i].detach().cpu().numpy()
                    neg_list = list(set(np.array(total_c)).difference(set(pos_sample)))
                    neg_item = choice(neg_list)
                    neg_ids.append(neg_item)
                neg_ids = torch.tensor(neg_ids, dtype=torch.long, device=cshft.device).repeat(cshft.size(1),1).permute(1,0)
                pos_mean, _ = self.add_position_mean_embedding(cshft, rshft)
                pos_cov, _ = self.add_position_cov_embedding(cshft, rshft)    
                neg_mean, _ = self.add_position_mean_embedding(neg_ids, rshft)
                neg_cov, _ = self.add_position_cov_embedding(neg_ids, rshft)
                pos_vs_neg = wasserstein_distance(pos_mean, pos_cov, neg_mean, neg_cov).unsqueeze(2)
                # print(f"pos_vs_neg:{pos_vs_neg.shape}")
                pos_vs_neg = self.out_neg(pos_vs_neg).squeeze(-1)
                istarget = (cshft >= 0).view(cshft.size(0), cshft.size(1)).float() # [batch*seq_len]
                pvn_loss = self.args.pvn_weight * torch.sum(torch.clamp(pos_logits[:,1:].squeeze(-1) - pos_vs_neg, 0) * istarget) / torch.sum(istarget)

        if train:
            return preds, pvn_loss
        else:
            if qtest:
                return preds, mean_sequence_output, cov_sequence_output, mean_sequence_emb, cov_sequence_emb
            else:
                return preds

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.01, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DistMeanSAModel(StosaKT):
    def __init__(self, args):
        super(DistMeanSAModel, self).__init__(args)
        self.item_encoder = DistMeanSAEncoder(args)