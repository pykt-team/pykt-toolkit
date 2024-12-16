import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from torch.nn import (
    Module,
    Embedding,
    LSTM,
    Linear,
    Dropout,
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    CrossEntropyLoss,
    BCELoss,
    MultiheadAttention,
)
from torch.nn.functional import (
    one_hot,
    cross_entropy,
    multilabel_margin_loss,
    binary_cross_entropy,
)
from .que_base_model import QueBaseModel, QueEmb
from torch.utils.checkpoint import checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class DatasetEmb(nn.Module):
    def __init__(self, size, embed_l, device):
        super().__init__()
        self.dataset_prompt = nn.Parameter(
            torch.randn(size, embed_l).to(device), requires_grad=True
        )


class EmbeddingC(nn.Module):
    def __init__(self, size, embed_l, device):
        super().__init__()
        self.emb_c = nn.Parameter(
            torch.randn(size, embed_l).to(device), requires_grad=True
        )


class PromptQue(nn.Module):
    def __init__(self, embed_l, max_val, num_meta):
        super().__init__()
        self.param = torch.nn.Parameter(
            torch.zeros(num_meta, embed_l).to(device), requires_grad=True
        )
        self.meta_prompt_q = torch.nn.init.uniform_(self.param, a=-max_val, b=max_val)


class PromptKC(nn.Module):
    def __init__(self, embed_l, max_val, num_meta):
        super().__init__()
        self.param = torch.nn.Parameter(
            torch.zeros(num_meta, embed_l).to(device), requires_grad=True
        )
        self.meta_prompt_kc = torch.nn.init.uniform_(self.param, a=-max_val, b=max_val)


class autodis(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dims,
        dropout,
        number,
        temperature,
        output_layer=True,
        max_val=0.01,
    ):
        # input (1*16)->MLP->softmax->(number,1),multiply meta-embedding, output(1*16)
        super().__init__()
        layers = list()
        self.input_dim = input_dim
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())  # try torch.nn.Sigmoid & torch.nn.Tanh
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, number))
        self.mlp = torch.nn.Sequential(*layers)
        self.temperature = temperature

        self.PromptQue = PromptQue(self.input_dim, max_val, number)
        self.PromptKC = PromptKC(self.input_dim, max_val, number)

    def forward(self, q, c):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        logits_score_q = self.mlp(q)  # output(1*20)
        logits_norm_score_q = torch.nn.Softmax(dim=1)(logits_score_q / self.temperature)
        autodis_embedding_q = torch.matmul(
            logits_norm_score_q, self.PromptQue.meta_prompt_q
        )

        logits_score_c = self.mlp(c)  # output(1*20)
        logits_norm_score_kc = torch.nn.Softmax(dim=1)(
            logits_score_c / self.temperature
        )
        autodis_embedding_c = torch.matmul(
            logits_norm_score_kc, self.PromptKC.meta_prompt_kc
        )
        # print(f"autodis_embedding_c:{autodis_embedding_c.shape}")

        return autodis_embedding_q, autodis_embedding_c


class PromptDomain(nn.Module):
    def __init__(self, embed_l, max_val, num_meta):
        super().__init__()
        self.param = torch.nn.Parameter(
            torch.zeros(num_meta, embed_l).to(device), requires_grad=True
        )
        self.meta_prompt_domain = torch.nn.init.uniform_(
            self.param, a=-max_val, b=max_val
        )


class MultiLayerPerceptron_normal(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        input_d = input_dim
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, input_d))
        self.mlp = torch.nn.Sequential(*layers)

        for param in self.mlp.parameters():
            torch.nn.init.constant_(param.data, 0)
            # param.requires_grad = False

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class autodis_domain(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dims,
        dropout,
        number,
        temperature,
        output_layer=True,
        max_val=0.01,
    ):
        # input (1*16)->MLP->softmax->(number,1),multiply meta-embedding, output(1*16)
        super().__init__()
        layers = list()
        self.input_dim = input_dim
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())  # try torch.nn.Sigmoid & torch.nn.Tanh
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, number))
        self.mlp = torch.nn.Sequential(*layers)
        self.temperature = temperature

        self.PromptDomain = PromptDomain(self.input_dim, max_val, number)

    def forward(self, dataset):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        logits_score_domain = self.mlp(dataset)  # output(1*20)
        # print(f"logits_score_domain:{logits_score_domain.shape}")
        logits_norm_score_domain = torch.nn.Softmax(dim=1)(
            logits_score_domain / self.temperature
        )
        autodis_embedding_domain = torch.matmul(
            logits_norm_score_domain, self.PromptDomain.meta_prompt_domain
        )
        # print(f"autodis_embedding_domain:{autodis_embedding_domain.shape}")
        return autodis_embedding_domain


class promptKT(nn.Module):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        loss1=0.5,
        loss2=0.5,
        loss3=0.5,
        start=50,
        num_layers=2,
        nheads=4,
        seq_len=1024,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        num_attn_heads=8,
        separate_qa=False,
        l2=1e-5,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        cf_weight=0.3,
        t_weight=0.3,
        local_rank=1,
        num_sgap=None,
        c0=0,
        max_epoch=0,
        re_mapping=False,
        embed_dims=(32, 32),
        temperature=1e-5,
        max_val=0.01,
        num_meta=8,
    ):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "promptkt"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.ce_loss = BCELoss()
        self.embed_l = d_model
        self.remapping = re_mapping
        if self.remapping == "0":
            self.remapping = False
        if self.remapping == "1":
            self.remapping = True
        print(f"self.remapping:{self.remapping}")
        self.max_val = max_val
        self.num_meta = num_meta
        self.temperature = temperature
        self.embed_dims = embed_dims

        if self.emb_type.find("nodata") == -1:
            self.dataset_emb = nn.Embedding(20, self.embed_l).to(device)

        # dataset_id embedding
        if self.emb_type.find("freeze") != -1:
            for param in self.dataset_emb.parameters():
                param.requires_grad = False

        if self.emb_type.find("prompt_qc") != -1:
            self.dataset_prompt = DatasetEmb(20, self.embed_l, device)

        self.qa_embed = nn.Embedding(2, self.embed_l)
        if self.remapping:
            self.emb_q = nn.Embedding(self.n_pid + 1, self.embed_l)
            # question embeding
            self.emb_c = EmbeddingC(self.n_question + 1, self.embed_l, device)
        else:
            self.emb_q = nn.Embedding(200000, self.embed_l).to(
                device
            )  # question embedding
            self.emb_c = EmbeddingC(1000, self.embed_l, device)

        if self.emb_type.find("prompt_qc") != -1:
            self.autodis = autodis(
                self.embed_l,
                embed_dims,
                self.dropout,
                self.num_meta,
                self.temperature,
                max_val=self.max_val,
            )
        if self.emb_type.find("gene") != -1:
            self.mlp_2 = MultiLayerPerceptron_normal(self.embed_l, (32, 32), dropout)
        if self.emb_type.find("fusion") != -1:
            self.autodis = autodis_domain(
                self.embed_l,
                embed_dims,
                self.dropout,
                self.num_meta,
                self.temperature,
                max_val=self.max_val,
            )

        self.model = Architecture(
            n_question=n_question,
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model / num_attn_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            model_type=self.model_type,
            seq_len=seq_len,
        )

        self.out = nn.Sequential(
            nn.Linear(d_model + self.embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1),
        )

        if emb_type.find("frozen") != -1:
            # print(f"frozen")
            for param in self.parameters():
                param.requires_grad = False

            for name, param in self.named_parameters():
                if emb_type.find("gene") != -1:
                    # print(f"run gene_ft")
                    if "mlp_2" in name:
                        param.requires_grad = True
                if emb_type.find("fusion") != -1:
                    # print(f"run fusion_ft")
                    if "autodis" in name:
                        param.requires_grad = True
                if emb_type.find("frozen1") != -1:  # prompt tuning 更新dataset prompt
                    if "dataset_emb" in name:
                        param.requires_grad = True
                elif (
                    emb_type.find("frozen2") != -1
                ):  # prompt tuning 更新dataset prompt/multihead linear/output layer
                    # print(f"run fusion_ft")
                    if "dataset_emb" in name or "out." in name or "out_proj." in name:
                        param.requires_grad = True
                elif (
                    emb_type.find("frozen3") != -1
                ):  # prompt tuning 更新dataset prompt/que prompt/KC prompt/multihead linear/output layer
                    if (
                        "dataset_emb" in name
                        or "out." in name
                        or "out_proj." in name
                        or "dataset_prompt" in name
                        or "autodis" in name
                    ):
                        param.requires_grad = True
                elif (
                    emb_type.find("frozen4") != -1
                ):  # prompt tuning更新 dataset prompt/multihead linear
                    if "dataset_emb" in name or "out_proj." in name:
                        param.requires_grad = True
                elif (
                    emb_type.find("frozen5") != -1
                ):  # prompt tuning更新 dataset prompt/FFN linear
                    if (
                        "dataset_emb" in name
                        or "linear1." in name
                        or "linear2." in name
                    ):
                        param.requires_grad = True
                elif (
                    emb_type.find("frozen6") != -1
                ):  # prompt tuning更新 dataset prompt/FFN linear/output layer
                    if (
                        "dataset_emb" in name
                        or "linear1." in name
                        or "linear2." in name
                        or "out." in name
                    ):
                        param.requires_grad = True
                elif (
                    emb_type.find("frozen7") != -1
                ):  # prompt tuning更新 dataset prompt/multihead linear/FFN linear/output layer
                    if (
                        "dataset_emb" in name
                        or "out_proj." in name
                        or "linear1." in name
                        or "linear2." in name
                        or "out." in name
                    ):
                        param.requires_grad = True
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.0)

    def get_avg_skill_emb(self, c):
        # add zero for padding
        concept_emb_cat = torch.cat(
            [torch.zeros(1, self.embed_l).to(device), self.emb_c.emb_c], dim=0
        )
        # shift c

        related_concepts = (c + 1).long()
        # [batch_size, seq_len, emb_dim]
        concept_emb_sum = concept_emb_cat[related_concepts, :].sum(axis=-2)

        # [batch_size, seq_len,1]
        concept_num = (
            torch.where(related_concepts != 0, 1, 0).sum(axis=-1).unsqueeze(-1)
        )
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = concept_emb_sum / concept_num
        return concept_avg

    def forward(self, dcur, qtest=False, train=False, dgaps=None):
        q, c, r = (
            dcur["qseqs"].long().to(device),
            dcur["cseqs"].long().to(device),
            dcur["rseqs"].long().to(device),
        )
        qshft, cshft, rshft = (
            dcur["shft_qseqs"].long().to(device),
            dcur["shft_cseqs"].long().to(device),
            dcur["shft_rseqs"].long().to(device),
        )

        if self.emb_type.find("nodata") == -1:
            dataset_id = dcur["dataset_id"].long().to(device)
        # print(f"dataset_id:{dataset_id}")
        pid_data = torch.cat((q[:, 0:1], qshft), dim=1)  # shape[batch,200]
        q_data = torch.cat((c[:, 0:1], cshft), dim=1)  # shape[batch,200,7]
        target = torch.cat((r[:, 0:1], rshft), dim=1)

        emb_q = self.emb_q(pid_data)  # [batch,max_len-1,emb_size]
        emb_c = self.get_avg_skill_emb(q_data)  # [batch,max_len-1,emb_size]
        if self.emb_type.find("nodata") == -1:
            dataset_embed_data = self.dataset_emb(dataset_id).unsqueeze(1)
        # print(f"dataset_embed_data:{dataset_embed_data.shape}")
        qa_embed_data = self.qa_embed(target)

        if self.emb_type.find("nodata") == -1:
            # print(f"add_dataset")
            q_embed_data = emb_q + emb_c + dataset_embed_data
        else:
            q_embed_data = emb_q + emb_c

        if self.emb_type.find("prompt_qc") != -1:
            # 构建question的prompt
            bs, seqlen = pid_data.size(0), pid_data.size(1)
            multi_hot_q = torch.zeros(bs, seqlen, 6).to(device)
            # 根据条件设置multi_hot中对应位置上的值
            multi_hot_q[pid_data < 948] = (
                torch.tensor([1, 1, 1, 1, 1, 1]).float().to(device)
            )  # nips34
            multi_hot_q[(pid_data >= 948) & (pid_data < 7652)] = (
                torch.tensor([1, 1, 1, 1, 0, 1]).float().to(device)
            )  # peiyou
            multi_hot_q[(pid_data >= 7652) & (pid_data < 12277)] = (
                torch.tensor([1, 1, 1, 1, 0, 0]).float().to(device)
            )  # ednet_all
            multi_hot_q[(pid_data >= 12277) & (pid_data < 17737)] = (
                torch.tensor([0, 1, 1, 1, 0, 0]).float().to(device)
            )  # assist2009
            multi_hot_q[(pid_data >= 17737) & (pid_data < 129263)] = (
                torch.tensor([0, 0, 1, 1, 0, 0]).float().to(device)
            )  # algebra2006
            multi_hot_q[(pid_data >= 129263) & (pid_data < 173113)] = (
                torch.tensor([0, 0, 1, 0, 0, 0]).float().to(device)
            )  # algebra2005
            multi_hot_q = torch.cat(
                [
                    multi_hot_q,
                    torch.zeros(bs, seqlen, 20 - multi_hot_q.size(2)).to(device),
                ],
                dim=2,
            )
            multi_hot_q = multi_hot_q.clone()
            dataset_emb_data = self.dataset_prompt.dataset_prompt
            multi_hot_q_emb = torch.matmul(multi_hot_q, dataset_emb_data) / (
                (torch.sum(multi_hot_q, dim=2).unsqueeze(-1) + 0.01)
            )
            # print(f"multi_hot_q_emb:{multi_hot_q_emb.shape}")

            # 构建KC的prompt
            multi_hot_kc = torch.zeros(
                q_data.size(0), q_data.size(1), q_data.size(2), 6
            ).to(device)
            multi_hot_kc[q_data < 0] = (
                torch.tensor([0, 0, 0, 0, 0, 0]).float().to(device)
            )  # nips34
            multi_hot_kc[(q_data >= 0) & (q_data < 57)] = (
                torch.tensor([1, 1, 1, 1, 1, 1]).float().to(device)
            )  # nips34
            multi_hot_kc[(q_data >= 57) & (q_data < 112)] = (
                torch.tensor([1, 1, 1, 1, 0, 1]).float().to(device)
            )  # algebra2006
            multi_hot_kc[(q_data >= 112) & (q_data < 123)] = (
                torch.tensor([1, 1, 0, 1, 0, 1]).float().to(device)
            )  # assist2009
            multi_hot_kc[(q_data >= 123) & (q_data < 188)] = (
                torch.tensor([1, 0, 0, 1, 0, 1]).float().to(device)
            )  # ednall
            multi_hot_kc[(q_data >= 188) & (q_data < 493)] = (
                torch.tensor([0, 0, 0, 1, 0, 1]).float().to(device)
            )  # algebra2006
            multi_hot_kc[(q_data >= 493) & (q_data < 865)] = (
                torch.tensor([0, 0, 0, 0, 0, 1]).float().to(device)
            )  # peiyou
            # print(f"multi_hot_kc 1:{multi_hot_kc.shape}")
            tmp_kc_sum = torch.sum(multi_hot_kc, dim=3).unsqueeze(-1) + 0.01
            multi_hot_kc = torch.cat(
                [
                    multi_hot_kc,
                    torch.zeros(
                        bs, seqlen, q_data.size(2), 20 - multi_hot_kc.size(3)
                    ).to(device),
                ],
                dim=3,
            )
            # print(f"multi_hot_kc 2:{multi_hot_kc.shape}")
            # print(f"dataset_emb_data:{dataset_emb_data.shape}")

            tmp_kc_result = torch.matmul(multi_hot_kc, dataset_emb_data)
            # print(f"tmp_result:{tmp_kc_result.shape}")
            tmp_kc_sum = torch.sum(multi_hot_kc, dim=3).unsqueeze(-1) + 0.01
            multi_hot_kc_emb = torch.mean(tmp_kc_result / tmp_kc_sum, dim=2)
            # print(f"multi_hot_kc:{multi_hot_kc_emb.shape}")

            autodis_embedding_q, autodis_embedding_c = self.autodis(
                multi_hot_q_emb.reshape(-1, self.embed_l),
                multi_hot_kc_emb.reshape(-1, self.embed_l),
            )
            autodis_embedding_q, autodis_embedding_c = autodis_embedding_q.reshape(
                q_embed_data.size(0), q_embed_data.size(1), -1
            ), autodis_embedding_c.reshape(
                q_embed_data.size(0), q_embed_data.size(1), -1
            )
            q_embed_data_all = q_embed_data + autodis_embedding_q + autodis_embedding_c

            qa_embed_data_all = q_embed_data_all + qa_embed_data

        # BS.seqlen,d_model
        y2, y3 = 0, 0
        if self.emb_type.find("prompt_qc") != -1:
            d_output = self.model((q_embed_data_all, qa_embed_data_all))
        else:
            if self.emb_type.find("fusion") != -1:
                # print(f"add fusion")
                autodis_embedding_domain = self.autodis(
                    dataset_embed_data.reshape(-1, self.embed_l),
                )
                autodis_embedding_domain = autodis_embedding_domain.reshape(
                    dataset_embed_data.size(0),
                    dataset_embed_data.size(1),
                    -1,
                )
                # print(f"dataset_embed_data:{dataset_embed_data.shape}")
                # print(f"autodis_embedding_domain:{dataset_embed_data.shape}")
                q_embed_data = q_embed_data + autodis_embedding_domain
            if self.emb_type.find("gene") != -1:
                # print(f"add mlp")
                domain_prompt = self.mlp_2(
                    dataset_embed_data.reshape(-1, self.embed_l)
                ).reshape(
                    dataset_embed_data.size(0),
                    dataset_embed_data.size(1),
                    -1,
                )
                # print(f"domain_prompt:{domain_prompt.shape}")
                q_embed_data = q_embed_data + domain_prompt
            qa_embed_data = q_embed_data + qa_embed_data
            d_output = self.model((q_embed_data, qa_embed_data))
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)

        cl_losses = 0

        if train:
            if self.emb_type in ["qid", "qid_prompt_qc"]:
                return preds, y2, y3
            else:
                return preds, y2, y3, cl_losses
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds


class Architecture(nn.Module):
    def __init__(
        self,
        n_question,
        n_blocks,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        model_type,
        seq_len,
    ):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {"promptkt", "unikt"}:
            self.blocks_2 = nn.ModuleList(
                [
                    TransformerLayer(
                        d_model=d_model,
                        d_feature=d_model // n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        n_heads=n_heads,
                        kq_same=kq_same,
                    )
                    for _ in range(n_blocks)
                ]
            )
        self.position_emb = CosinePositionalEmbedding(
            d_model=self.d_model, max_len=seq_len
        )

    def forward(self, inputs):
        # target shape  bs, seqlen
        q_embed_data, qa_embed_data = inputs
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder

        for block in self.blocks_2:
            # x.requires_grad_(True)
            # y.requires_grad_(True)
            # def run_block(mask, query, key, values, apply_pos):
            #     return block(mask, query, key, values, apply_pos)
            # x = checkpoint(run_block, mask, x, x, y, apply_pos)

            x = checkpoint(block, x, x, y)
            # x = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
            # x = input_data[1]
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, values):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """
        mask = 0
        apply_pos = True
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True
            )  # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False
            )

        query = query + self.dropout1((query2))  # 残差1
        query = self.layer_norm1(query)  # layer norm
        if apply_pos:
            query2 = self.linear2(
                self.dropout(self.activation(self.linear1(query)))  # FFN
            )
            query = query + self.dropout2((query2))  # 残差
            query = self.layer_norm2(query)  # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
        d_k
    )  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)  # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).long()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).long() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]  # ( 1,seq,  Feature)
