import os
import json

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DKVMN_Enhance_PRO(Module):
    def __init__(self, num_q, dim_s, size_m, dropout=0.2, emb_type='qid', emb_path="", pretrain_dim=768, dataset_name="THINK"):
        super().__init__()
        self.model_name = "dkvmn_enhance_pro"
        self.num_q = num_q
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.k_emb_layer = Embedding(self.num_q, self.dim_s)
            self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_q * 2, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)
        
        '''
        embedding 增强部分  2025.1018
        '''
        # 通过数据集名称判断到底用哪一个文件
        self.dataset_name = dataset_name
        self.emb_size = dim_s
        num = "3"
        # print("+" * 100)
        # print(f"dataset_name is {self.dataset_name}")
        # print("+" * 100)
        if self.dataset_name in ["THINK", "jiuzhang_grade3_en"]:
            num = "3"
        elif self.dataset_name in ["MATH_G4-5", "jiuzhang_grade45_cn"]:
            num = "45"
        elif self.dataset_name in ["MATH_G7", "jiuzhang_grade7_cn"]:
            num = "7"
        else:
            raise ValueError(
                "dkvmn_enhance_pro only supports datasets with question text embeddings: "
                "jiuzhang_grade3_en, jiuzhang_grade45_cn, jiuzhang_grade7_cn. "
                f"Got dataset_name={self.dataset_name!r}."
            )

        # 加载语义embedding和无语义embedding的问题列表
        self.semantic_emb_path = f'../data/pro_emb/qid_final/{num}/question_embeddings_overall.npy'
        self.no_q_path = f'../data/pro_emb/qid_final/{num}/data/no_q.json'
        self.setup_question_embeddings()
        
        self.ans_emb = Embedding(2, dim_s)
        
    '''
    embedding 增强部分  2025.1018
    '''
    def setup_question_embeddings(self):
        """设置语义embedding和随机embedding"""
        # 加载无语义embedding的问题id列表
        if self.no_q_path and os.path.exists(self.no_q_path):
            with open(self.no_q_path, 'r') as f:
                self.no_semantic_questions = set(json.load(f))
            print(f"Loaded {len(self.no_semantic_questions)} questions without semantic embeddings")
        else:
            self.no_semantic_questions = set()
            print("No no_q.json file found, using random embeddings for all questions")
        
        # 加载预训练的语义embedding
        if self.semantic_emb_path and os.path.exists(self.semantic_emb_path):
            # print("+" * 100)
            # print(f"semantic_emb_path is {self.semantic_emb_path}")
            # print("+" * 100)
            semantic_embeddings = np.load(self.semantic_emb_path)
            print(f"Loaded semantic embeddings with shape: {semantic_embeddings.shape}")
            
            # 转换为torch tensor
            semantic_embeddings = torch.FloatTensor(semantic_embeddings)
            
            # 创建预训练embedding层（可训练）
            self.semantic_emb = Embedding.from_pretrained(semantic_embeddings, freeze=False).to(device)
            
            # 如果语义embedding的维度与emb_size不同，添加线性变换
            if semantic_embeddings.shape[1] != self.emb_size:
                self.semantic_proj = Linear(semantic_embeddings.shape[1], self.emb_size).to(device)
                print(f"Added projection layer from {semantic_embeddings.shape[1]} to {self.emb_size}")
            else:
                self.semantic_proj = None
        else:
            # print("+" * 100)
            # print(f"semantic_emb_path is {self.semantic_emb_path}, semantic_emb_path is not found")
            # print("+" * 100)
            self.semantic_emb = None
            self.semantic_proj = None
            print("No semantic embeddings file found, using random embeddings for all questions")

    def get_question_embedding(self, q_ids):
        """根据问题id获取对应的embedding（语义或随机）"""
        batch_size, seq_len = q_ids.shape
        device = q_ids.device
        
        # 初始化输出embedding
        question_emb = torch.zeros(batch_size, seq_len, self.emb_size, device=device)
        
        if self.semantic_emb is not None:
            # 创建mask来区分有无语义embedding的问题
            no_semantic_mask = torch.zeros_like(q_ids, dtype=torch.bool)
            for no_q_id in self.no_semantic_questions:
                no_semantic_mask |= (q_ids == no_q_id)
            
            # 对于有语义embedding的问题
            semantic_mask = ~no_semantic_mask
            if semantic_mask.any():
                semantic_q_ids = q_ids[semantic_mask]
                semantic_emb = self.semantic_emb(semantic_q_ids)
                
                # 如果需要维度变换
                if self.semantic_proj is not None:
                    semantic_emb = self.semantic_proj(semantic_emb)
                
                question_emb[semantic_mask] = semantic_emb
            
            # 对于没有语义embedding的问题，使用随机embedding
            if no_semantic_mask.any():
                no_semantic_q_ids = q_ids[no_semantic_mask]
                random_emb = self.pro_emb(no_semantic_q_ids)
                question_emb[no_semantic_mask] = random_emb
        else:
            # 如果没有语义embedding文件，全部使用随机embedding
            # print("+" * 100)
            # print("No semantic embeddings file found, using random embeddings for all questions")
            # print("+" * 100)
            question_emb = self.pro_emb(q_ids)
        
        return question_emb

    '''
    embedding 增强部分  2025.1018
    '''

    def forward(self, q, r, qtest=False):
        emb_type = self.emb_type
        batch_size = q.shape[0]
        if emb_type == "qid":
            # x = q + self.num_q * r
            # 替换成语义增强的embedding
            x = self.get_question_embedding(q) + self.ans_emb(r)
            # k = self.k_emb_layer(q)
            k = self.get_question_embedding(q)
            # v = self.v_emb_layer(x)
            v = x
        
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        p = self.p_layer(self.dropout_layer(f))

        p = torch.sigmoid(p)
        # print(f"p: {p.shape}")
        p = p.squeeze(-1)
        if not qtest:
            return p
        else:
            return p, f
