import torch
import numpy as np
from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
        
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAKT_Enhance_PRO(Module):
    def __init__(self, num_q, seq_len, emb_size, num_attn_heads, dropout, num_en=2, emb_type="qid", emb_path="", pretrain_dim=768, dataset_name='THINK'):
        super().__init__()
        self.model_name = "sakt_enhance_pro"
        self.emb_type = emb_type

        self.num_q = num_q
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en

        if emb_type.startswith("qid"):
            # num_c, seq_len, emb_size, num_attn_heads, dropout, emb_path="")
            self.interaction_emb = Embedding(num_q * 2, emb_size)
            self.exercise_emb = Embedding(num_q, emb_size)
            # self.P = Parameter(torch.Tensor(self.seq_len, self.emb_size))
        self.position_emb = Embedding(seq_len, emb_size)

        self.blocks = get_clones(Blocks(emb_size, num_attn_heads, dropout), self.num_en)
        self.ans_emb = Embedding(2, emb_size)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)
        
        
        '''
        embedding 增强部分  2025.1018
        '''
        # 通过数据集名称判断到底用哪一个文件
        self.dataset_name = dataset_name
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
                "sakt_enhance_pro only supports datasets with question text embeddings: "
                "jiuzhang_grade3_en, jiuzhang_grade45_cn, jiuzhang_grade7_cn. "
                f"Got dataset_name={self.dataset_name!r}."
            )

        # 加载语义embedding和无语义embedding的问题列表
        self.semantic_emb_path = f'../data/pro_emb/qid_final/{num}/question_embeddings_overall.npy'
        self.no_q_path = f'../data/pro_emb/qid_final/{num}/data/no_q.json'
        self.setup_question_embeddings()


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
    
    
    def base_emb(self, q, r, qry):
        
        # x = q + self.num_q * r
        # qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
        
        # 替换成对应的语义embedding
        qshftemb = self.get_question_embedding(qry)
        xemb = self.get_question_embedding(q) + self.ans_emb(r)
    
        posemb = self.position_emb(pos_encode(xemb.shape[1]))
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, q, r, c, qry, cshift, qtest=False):
        emb_type = self.emb_type
        qemb, qshftemb, xemb = None, None, None
        if emb_type == "qid":
            qshftemb, xemb = self.base_emb(q, r, qry)
        # print(f"qemb: {qemb.shape}, xemb: {xemb.shape}, qshftemb: {qshftemb.shape}")
        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        if not qtest:
            return p
        else:
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
        # attn -> drop -> skip -> norm 
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(seq_len = k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb
