import os
import numpy as np
import torch
import json
from torch import nn as nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DKT_Enhance_Pro(Module):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", dataset_name='THINK',
                 pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt_enhance_pro"
        self.num_c = num_c
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.dropout = dropout
        self.dataset_name = dataset_name
        
        # 通过数据集名称判断到底用哪一个文件
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
                "dkt_enhance_pro only supports datasets with question text embeddings: "
                "jiuzhang_grade3_en, jiuzhang_grade45_cn, jiuzhang_grade7_cn. "
                f"Got dataset_name={self.dataset_name!r}."
            )

        # 加载语义embedding和无语义embedding的问题列表
        self.semantic_emb_path = f'../data/pro_emb/qid_final/{num}/question_embeddings_overall.npy'
        self.no_q_path = f'../data/pro_emb/qid_final/{num}/data/no_q.json'
        self.setup_question_embeddings()

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True).to(device)
        self.dropout_layer = Dropout(dropout).to(device)
        
        # 修复原代码中的变量名问题
        d = self.hidden_size
        self.out_layer = nn.Sequential(
            nn.Linear(2 * d, d).to(device),
            nn.ReLU(),
            nn.Dropout(p=self.dropout).to(device),
            nn.Linear(d, 1).to(device)
        ).to(device)
        
        self.ans_emb = nn.Embedding(2, self.emb_size).to(device)
        # 注意：这里的随机问题embedding现在只用于没有语义embedding的问题
        self.pro_emb = nn.Embedding(self.num_q, self.emb_size).to(device)


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

    def forward(self, last_pro, last_ans, last_skill, next_pro, next_skill, perb=None):
        '''
        Args:
            last_pro: [batch_size, seq_len] - 上一题的问题id
            last_ans: [batch_size, seq_len] - 上一题的答案（0或1）
            last_skill: [batch_size, seq_len] - 上一题的技能
            next_pro: [batch_size, seq_len] - 下一题的问题id
            next_skill: [batch_size, seq_len] - 下一题的技能
        '''
        
        # 将输入移动到设备
        last_pro = last_pro.to(device)
        last_ans = last_ans.to(device)
        next_pro = next_pro.to(device)
        
        # 获取问题的embedding（混合语义和随机）
        last_pro_embedding = self.get_question_embedding(last_pro)
        last_ans_embedding = self.ans_emb(last_ans)
        next_pro_embedding = self.get_question_embedding(next_pro)

        # 用于测试随机embedding
        # last_pro_embedding = self.pro_emb(last_pro)
        # last_ans_embedding = self.ans_emb(last_ans)
        # next_pro_embedding = self.pro_emb(next_pro)
        
        # 组合上一题的问题和答案embedding
        xemb = last_pro_embedding + last_ans_embedding
        next_xemb = next_pro_embedding

        # LSTM 层
        h, _ = self.lstm_layer(xemb.to(device))

        # 输出层：结合LSTM输出和下一题embedding
        combined_features = torch.cat([h, next_xemb], dim=-1).to(device)
        y = torch.sigmoid(self.out_layer(combined_features)).squeeze(-1)

        return y

