import random
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class UKT(nn.Module):
    """
    Uncertainty-aware Knowledge Tracing (UKT) model.
    
    UKT tracks students' knowledge states using stochastic embeddings (mean and covariance)
    to represent uncertainty in learning, with a Wasserstein-based attention mechanism to
    capture knowledge state transitions.
    """
    def __init__(self, n_question, n_pid, 
            d_model, n_blocks, dropout, d_ff=256, num_layers=2, seq_len=200, nheads=8, loss1=0, loss2=0, loss3=0, start=50,
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, use_CL = True,use_mean_cov_diff=False, cl_weight=0.02, use_uncertainty_aug=True, l2=1e-5, emb_type="stoc_qid",atten_type='w2', emb_path="", pretrain_dim=768):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
            use_CL (bool): Whether to use contrastive learning for handling aleatory uncertainty.
            cl_weight (float): Weight for the contrastive learning loss.
            use_uncertainty_aug (bool): Whether to use uncertainty augmentation in contrastive learning.
            l2 (float): L2 regularization coefficient.
        """
        self.model_name = "ukt"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.use_CL = use_CL
        self.use_uncertainty_aug = use_uncertainty_aug
        self.atten_type = atten_type
      

        embed_l = d_model
        if use_CL:
            self.wloss = WassersteinNCELoss(1)
            self.cl_weight = cl_weight

        self.embed_l = d_model
        if self.n_pid > 0 :
            if emb_type.find("scalar") != -1:
                self.difficult_param = nn.Embedding(self.n_pid+1, 1) # question difficulty
            else:
                self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) 

            self.q_embed_diff  = nn.Embedding(self.n_question+1, embed_l) # question emb, summarizes the changes of problems (questions) including the current question (concept)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 

        if emb_type.startswith("qid") or emb_type.startswith("stoc"):
            # n_question+1 ,d_model
            self.mean_q_embed = nn.Embedding(self.n_question, embed_l) # mean embedding
            self.cov_q_embed = nn.Embedding(self.n_question, embed_l) # covariance embedding
            # interaction embedding
            if self.separate_qa:
                    self.mean_qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
                    self.cov_qa_embed = nn.Embedding(2*self.n_question+1, embed_l)

            else: # false default
                self.mean_qa_embed = nn.Embedding(2, embed_l)
                self.cov_qa_embed = nn.Embedding(2, embed_l)


        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)

        self.out = nn.Sequential(
            nn.Linear(embed_l+embed_l+embed_l+embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )
        self.reset()


    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_mean_embed_data = self.mean_q_embed(q_data) # mean embeddings for questions/KCs.
        q_cov_embed_data = self.cov_q_embed(q_data) # covariance embeddings for questions/KCs.

        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_mean_embed_data = self.mean_qa_embed(qa_data) # mean embeddings for interactions.
            qa_cov_embed_data  = self.cov_qa_embed(qa_data) # covariance embeddings for interactions.

        else:
            qa_mean_embed_data = self.mean_qa_embed(target) + q_mean_embed_data
            qa_cov_embed_data  = self.cov_qa_embed(target)  + q_cov_embed_data

        return q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)
        mask = dcur["masks"]

        # Prepare augmented response data for CL
        if train and self.use_CL:
            if self.use_uncertainty_aug:
                # Create augmented target by introducing aleatory uncertainty
                rshft_aug = dcur["shft_r_aug"].long()
                r_aug     = dcur["r_aug"].long()
                target_aug = torch.cat((r_aug[:,0:1], rshft_aug), dim=1)
            else:
                target_aug = target

        emb_type = self.emb_type
         # Generate stochastic embeddings for questions and responses # Equation (1)
        if emb_type.startswith("qid") or emb_type.startswith("stoc"):
            q_mean_embed_data,q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data = self.base_emb(q_data, target)

            if train and self.use_CL:
                mean_q_aug_embed_data,cov_q_aug_embed_data, mean_qa_aug_embed_data, cov_qa_aug_embed_data = self.base_emb(q_data, target_aug)

        if self.n_pid > 0 and emb_type.find("norasch") == -1: # have problem id
            if emb_type.find("aktrasch") == -1:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_mean_embed_data = q_mean_embed_data + pid_embed_data *\
                        q_embed_diff_data
                q_cov_embed_data = q_cov_embed_data + pid_embed_data *\
                        q_embed_diff_data
                if train and self.use_CL:
                    mean_q_aug_embed_data = mean_q_aug_embed_data + pid_embed_data *\
                            q_embed_diff_data
                    cov_q_aug_embed_data = cov_q_aug_embed_data + pid_embed_data *\
                            q_embed_diff_data

            else:
                # here
                q_embed_diff_data = self.q_embed_diff(q_data)
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度

                q_mean_embed_data = q_mean_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder
                q_cov_embed_data = q_cov_embed_data + pid_embed_data *\
                    q_embed_diff_data

                qa_embed_diff_data = self.qa_embed_diff(target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量

                qa_mean_embed_data = qa_mean_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)
                qa_cov_embed_data = qa_cov_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)
                if train and self.use_CL:
                    qa_aug_embed_diff_data = self.qa_embed_diff(target_aug)
                    mean_q_aug_embed_data = mean_q_aug_embed_data + pid_embed_data * q_embed_diff_data
                    cov_q_aug_embed_data = cov_q_aug_embed_data + pid_embed_data * q_embed_diff_data
                    mean_qa_aug_embed_data = mean_qa_aug_embed_data + pid_embed_data * (qa_aug_embed_diff_data+q_embed_diff_data)
                    cov_qa_aug_embed_data = cov_qa_aug_embed_data + pid_embed_data * (qa_aug_embed_diff_data+q_embed_diff_data)

        # BS.seqlen,d_model

        y2, y3 = 0, 0
     
        if emb_type in ["qid","stoc_qid", "qidaktrasch", "qid_scalar", "qid_norasch"]:

            mean_d_output,cov_d_output = self.model(q_mean_embed_data,q_cov_embed_data, qa_mean_embed_data,qa_cov_embed_data,self.atten_type)
            # Calculate CL loss if in training mode
            if train and self.use_CL:
                # Process augmented embeddings
                mean_d2_output, cov_d2_output = self.model(mean_q_aug_embed_data,cov_q_aug_embed_data, mean_qa_aug_embed_data,cov_qa_aug_embed_data,self.atten_type)
                # Create mask for pooling
                mas = mask
                true_tensor = torch.ones(mas.size(0), 1, dtype=torch.bool).to(device)
                mas = torch.cat((true_tensor, mas), dim=1).unsqueeze(-1)

                # Pool outputs for contrastive learning
                pooled_mean_d_output = torch.mean(mean_d_output*mas,dim = 1)
                pooled_cov_d_output = torch.mean(cov_d_output*mas,dim = 1)
                pooled_mean_d2_output = torch.mean(mean_d2_output*mas,dim = 1)
                pooled_cov_d2_output = torch.mean(cov_d2_output*mas,dim = 1)
                # Calculate Wasserstein distance-based contrastive loss. Equation (7)
                if emb_type == "stoc_qid":
                    # both mean and covariance
                    loss = self.wloss(pooled_mean_d_output, pooled_cov_d_output, pooled_mean_d2_output, pooled_cov_d2_output)
                else:
                    # only mean
                    loss = self.wloss(pooled_mean_d_output, pooled_mean_d_output, pooled_mean_d2_output, pooled_mean_d2_output)
            # Calculate uncertainty measure
            activation = nn.ELU()
            temp = torch.mean(torch.mean(activation(cov_d_output)+1,dim = -1),-1)
            # Concatenate features for final prediction
            if emb_type == "stoc_qid":
                concat_q = torch.cat([mean_d_output,cov_d_output,q_mean_embed_data,q_cov_embed_data], dim=-1)
            else:
                concat_q = torch.cat([mean_d_output,mean_d_output,q_cov_embed_data,q_cov_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)

        if train:
            if self.use_CL:
                return preds, loss, y2, y3, temp
            else:
                return preds, y2, y3
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

def wasserstein_distance(mean1, cov1, mean2, cov2):
    ret = torch.sum((mean1 - mean2) * (mean1 - mean2), -1)
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24))
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    ret = ret + torch.sum((cov1_sqrt - cov2_sqrt) * (cov1_sqrt - cov2_sqrt), -1)

    return ret

def d2s_1overx(distance):

    return 1/(1+distance)

class WassersteinNCELoss(nn.Module):
    def __init__(self, temperature):
        super(WassersteinNCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.temperature = temperature
        self.activation = nn.ELU()

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one_mean, batch_sample_one_cov, batch_sample_two_mean, batch_sample_two_cov):
        # batch_sample_one_mean = nn.functional.normalize(batch_sample_one_mean)
        # batch_sample_two_mean = nn.functional.normalize(batch_sample_two_mean)
        # batch_sample_one_cov = nn.functional.normalize(self.activation(batch_sample_one_cov) + 1)
        # batch_sample_two_cov = nn.functional.normalize(self.activation(batch_sample_two_cov) + 1)
        batch_sample_one_cov = self.activation(batch_sample_one_cov) + 1
        batch_sample_two_cov = self.activation(batch_sample_two_cov) + 1
        #batch_sample_one_cov = torch.ones_like(batch_sample_one_cov) 
        #batch_sample_two_cov = torch.ones_like(batch_sample_one_cov) 
        sim11 = d2s_1overx(wasserstein_distance_matmul(batch_sample_one_mean, batch_sample_one_cov, batch_sample_one_mean, batch_sample_one_cov)) / self.temperature
        sim22 = d2s_1overx(wasserstein_distance_matmul(batch_sample_two_mean, batch_sample_two_cov, batch_sample_two_mean, batch_sample_two_cov)) / self.temperature
        sim12 = -d2s_1overx(wasserstein_distance_matmul(batch_sample_one_mean, batch_sample_one_cov, batch_sample_two_mean, batch_sample_two_cov)) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss



class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        self.position_mean_embeddings = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)
        self.position_cov_embeddings = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)


        if model_type in {'ukt'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])




    def forward(self, q_mean_embed_data,q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data, atten_type='w2'):
        # target shape  bs, seqlen
        seqlen, batch_size = q_mean_embed_data.size(1), q_mean_embed_data.size(0)
        # Equation (2)
        mean_q_posemb = self.position_mean_embeddings(q_mean_embed_data)
        cov_q_posemb = self.position_cov_embeddings(q_cov_embed_data)

        q_mean_embed_data = q_mean_embed_data + mean_q_posemb
        q_cov_embed_data = q_cov_embed_data + cov_q_posemb

        qa_mean_posemb = self.position_mean_embeddings(qa_mean_embed_data)
        qa_cov_posemb = self.position_cov_embeddings(qa_cov_embed_data)

        qa_mean_embed_data = qa_mean_embed_data + qa_mean_posemb
        qa_cov_embed_data = qa_cov_embed_data + qa_cov_posemb
        
        # Equation (3)
        elu_act = torch.nn.ELU()
        q_mean_embed_data = q_mean_embed_data
        q_cov_embed_data = elu_act(q_cov_embed_data) + 1
        qa_mean_embed_data = qa_mean_embed_data
        qa_cov_embed_data = elu_act(qa_cov_embed_data) + 1

        mean_qa_pos_embed = qa_mean_embed_data
        cov_qa_pos_embed = qa_cov_embed_data

        mean_q_pos_embed = q_mean_embed_data
        cov_q_pos_embed = q_cov_embed_data

        # y = qa_pos_embed
        y_mean = mean_qa_pos_embed
        y_cov  = cov_qa_pos_embed

        seqlen, batch_size = y_mean.size(1), y_mean.size(0)

        # x = q_pos_embed
        x_mean = mean_q_pos_embed
        x_cov = cov_q_pos_embed

        # encoder
        for block in self.blocks_2:
            x_mean,x_cov = block(mask=0, query_mean=x_mean, query_cov = x_cov, key_mean=x_mean,key_cov=x_cov, values_mean=y_mean, values_cov = y_cov, atten_type=atten_type, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever

        return x_mean,x_cov

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.mean_linear1 = nn.Linear(d_model, d_ff)
        self.cov_linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.mean_linear2 = nn.Linear(d_ff, d_model)
        self.cov_linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation2 = nn.ELU()

    def forward(self, mask, query_mean, query_cov, key_mean, key_cov, values_mean, values_cov, atten_type='w2',apply_pos=True):

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

        # seqlen, batch_size = query.size(1), query.size(0)
        seqlen, batch_size = query_mean.size(1), query_mean.size(0)


        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')

        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2_mean, query2_cov = self.masked_attn_head(
                query_mean, query_cov, key_mean, key_cov, values_mean, values_cov, mask=src_mask, atten_type=atten_type,zero_pad=True # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
            )

        else:
            # Calls block.masked_attn_head.forward() method
            query2_mean, query2_cov = self.masked_attn_head(
                query_mean, query_cov, key_mean, key_cov, values_mean, values_cov, mask=src_mask, atten_type=atten_type,zero_pad=False
            )


        query_mean = query_mean + self.dropout1((query2_mean)) #残差
        query_cov = query_cov + self.dropout1((query2_cov))

        query_mean = self.layer_norm1(query_mean)
        query_cov = self.layer_norm1(self.activation2(query_cov) + 1)
        # Equation (6)
        if apply_pos:
            query2_mean = self.mean_linear2(self.dropout( 
                self.activation(self.mean_linear1(query_mean))))
            query2_cov = self.cov_linear2(self.dropout( 
                self.activation(self.cov_linear1(query_cov))))

            query_mean = query_mean + self.dropout2((query2_mean))
            query_cov = query_cov + self.dropout2((query2_cov)) 
            query_mean = self.layer_norm2(query2_mean)
            query_cov = self.layer_norm2(self.activation2(query2_cov)+1)

        return query_mean, query_cov

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
        self.activation = nn.ELU()
        self.v_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_cov_linear = nn.Linear(d_model, d_model, bias=bias)

        self.k_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_cov_linear = nn.Linear(d_model, d_model, bias=bias)

        if kq_same is False:
            # self.q_linear = nn.Linear(d_model, d_model, bias=bias)
            self.q_mean_linear = nn.Linear(d_model, d_model, bias=bias)
            self.q_cov_linear = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        # self.out_proj = nn.Linear(d_model, d_model, bias=bias)
       
        self.out_mean_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_cov_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_mean_linear.weight)
        xavier_uniform_(self.k_cov_linear.weight)

        xavier_uniform_(self.v_mean_linear.weight)
        xavier_uniform_(self.v_cov_linear.weight)


        if self.kq_same is False:
            xavier_uniform_(self.q_mean_linear.weight)
            xavier_uniform_(self.q_cov_linear.weight)

        if self.proj_bias:
            constant_(self.k_mean_linear.bias, 0.)
            constant_(self.k_cov_linear.bias, 0.)

            constant_(self.v_mean_linear.bias, 0.)
            constant_(self.v_cov_linear.bias, 0.)

            if self.kq_same is False:
                constant_(self.q_mean_linear.bias, 0.)
                constant_(self.q_cov_linear.bias, 0.)

            constant_(self.out_mean_proj.bias, 0.)
            constant_(self.out_cov_proj.bias, 0.)

    # def forward(self, q, k, v, mask, zero_pad):
    def forward(self, q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, mask, atten_type,zero_pad):

        bs = q_mean.size(0)

        # perform linear operation and split into h heads
        k_mean = self.k_mean_linear(k_mean).view(bs, -1, self.h, self.d_k)
        k_cov = self.k_cov_linear(k_cov).view(bs, -1, self.h, self.d_k)

        if self.kq_same is False:
            q_mean = self.q_mean_linear(q_mean).view(bs, -1, self.h, self.d_k)
            q_cov = self.q_cov_linear(q_cov).view(bs, -1, self.h, self.d_k)
        else:
            q_mean = self.k_mean_linear(q_mean).view(bs, -1, self.h, self.d_k)
            q_cov = self.k_cov_linear(q_cov).view(bs, -1, self.h, self.d_k)

        # v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        v_mean = self.v_mean_linear(v_mean).view(bs, -1, self.h, self.d_k)
        v_cov = self.v_cov_linear(v_cov).view(bs, -1, self.h, self.d_k)

        k_mean = k_mean.transpose(1, 2)
        q_mean = q_mean.transpose(1, 2)
        v_mean = v_mean.transpose(1, 2)
        k_cov = k_cov.transpose(1, 2)
        q_cov = q_cov.transpose(1, 2)
        v_cov = v_cov.transpose(1, 2)


        # calculate attention using function we will define next
        gammas = self.gammas
        if(atten_type == 'w2'):
            scores_mean, scores_cov = uattention(q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, self.d_k,
                        mask, self.dropout, zero_pad,gammas)
        elif(atten_type == 'dp'):
            scores_mean, scores_cov = attention(q_mean, q_cov, k_mean, k_cov, v_mean, v_cov,self.d_k,
                        mask, self.dropout, zero_pad,gammas)
        # concatenate heads and put through final linear layer
        concat_mean = scores_mean.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)
        concat_cov = scores_cov.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)

        output_mean = self.out_mean_proj(concat_mean)
        output_cov  = self.out_cov_proj(concat_cov)

        return output_mean, output_cov



# def attention(q, k, v, d_k, mask, dropout, zero_pad):
#     """
#     This is called by Multi-head atention object to find the values.
#     """
#     # d_k: 每一个头的dim
#     scores = torch.matmul(q, k.transpose(-2, -1)) / \
#         math.sqrt(d_k)  # BS, 8, seqlen, seqlen
#     bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

#     scores.masked_fill_(mask == 0, -1e32)
#     scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
#     # print(f"before zero pad scores: {scores.shape}")
#     # print(zero_pad)
#     if zero_pad:
#         pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
#         scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
#     scores = dropout(scores)
#     output = torch.matmul(scores, v)

#     return output

def attention(q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, d_k, mask, dropout, zero_pad, gamma):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores_mean = torch.matmul(q_mean, k_mean.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    scores_cov = torch.matmul(q_cov, k_cov.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen


    bs, head, seqlen = scores_mean.size(0), scores_mean.size(1), scores_mean.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():

        # scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_mean_ = scores_mean.masked_fill(mask == 0, -1e32)
        scores_cov_ = scores_cov.masked_fill(mask == 0, -1e32)
        
        # scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_mean_ = F.softmax(scores_mean_, dim=-1)  # BS,8,seqlen,seqlen
        scores_cov_ = F.softmax(scores_cov_, dim=-1)  # BS,8,seqlen,seqlen

        # scores_ = scores_ * mask.float().to(device) # 结果和上一步一样
        scores_mean_ = scores_mean_ * mask.float().to(device) # 结果和上一步一样
        scores_cov_ = scores_cov_ * mask.float().to(device) # 结果和上一步一样

        # distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        distcum_scores_mean = torch.cumsum(scores_mean_, dim=-1)  # bs, 8, sl, sl
        distcum_scores_cov = torch.cumsum(scores_cov_, dim=-1)  # bs, 8, sl, sl

        # disttotal_scores = torch.sum(
        #     scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
        disttotal_scores_mean = torch.sum(
            scores_mean_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
        disttotal_scores_cov = torch.sum(
            scores_cov_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
        

        # print(f"distotal_scores: {disttotal_scores}")
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
        # bs, 8, sl, sl positive distance

        # dist_scores = torch.clamp(
        #     (disttotal_scores-distcum_scores)*position_effect, min=0.) # score <0 时，设置为0
        dist_scores_mean = torch.clamp(
            (disttotal_scores_mean-distcum_scores_mean)*position_effect, min=0.) # score <0 时，设置为0
        dist_scores_cov = torch.clamp(
            (disttotal_scores_cov-distcum_scores_cov)*position_effect, min=0.) # score <0 时，设置为0
        
        # dist_scores = dist_scores.sqrt().detach()
        dist_scores_mean = dist_scores_mean.sqrt().detach()
        dist_scores_cov = dist_scores_cov.sqrt().detach()

    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
   
    total_effect_mean = torch.clamp(torch.clamp(
        (dist_scores_mean*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
    total_effect_cov = torch.clamp(torch.clamp(
        (dist_scores_cov*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分

    # scores = scores * total_effect
    scores_mean = scores_mean * total_effect_mean
    scores_cov = scores_cov * total_effect_cov


    # scores.masked_fill_(mask == 0, -1e32)
    scores_mean.masked_fill_(mask == 0, -1e32)
    scores_cov.masked_fill_(mask == 0, -1e32)

    # scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    scores_mean = F.softmax(scores_mean, dim=-1) 
    scores_cov = F.softmax(scores_cov, dim=-1) 

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        # scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
        scores_mean = torch.cat([pad_zero, scores_mean[:, :, 1:, :]], dim=2) # 第一行score置0
        scores_cov = torch.cat([pad_zero, scores_cov[:, :, 1:, :]], dim=2) # 第一行score置0
    # scores = dropout(scores)
    scores_mean = dropout(scores_mean)
    scores_cov = dropout(scores_cov)

    output_mean = torch.matmul(scores_mean, v_mean)
    output_cov = torch.matmul(scores_cov, v_cov)


    return output_mean, output_cov

# def uattention(q, k, v, d_k, mask, dropout, zero_pad):
def uattention(q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, d_k, mask, dropout, zero_pad, gamma):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = (-wasserstein_distance_matmul(q_mean, q_cov, k_mean, k_cov))/ \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device) # 结果和上一步一样
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
        # print(f"distotal_scores: {disttotal_scores}")
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.) # score <0 时，设置为0
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分

    scores = scores * total_effect


    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    scores = dropout(scores)

    output_mean = torch.matmul(scores, v_mean)
    output_cov = torch.matmul(scores ** 2, v_cov)

    return output_mean, output_cov

def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):  # Equation (4)
    mean1_2 = torch.sum(mean1**2, -1, keepdim=True) # BS, heads, seqlen, 1
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True) # BS, heads, seqlen, 1
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2) # BS, heads, seqlen, seqlen
    #ret = torch.clamp(-2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2), min=1e-24)
    #ret = torch.sqrt(ret)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    #cov_ret = torch.clamp(-2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2), min=1e-24)
    #cov_ret = torch.sqrt(cov_ret)
    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2)

    return ret + cov_ret # BS, heads, seqlen, seqlen

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)