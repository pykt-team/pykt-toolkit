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
from .simplekt_utils import NCELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class simpleKT_SR(nn.Module):
    def create_mlp(self, ln_in, ln_out):
        '''
        Create MultiLayer Perceptron (MLP) layers
        '''
        LL = nn.Linear(ln_in, ln_out)

        #initiate weights randomly
        mean = 0.0
        std_dev = np.sqrt(2 / (ln_in + ln_out))
        W = np.random.normal(mean, std_dev, size=(ln_out, ln_in)).astype(np.float32)
        LL.weight.data = torch.tensor(W, requires_grad=True)

        std_dev = np.sqrt(1 / ln_out)
        bt = np.random.normal(mean, std_dev, size=ln_out).astype(np.float32)
        LL.bias.data = torch.tensor(bt, requires_grad=True)

        return LL

    def __init__(self, n_question, n_pid, 
            d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4, seq_len=200, 
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768, augment_type="random", tao=0.2, gamma=0.7, beta=0.2, n_views=2,cf_weight=0.1, seq_representation_instancecl_type="mean",temperature=1.0):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "simplekt_sr"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.seq_representation_instancecl_type = seq_representation_instancecl_type
        self.temperature = temperature
        self.cf_weight = cf_weight
        self.ce_loss = CrossEntropyLoss()

        embed_l = d_model
        if self.n_pid > 0:
            if emb_type.find("scalar") != -1:
                # print(f"question_difficulty is scalar")
                self.difficult_param = nn.Embedding(self.n_pid+1, 1) # 题目难度
            elif emb_type.find("s3") != -1:
                self.weight_aap = nn.Parameter(torch.randn(embed_l, embed_l)).to(device)
                self.difficult_param = nn.Embedding(self.n_pid+1, embed_l)
            else:
                self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上
        
        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question+1, embed_l)
            if self.separate_qa: 
                    self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=emb_type)
        self.linear = nn.Linear(embed_l*2, embed_l)
        if emb_type.find("rnn") != -1:
            # self.i2h = self.create_mlp(d_model + embed_l, final_fc_dim)
            self.i2o = self.create_mlp(d_model + embed_l, 1)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )

        self.out_2 = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, self.n_question)
        )

        self.cf_criterion = NCELoss(self.temperature, device)

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def _instance_cl_one_pair_contrastive_learning(self, cl_batch, q_embed_data, qa_embed_data, intent_ids=None):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        """
        cl_sequence_output = self.model(q_embed_data, qa_embed_data)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        if self.seq_representation_instancecl_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        # if self.args.de_noise:
        #     cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids)
        # else:
        cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None)
        return cl_loss

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)
        batch_size = q.size(0)

        emb_type = self.emb_type

        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.n_pid > 0 and emb_type.find("norasch") == -1: # have problem id
            if emb_type.find("s3") != -1:
                q_embed_diff_data = self.q_embed_diff(q_data)
                pid_embed_data_ = torch.matmul(q_embed_data, self.weight_aap).view(q_embed_diff_data.size(0),q_embed_diff_data.size(1),1,q_embed_diff_data.size(2))
                # print(f"pid_embed_data_: {pid_embed_data_.shape}")
                pid_embed_data = self.difficult_param(pid_data).unsqueeze(-1)
                # print(f"pid_embed_data: {pid_embed_data.shape}")
                pid_embed_data = torch.matmul(pid_embed_data_, pid_embed_data).squeeze(3)
                # print(f"pid_embed_data: {pid_embed_data.shape}")
                sig = nn.Sigmoid()
                pid_embed_data = sig(pid_embed_data)
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder
            elif emb_type.find("aktrasch") == -1:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder
            else:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

                qa_embed_diff_data = self.qa_embed_diff(
                    target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
                qa_embed_data = qa_embed_data + pid_embed_data * \
                        (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        y2, y3 = 0, 0
        if emb_type in ["qid", "qidaktrasch", "qid_scalar", "qid_norasch","qid_filter", "qid_s3", "qid_cl", "qid_rnn", "qid_par", "qid_mt"]:
            if emb_type == "qid_par":
                bs = q_embed_data.size(0)
                seqlen = q_embed_data.size(1)
                dim = q_embed_data.size(2)
                nums = torch.arange(seqlen+1)[1:].unsqueeze(-1).to(device)
                hidden = torch.cumsum(q_embed_data,dim=1)/nums
                query = torch.cat((q_embed_data, hidden), 2)
                q_embed_data = self.linear(query)
            d_output = self.model(q_embed_data, qa_embed_data)
            if emb_type.find("rnn") != -1:
                # print(f"running qid_rnn")
                input_combined = torch.cat((d_output, q_embed_data), -1)
                # hidden = F.relu(self.i2h(input_combined))
                output = self.i2o(input_combined).squeeze(-1)
                m = nn.Sigmoid()
                preds = m(output)
            else:
                concat_q = torch.cat([d_output, q_embed_data], dim=-1)
                output = self.out(concat_q).squeeze(-1)
                m = nn.Sigmoid()
                preds = m(output)

            if emb_type == "qid_cl" and train:
                cl_losses = []
                cl_batches, cl_r_batches = dcur["cseqs_cl"], dcur["rseqs_cl"]
                # print(f"cl_batches:{cl_batches}")
                # print(f"cl_r_batchess:{cl_r_batches}")
                for idx,cl_batch in enumerate(cl_batches):
                    cl_batch = torch.cat(cl_batch, dim=0)
                    cl_batch = cl_batch.to(device)
                    target = torch.cat(cl_r_batches[idx], dim=0)
                    target = target.to(device)
                    q_embed_data, qa_embed_data = self.base_emb(cl_batch, target)
                    if self.n_pid > 0:
                        cl_q_batches = dcur["qseqs_cl"]
                        cl_q_batch = torch.cat(cl_q_batches[idx], dim=0)
                        pid_data = cl_q_batch.to(device)
                        q_embed_diff_data = self.q_embed_diff(cl_batch)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                        pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                        q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data  # uq *d_ct + c_ct # question encoder
                        qa_embed_diff_data = self.qa_embed_diff(target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
                        qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
                    cl_loss = self._instance_cl_one_pair_contrastive_learning(cl_batch, q_embed_data, qa_embed_data)
                    cl_losses.append(self.cf_weight * cl_loss)
            elif emb_type == "qid_mt" and train:
                cl_losses = []
                sm = dcur["smasks"].long()
                start = 0
                next_c_pred = self.out_2(concat_q)
                # print(f"next_c_pred:{next_c_pred.shape}")
                # print(f"cshft:{cshft.shape}")
                flag = sm[:,start:]==1
                cl_loss = self.ce_loss(next_c_pred[:,:-1,:][flag], cshft[:,start:][flag])
                cl_losses.append(self.cf_weight * cl_loss)
                # print(f"y2:{y2}")

                # next_c_pred = (next_c_pred[:,:-1,:] * one_hot(cshft.long(), self.n_question)).sum(-1)
                # print(f"next_c_pred:{next_c_pred.shape}")

        if train:
            if emb_type not in ["qid_cl", "qid_mt"]:
                return preds, y2, y3
            else:
                return preds, y2, y3, cl_losses
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len, emb_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'simplekt_sr'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
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
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same,emb_type):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same,emb_type=emb_type)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
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

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True,seq_len=200,emb_type="qid",init_eps = 1e-3):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.seq_len = seq_len
        self.emb_type = emb_type

        if self.emb_type.find("filter") != -1:
            self.mask = torch.tril(torch.ones(seq_len,seq_len), diagonal=-1).to(device)
            # self.add_avg = torch.tensor(np.arange(1,seq_len+1)).reshape(-1,1).to(device)
            # self.cross_interaction = nn.Parameter(torch.FloatTensor(np.zeros((seq_len,seq_len)))).to(device)
            self.cross_interaction = nn.Parameter(torch.FloatTensor(np.zeros((64,d_model,seq_len)))).to(device)
            # self.cross_interaction = nn.Parameter(torch.randn(64, d_model, seq_len, dtype=torch.float32) * 0.02).to(device)
            self.complex_weight_v = nn.Parameter(torch.randn(1, seq_len//2 + 1, d_model, 2, dtype=torch.float32) * 0.02)
            # self.complex_weight_q = nn.Parameter(torch.randn(1, seq_len//2 + 1, d_model, 2, dtype=torch.float32) * 0.02)
            self.out_dropout = nn.Dropout(dropout)
            # self.LayerNorm = LayerNorm(d_model, eps=1e-12)

            
            nn.init.uniform_(self.cross_interaction, -init_eps, init_eps)
            
            self.norm = nn.LayerNorm(d_model)
            # self.proj_in = nn.Sequential(nn.Linear(d_model, d_model),nn.GELU())
            self.proj_out = nn.Linear(d_model,d_model)    
        else:
            self.v_linear = nn.Linear(d_model, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
            if kq_same is False:
                self.q_linear = nn.Linear(d_model, d_model, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.proj_bias = bias
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
            self.complex_weight_v = nn.Parameter(torch.randn(1, seq_len//2 + 1, d_model, 2, dtype=torch.float32) * 0.02)

            self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):
        if self.emb_type.find("filter") != -1:
            #  jiahao
            # q = self.proj_in(q)
            # res, gate = q.chunk(2, dim = -1)
            # #sgu模块
            # gate = self.norm(gate)#batch_size,max_len,dim
            # gate = torch.matmul(self.cross_interaction*self.mask,gate)/self.add_avg#change result to mean
            # print(f"gate is {gate.shape}")
            # output = self.proj_out(gate*v) 

            # shyann_old_version
            # v = torch.fft.rfft(v, dim=1, norm='ortho')
            # v_weight = torch.view_as_complex(self.complex_weight_v)   
            # v = v * v_weight
            # v_fft = torch.fft.irfft(v, n=self.seq_len, dim=1, norm='ortho')
            # gate = self.norm(q)
            # gate = torch.matmul(self.cross_interaction*self.mask,gate)/self.add_avg#change result to mean
            # output = self.proj_out(torch.matmul(gate, v_fft)) 

            # batch_size = q.size(0)
            # v = torch.fft.rfft(v, dim=1, norm='ortho')
            # v_weight = torch.view_as_complex(self.complex_weight_v)   
            # v = v * v_weight
            # v_fft = torch.fft.irfft(v, n=self.seq_len, dim=1, norm='ortho')
            # # print(f"v_fft:{v_fft.shape}")
            # gate = self.norm(q)
            # # print(f"gate:{gate.shape}")
            # gate = torch.matmul(gate,self.cross_interaction[:batch_size,:,:]) * mask
            # # print(f"gate:{gate.shape}")
            # output = torch.matmul(gate, v_fft).squeeze(0)
            # output = self.proj_out(output) 

            # shyann_newest
            batch_size = q.size(0)
            v = torch.fft.rfft(v, dim=1, norm='ortho')
            v_weight = torch.view_as_complex(self.complex_weight_v)   
            v = v * v_weight
            v_fft = torch.fft.irfft(v, n=self.seq_len, dim=1, norm='ortho')
            gate = self.norm(q)
            # print(f"gate:{gate.shape}")
            gate = torch.matmul(gate,self.cross_interaction[:batch_size,:,:])
            gate.masked_fill_(self.mask == 0, -1e32)
            gate = F.softmax(gate, dim=-1) 
            pad_zero = torch.zeros(batch_size, 1, self.seq_len).to(device)
            gate = torch.cat([pad_zero, gate[:, 1:, :]], dim=1)
            # print(f"gate:{gate}")
            output = torch.matmul(gate, v_fft)
            output = self.proj_out(output) 
            print(f"output:{output.shape}")
        else:
            bs = q.size(0)

            # perform linear operation and split into h heads

            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            
            # # print(f"v_before:{v.dtype}")
            # v = torch.fft.rfft(v, dim=1, norm='ortho')
            # v_weight = torch.view_as_complex(self.complex_weight_v)
            # # v_weight_ = torch.tensor(v_weight,dtype=torch.float32).to(device)
            # # print(f"v_after:{v_weight_.dtype}")
            # v = v * v_weight
            # # print(f"v_after:{v.dtype}")
            # v_fft = torch.fft.irfft(v, self.seq_len, dim=1, norm='ortho')

            # v = self.v_linear(v_fft).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
            # print(f"v:{v.dtype}")

            # transpose to get dimensions bs * h * sl * d_model

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            # calculate attention using function we will define next
            scores = attention(q, k, v, self.d_k,
                            mask, self.dropout, zero_pad, self.complex_weight_v)

            # concatenate heads and put through final linear layer
            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)

            output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, complex_weight_v):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
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