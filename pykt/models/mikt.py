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
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy, mse_loss
from .simplekt_utils import NCELoss
from random import choice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class timeGap(nn.Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_size) -> None:
        super().__init__()
        self.rgap_eye = torch.eye(num_rgap)
        self.sgap_eye = torch.eye(num_sgap)
        self.pcount_eye = torch.eye(num_pcount)

        input_size = num_rgap + num_sgap + num_pcount

        self.time_emb = nn.Linear(input_size, emb_size, bias=False)

    def forward(self, rgap, sgap, pcount):
        rgap = self.rgap_eye[rgap].to(device)
        sgap = self.sgap_eye[sgap].to(device)
        pcount = self.pcount_eye[pcount].to(device)

        tg = torch.cat((rgap, sgap, pcount), -1)
        tg_emb = self.time_emb(tg)

        return rgap, sgap, pcount, tg_emb

class MIKT(nn.Module):

    def __init__(self, n_question, n_pid, 
            d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, seq_len=200, 
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768, num_rgap=None, num_sgap=None, num_pcount=None, cf_weight1=0.3, cf_weight2=0.3):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "mikt"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = n_question
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.d_model = d_model

        self.num_rgap = num_rgap
        self.num_sgap = num_sgap
        self.num_pcount = num_pcount
        print(f"temporal info:num_rgap:{self.num_rgap}, num_sgap:{self.num_sgap}, num_pcount:{self.num_pcount}")

        self.cf_weight1 = cf_weight1
        self.cf_weight2 = cf_weight2

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
        
        self.emb_lstm = LSTM(embed_l, embed_l, batch_first=True)
        # self.yemb_lstm = LSTM(embed_l, embed_l, batch_first=True)
        self.time_emb = timeGap(num_rgap, num_sgap, num_pcount, d_model)
        # self.out_lstm = LSTM(embed_l*2, embed_l, batch_first=True)

        if emb_type.find("enh") != -1:
            # self.init_y_pre = nn.Parameter(torch.randn(1, embed_l))
            self.model2 = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=emb_type)
        
            # self.model3 = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
            #                 d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=emb_type)

        #     self.comb = nn.Sequential(
        #     nn.Linear(embed_l*2,
        #             final_fc_dim), nn.Tanh(), nn.Dropout(self.dropout)
        # )

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=emb_type)
        self.out = nn.Sequential(
            nn.Linear(embed_l*2,
                    final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )

        if emb_type.find("pt") != -1: 
            self.t_out = nn.Sequential(
                nn.Linear(embed_l*2,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim2, 1)
            )
            if emb_type.find("doublept") != -1: 
                self.cit_out = nn.Sequential(
                    nn.Linear(embed_l*2,
                            final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                    nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
                    ), nn.Dropout(self.dropout),
                    nn.Linear(final_fc_dim2, 1)
                )
        
        self.relu = nn.ReLU()
        self.m = nn.Sigmoid()

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
            target_embed_data = self.qa_embed(target)
            qa_embed_data = target_embed_data+q_embed_data
        return q_embed_data, qa_embed_data, target_embed_data

    def forward(self, dcur, qtest=False, train=False, dgaps=None):
        q, c, r = dcur["qseqs"].long().to(device), dcur["cseqs"].long().to(device), dcur["rseqs"].long().to(device)
        qshft, cshft, rshft = dcur["shft_qseqs"].long().to(device), dcur["shft_cseqs"].long().to(device), dcur["shft_rseqs"].long().to(device)
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        # print(f"q_data:{q_data.shape}")
        target = torch.cat((r[:,0:1], rshft), dim=1)
        # print(f"target:{targets.shape}")
        batch_size = q.size(0)

        emb_type = self.emb_type

        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data, target_embed_data = self.base_emb(q_data, target)
        if self.n_pid > 0 and emb_type.find("norasch") == -1: # have problem id
            if emb_type.find("aktrasch") == -1:
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
        if emb_type.startswith("qid"):
            rg, sg, p = dgaps["rgaps"].long(), dgaps["sgaps"].long(), dgaps["pcounts"].long()
            rgshft, sgshft, pshft = dgaps["shft_rgaps"].long(), dgaps["shft_sgaps"].long(), dgaps["shft_pcounts"].long()

            r_gaps = torch.cat((rg[:, 0:1], rgshft), dim=1)
            # print(f"r_gaps:{r_gaps.shape}")
            s_gaps = torch.cat((sg[:, 0:1], sgshft), dim=1)
            pcounts = torch.cat((p[:, 0:1], pshft), dim=1)
            # print(f"q_embed_data:{q_embed_data.shape}")
            rgap, sgap, pcount, temb = self.time_emb(r_gaps, s_gaps, pcounts)
            # print(f"temb:{temb.shape}")
            qt_embed_data = q_embed_data + temb
            # print(f"qt_emb:{qt_embed_data}")
            query, (hidden_state,cell) = self.emb_lstm(qt_embed_data)
            # qa_embed_data, (hidden_state,cell) = self.yemb_lstm(qa_embed_data+temb)
            # print(f"query:{query}")

            if emb_type.find("enh") != -1:
                # # d_output, y_pre = self.model(qa_embed_data, target_embed_data, only_hist=True)
                # pre_d_output, y_pre = self.model(qa_embed_data, target_embed_data)
                # # print(f"y_pre:{y_pre.shape}")
                # init_y_pre = self.init_y_pre.repeat(q_embed_data.size(0),1,1)
                # y_pre = torch.cat((init_y_pre,pre_d_output[:,:-1,:]),dim=1)
                # qa_embed_data = y_pre+qa_embed_data
                # # query_ = query+y_pre
                # d_output, y_pre = self.model(query, qa_embed_data)

                # previous version
                # pre_d_output, y_pre = self.model(qa_embed_data, target_embed_data)
                # pre_d_output, y_pre = self.model2(qa_embed_data, target_embed_data)
                pre_d_output, y_pre = self.model2(target_embed_data, temb)
                pre_d_output = pre_d_output + qa_embed_data
                # print(f"y_pre:{y_pre.shape}")
                # init_y_pre = self.init_y_pre.repeat(q_embed_data.size(0),1,1)
                # y_pre = torch.cat((init_y_pre,pre_d_output[:,:-1,:]),dim=1)
                # query_ = y_pre+query
                # qa_embed_data = pre_d_output+qa_embed_data
                # qa_embed_data = self.comb(torch.cat((qa_embed_data, pre_d_output),dim=2))
                # query_ = query+y_pre
                # query_ = self.comb(torch.cat((query, y_pre),dim=2))
                # d_output, y_pre = self.model(query, qa_embed_data)
                # d_output, y_pre = self.model(query, pre_d_output)
                d_output, y_pre = self.model(query, pre_d_output)

                # pre_y_output, y_pre = self.model2(target_embed_data, temb)
                # init_y_pre = self.init_y_pre.repeat(q_embed_data.size(0),1,1)
                # pre_y_output = torch.cat((init_y_pre, pre_y_output[:,:-1,:]),dim=1)
                # pre_x_output, y_pre = self.model3(target_embed_data, q_embed_data)
                # pre_x_output = torch.cat((init_y_pre, pre_y_output[:,:-1,:]),dim=1)
                # pre_y_output = pre_y_output + qa_embed_data
                # query_ = pre_x_output + query
                # d_output, y_pre = self.model(query_, pre_y_output)

                # d_output, y_pre = self.model(query, qa_embed_data, temb=temb)                
            else:
                d_output, y_pre = self.model(query, qa_embed_data)


            input_combined = torch.cat((d_output, query), -1)
            output = self.out(input_combined).squeeze(-1)
            preds = self.m(output)
            # print(f"preds:{preds}")

        if train:
            if emb_type.find("pt") != -1:
                cl_losses = 0
                # t_label= dgaps["shft_tlabel"].double()
                t_label= dgaps["shft_pretlabel"].double()
                # print(f"t_label:{t_label}")
                t_combined = torch.cat((d_output, temb), -1)
                t_output = self.t_out(t_combined).squeeze(-1)
                t_pred = self.m(t_output)[:,1:]
                # print(f"t_pred:{t_pred}")
                sm = dcur["smasks"]
                ty = torch.masked_select(t_pred, sm)
                # print(f"min_y:{torch.min(ty)}")
                tt = torch.masked_select(t_label, sm)
                # print(f"min_t:{torch.min(tt)}")
                t_loss = binary_cross_entropy(ty.double(), tt.double())
                # t_loss = mse_loss(ty.double(), tt.double())
                # print(f"t_loss:{t_loss}")
                cl_losses += self.cf_weight1 * t_loss
                # print(f"cl_losses:{cl_losses}")

                if emb_type.find("doublept") != -1:
                    cit_label= dgaps["shft_citlabel"].double()
                    cit_output = self.cit_out(t_combined).squeeze(-1)
                    cit_pred = self.m(cit_output)[:,1:]
                    city = torch.masked_select(cit_pred, sm)
                    citt = torch.masked_select(cit_label, sm)
                    cit_loss = binary_cross_entropy(city.double(), citt.double())
                    # cit_loss = mse_loss(city.double(), citt.double())
                    # print(f"cit_loss:{cit_loss}")
                    cl_losses += self.cf_weight2 * cit_loss
                    # print(f"cl_losses:{cl_losses}")
                return preds, y2, y3, cl_losses
            # elif emb_type.find("label") != -1:
            #     sm = dcur["smasks"]
            #     t, tshft = dcur["tseqs"].long(), dcur["shft_tseqs"].long()
            #     t_data = torch.cat((t[:,0:1], tshft), dim=1)
            #     t_data = t_data.long() // 1000 // 60

            #     cl_losses = 0
            #     post_data = t_data[:,1:]  
            #     pret_data = t_data[:,:-1]

            #     it_label = torch.where(post_data - pret_data > 43200, 43200,post_data - pret_data)
            #     it_label = torch.masked_select(it_label,sm)
            #     print(f"it_label:{it_label.shape}")
            #     print(f"it_label:{torch.max(it_label)}")
            #     t_combined = torch.cat((temb, d_output), -1)
            #     t_output = self.t_out(t_combined).squeeze(-1)
            #     t_pred = self.relu(t_output)[:,1:]
            #     print(f"t_pred:{torch.max(t_pred)}")
            #     it_pred = t_pred - pret_data
            #     it_pred = torch.masked_select(it_pred,sm)
            #     print(f"t_pred:{it_pred.shape}")
            #     print(f"it_pred:{torch.max(it_pred)}")
            #     cl_losses = (it_pred - it_label)/it_label
            #     cl_losses = torch.sum(cl_losses)
            #     print(f"cl_losses:{cl_losses}")
            #     return preds, y2, y3, cl_losses
            else:
                return preds, y2, y3
        else:
            if qtest:
                return preds, output
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
        self.emb_type = emb_type

        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
            for _ in range(n_blocks)
        ])

        if self.emb_type.find("enh") != -1:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                    d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks)
            ])

        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data, forget_rate=None, time_step=None, only_hist=False, temb=None):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb

        qa_posemb = self.position_emb(qa_embed_data)            
        qa_embed_data = qa_embed_data + qa_posemb

        # t_posemb = self.position_emb(temb)            
        # temb = temb + t_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data


        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed
        # print(f"x:{x.shape}")
        # print(f"y:{y.shape}")

        # encoder
        # y_pre = None     
        # if only_hist:
        #     for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
        #         y_pre = block(mask=1, query=y, key=y, values=y) # yt^   

        # else:
        #     for block in self.blocks_2:
        #         x = block(mask=0, query=x, key=x, values=y, apply_pos=True,forget_rate=forget_rate) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
        #         # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
        #         # print(x[0,0,:])

        y_pre = None     
        # for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
        #     y = block(mask=1, query=y, key=temb, values=y) # yt^   

        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True,forget_rate=forget_rate) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])

        return x, y_pre

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

    def forward(self, mask, query, key, values, apply_pos=True, forget_rate=None):
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
        # nopeek_mask = np.triu(
        #     np.ones((1, 1, seqlen, seqlen//2)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, forget_rate=forget_rate) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, forget_rate=forget_rate)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True,seq_len=200,emb_type="qid",init_eps = 1e-3,max_relative_positions=-1,position_buckets=-1):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.emb_type = emb_type

        if self.emb_type.find("disentangled_attention") != -1:
            self.attn = DisentangledSelfAttention(num_attention_heads=n_heads,hidden_size=d_model,hidden_dropout_prob=dropout,attention_probs_dropout_prob=dropout)
            self.max_relative_positions = max_relative_positions
            self.position_buckets = position_buckets
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        else:
            self.d_k = d_feature
            self.h = n_heads
            self.kq_same = kq_same
            self.seq_len = seq_len

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
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), bucket_size = self.position_buckets, max_position=self.max_relative_positions)
        return relative_pos

    def forward(self, q, k, v, mask, zero_pad, forget_rate=None):

        bs = q.size(0)

        if self.emb_type.find("disentangled_attention") != -1:#放在这里才能生效
            relative_pos = self.get_rel_pos(q, query_states=None, relative_pos=None)
            concat = self.attn(q,k,v,mask,zero_pad=zero_pad,relative_pos=relative_pos)['hidden_states']
        else:
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            # calculate attention using function we will define next

            scores = attention(q, k, v, self.d_k,
                            mask, self.dropout, zero_pad, forget_rate=forget_rate, emb_type=self.emb_type)

            # concatenate heads and put through final linear layer
            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, forget_rate=None, emb_type="qid"):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    # print(f"emb_type:{emb_type}")
    # if emb_type.find("sahp") != -1:
    #     scores = torch.exp(torch.matmul(q, k.transpose(-2, -1))) \
    #         / math.sqrt(d_k)
    # else:
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(3)
    if forget_rate is not None:
        forget_rate = forget_rate.repeat(1, head,1,seqlen)
        # print(f"forget_rate:{forget_rate.shape}")
        scores = scores*forget_rate
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)

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
