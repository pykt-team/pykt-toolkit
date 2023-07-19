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

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

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

class parKT(nn.Module):

    def __init__(self, n_question, n_pid, 
            d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, seq_len=200, 
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768, num_rgap=None, num_sgap=None, num_pcount=None, num_it=None, lambda_w1=0.05, lambda_w2=0.05, lamdba_guess=0.3, lamdba_slip=0.5, cf_weight1=1, cf_weight2=1, c_weight=0.3, t_weight=0.3):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "parkt"
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
        self.gelu = GELU()
        self.cf_weight1 = cf_weight1
        self.cf_weight2 = cf_weight2

        self.lamdba_w1 = lambda_w1
        self.lambda_w2 = lambda_w2

        self.lamdba_guess = lamdba_guess
        self.lamdba_slip = lamdba_slip

        self.num_rgap = num_rgap
        self.num_sgap = num_sgap
        self.num_pcount = num_pcount
        self.num_it = num_it
        print(f"num_it:{num_it}")

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
        
        if emb_type.find("rnn") != -1:
            if emb_type.find("time") != -1:
                self.emb_lstm = LSTM(embed_l, embed_l, batch_first=True)
                self.time_emb = timeGap(num_rgap, num_sgap, num_pcount, d_model)
                if emb_type.find("fr") != -1:
                    self.tanh = nn.Tanh()
                    self.beta = torch.nn.Parameter(torch.randn(1))
                    if emb_type not in ["qid_rnn_time_fr"]:
                        self.it_emb = nn.Embedding(self.num_it+10, embed_l)
                    # self.it_linear = nn.Linear(embed_l*2, 1,bias=True)
                    self.it_linear = nn.Linear(embed_l, 1,bias=True)
            else:
                self.emb_lstm = LSTM(embed_l, embed_l, batch_first=True)

            if emb_type.find("xy") != -1:
                self.y_emb_lstm = LSTM(embed_l, embed_l, batch_first=True)
            if emb_type.find("bi") != -1:
                self.bi_emb_lstm = LSTM(embed_l, embed_l, batch_first=True, bidirectional=True)
                # self.bi_linear = nn.Linear(embed_l*2, embed_l)

        if emb_type.find("bay") != -1 or emb_type.find("augment") != -1:
            self.guess = nn.Linear(embed_l+self.num_pcount, 1)
            self.slip = nn.Linear(embed_l+self.num_rgap+self.num_sgap+self.num_pcount, 1)

        # Architecture Object. It contains stack of attention block
        
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=emb_type)

        if emb_type.find("irt") != -1:
            self.out = nn.Sequential(
                nn.Linear(embed_l*2,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim2, 1)
            )
            self.problem_disc = torch.nn.Embedding(self.n_pid, 1)
            # self.problem_disc = torch.nn.Embedding(self.n_pid, embed_l)
        elif emb_type.find("sahp") != -1:
            self.start_layer = nn.Sequential(
                nn.Linear(self.d_model, self.d_model, bias=True),
                self.gelu
            )

            self.converge_layer = nn.Sequential(
                nn.Linear(self.d_model, self.d_model, bias=True),
                self.gelu
            )

            self.decay_layer = nn.Sequential(
                nn.Linear(self.d_model, self.d_model, bias=True)
                ,nn.Softplus(beta=10.0)
            )

            self.intensity_layer = nn.Sequential(
                nn.Linear(self.d_model*2, 1, bias = True)
                ,nn.Sigmoid()
            )
        elif emb_type.find("pt") != -1:
            self.out = nn.Sequential(
                nn.Linear(embed_l*2,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim2, 1)
            )

            self.t_out = nn.Sequential(
                nn.Linear(embed_l*2,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim2, 1)
            )
        elif emb_type.find("dual") != -1:
            self.c_weight = nn.Linear(d_model, d_model)
            self.t_weight = nn.Linear(d_model, d_model)
            self.time_emb = timeGap(num_rgap, num_sgap, num_pcount, d_model)
            self.model2 = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=emb_type)
            self.out = nn.Sequential(
                nn.Linear(embed_l*2,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim2, 1)
            )
        elif emb_type.find("trip") != -1:
            self.c_weight = c_weight
            self.t_weight = t_weight
            self.time_emb = timeGap(num_rgap, num_sgap, num_pcount, d_model)
            self.model2 = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=emb_type)
            self.model3 = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=emb_type)   
            self.model4 = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=emb_type)   
            # self.model5 = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
            #                         d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, emb_type=emb_type)
            self.outlinear = nn.Linear(4*embed_l, embed_l)         
            self.out = nn.Sequential(
                nn.Linear(embed_l*2,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim2, 1)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(embed_l*2,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim2, 1)
            )

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
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def state_decay(self, converge_point, start_point, omega, duration_t):
        # * element-wise product
        cell_t = torch.tanh(converge_point + (start_point - converge_point) * torch.exp(- omega * duration_t))
        return cell_t

    def _instance_cl_one_pair_contrastive_learning(self, cl_batch, q_embed_data, qa_embed_data, intent_ids=None):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        """
        cl_sequence_output = self.model(q_embed_data, qa_embed_data)
        input_combined = torch.cat((cl_sequence_output, q_embed_data), -1)
        cl_sequence_output,_ = self.lstm_layer(input_combined)
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

    def forward(self, dcur, qtest=False, train=False, dgaps=None):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        # print(f"q_data:{q_data}")
        target = torch.cat((r[:,0:1], rshft), dim=1)
        batch_size = q.size(0)

        emb_type = self.emb_type

        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
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
            if emb_type.find("rnn") != -1:
                if emb_type.find("time") != -1:
                    t, tshft = dcur["tseqs"].long(), dcur["shft_tseqs"].long()
                    t_data = torch.cat((t[:,0:1], tshft), dim=1)
                    t_data = t_data.double() / 1000
                    rg, sg, p = dgaps["rgaps"].long(), dgaps["sgaps"].long(), dgaps["pcounts"].long()
                    rgshft, sgshft, pshft = dgaps["shft_rgaps"].long(), dgaps["shft_sgaps"].long(), dgaps["shft_pcounts"].long()

                    r_gaps = torch.cat((rg[:, 0:1], rgshft), dim=1)
                    s_gaps = torch.cat((sg[:, 0:1], sgshft), dim=1)
                    pcounts = torch.cat((p[:, 0:1], pshft), dim=1)
                    # print(f"q_embed_data:{q_embed_data.shape}")
                    rgap, sgap, pcount, temb = self.time_emb(r_gaps, s_gaps, pcounts)
                    qt_embed_data = q_embed_data + temb
                    if emb_type.find("ytime") != -1:
                        qa_embed_data = qa_embed_data + temb

                if emb_type.find("time") != -1:  
                    query, (hidden_state,cell) = self.emb_lstm(qt_embed_data)
                    # print(f"query:{query}")
                else:
                    query, (hidden_state,cell) = self.emb_lstm(q_embed_data)
                    
                if emb_type.find("fr") != -1:
                    if emb_type in ["qid_rnn_time_fr"]:
                        # follow hawkesKT
                        relu = nn.ReLU()
                        delta_t = relu(self.it_linear(temb))
                        forget_rate = torch.exp(-(self.beta + 1) * delta_t).view(batch_size,temb.size(1),1)
                if emb_type.find("time") != -1:
                    if emb_type.find("fr") == -1:
                        if train and emb_type.find("augment") != -1:
                            aug_d_output = self.model(aug_query,aug_qa_embed_data, forget_rate=None) 
                            aug_input_combined = torch.cat((aug_d_output, aug_query), -1)
                            aug_output = self.out(aug_input_combined).squeeze(-1)
                            # print(f"aug_output: {aug_output.shape}")
                        elif emb_type.find("sahp") != -1:
                            d_output = self.model(query, qa_embed_data, time_step=t_data) 
                        elif emb_type.find("dual") != -1:
                            t_output = self.model2(temb, qa_embed_data) # 计算时间信息和基本信息的attention？
                            d_output = self.model(query, qa_embed_data) 
                        elif emb_type.find("trip") != -1:
                            qa_output = self.model2(temb, qa_embed_data)
                            ta_output = self.model3(temb, qa_embed_data)
                            qt_output = self.model4(q_embed_data, temb) # 计算时间信息和基本信息的attention
                            # tq_output = self.model5(temb, q_embed_data) # 计算时间信息和基本信息的attention                            
                            d_output = self.model(query, qa_embed_data) 
                        else:
                            d_output = self.model(query, qa_embed_data) 
                            # print(f"d_output:{d_output}")
                    else:
                        d_output = self.model(query*forget_rate, qa_embed_data*forget_rate, forget_rate=None)
                else:
                    d_output = self.model(query, qa_embed_data)  

                if emb_type.find("irt") != -1:
                    # follow ijcai
                    problem_diff = pid_embed_data
                    problem_disc = self.problem_disc(pid_data)
                    input_x = (d_output - problem_diff) * problem_disc * 10
                    input_x = torch.cat((input_x, q_embed_data), -1)
                    output = self.out(input_x).squeeze(-1)
                    # print(f"output:{output.shape}")
                elif emb_type.find("sahp") != -1:
                    embed_info = d_output
                    # print(f"embed_info:{torch.min(embed_info), torch.max(embed_info)}")
                    self.start_point = self.start_layer(embed_info)
                    # print(f"start_point:{torch.min(self.start_point), torch.max(self.start_point)}")
                    self.converge_point = self.converge_layer(embed_info)
                    # print(f"converge_point:{torch.min(self.converge_point), torch.max(self.converge_point)}")
                    self.omega = self.decay_layer(embed_info)
                    # print(f"omega:{torch.min(self.omega), torch.max(self.omega)}")
                    cell_t = self.state_decay(self.converge_point, self.start_point, self.omega, s_gaps[:, :, None])
                    # print(f"cell_t:{cell_t.shape}")
                    input_combined = torch.cat((cell_t, query), -1)
                    output = self.intensity_layer(input_combined).squeeze(-1)    
                    # print(f"output:{output}")
                    # print(f"output:{torch.min(output), torch.max(output)}")
                elif emb_type.find("dual") != -1:
                    w = torch.sigmoid(self.c_weight(d_output) + self.t_weight(t_output)) # w = sigmoid(基本信息编码 + 时间信息编码)，每一维设置为0-1之间的数值
                    d_output = w * d_output + (1 - w) * t_output # 每一维加权平均后的综合信息
                    input_combined = torch.cat((d_output, query), -1)
                    output = self.out(input_combined).squeeze(-1)
                elif emb_type.find("trip") != -1:
                    # w = torch.sigmoid(self.c_weight(d_output) + self.t_weight(t_output)) # w = sigmoid(基本信息编码 + 时间信息编码)，每一维设置为0-1之间的数值
                    # d_output = self.c_weight * d_output + self.t_weight * t_output + (1 - self.c_weight - self.t_weight)*qt_output # 每一维加权平均后的综合信息
                    d_output = self.outlinear(torch.cat((qa_output, ta_output, qt_output, d_output), -1))
                    input_combined = torch.cat((d_output, query), -1)
                    output = self.out(input_combined).squeeze(-1)
                else:
                    # pad_zero = torch.zeros(d_output.size(0), 1, d_output.size(2)).to(device)
                    # query = torch.cat([query[:, 1:, :], pad_zero], dim=1) # 第一行score置0
                    input_combined = torch.cat((d_output, query), -1)
                    output = self.out(input_combined).squeeze(-1)

                if emb_type.find("bi") != -1:
                    bs, seqlen = qt_embed_data.size(0), qt_embed_data.size(1)
                    bi_query, (hidden, cell) = self.bi_emb_lstm(qt_embed_data)
                    # print(f"hidden:{hidden.shape, cell.shape}")
                    # bi_query = self.bi_linear(bi_query)
                    bi_query_ = torch.reshape(bi_query,(-1,2,qt_embed_data.size(2)))
                    bi_query_ = torch.mean(bi_query_,dim=1).reshape(bs, seqlen, -1)
                    # print(f"bi_query:{bi_query.shape}")
                    bi_d_output = self.model(bi_query_, qa_embed_data)     
                    bi_input_combined = torch.cat((bi_d_output, bi_query_), -1)   
                    bi_output = self.out(bi_input_combined).squeeze(-1)

            else:
                d_output = self.model(q_embed_data, qa_embed_data)
                input_combined = torch.cat((d_output, q_embed_data), -1)
                output = self.out(input_combined).squeeze(-1)    
            
            if emb_type.find("sahp") != -1:
                preds = output
            else:
                preds = self.m(output)
                # print(f"preds:{preds}")

        if train:
            if emb_type not in ["qid_cl", "qid_mt", "qid_pvn", "qid_rnn_bi", "qid_rnn_time_augment", "qid_rnn_time_pt", "qid_birnn_time", "qid_birnn_time_pt"]:
                return preds, y2, y3
            else:
                if emb_type.find("augment") != -1:
                    bi_preds = self.m(aug_output)[:,1:]
                    sm = dcur["smasks"]
                    y = torch.masked_select(preds[:,1:], sm)
                    aug_y = torch.masked_select(bi_preds, sm)
                    # print(f"y:{y.shape}")
                    t = torch.masked_select(new_target[:,1:], sm)
                    # print(f"t:{t.shape}")
                    perturbation_loss = mse_loss(y, aug_y)
                    cl_losses = self.lamdba_w1 * (binary_cross_entropy(aug_y.double(), t.double())) + self.lambda_w2 * perturbation_loss
                elif emb_type.find("pt") != -1:
                    cl_losses = 0
                    t_label= dgaps["shft_tlabel"].double()
                    # print(f"t_label:{t_label}")
                    if emb_type.find("bi") != -1:
                        bi_query = bi_query.view(bs, seqlen, 2, -1)
                        output_ffw = bi_query[:,:,:1,:].view(bs, seqlen,-1)
                        output_bfw = bi_query[:,:,1:,:].view(bs,seqlen,-1)
                        zero_pad = torch.zeros(bs,1,self.d_model).to(device)
                        output_ffw = torch.cat((zero_pad, output_ffw),dim=1)[:,:-1,:]
                        # print(f"output_ffw:{output_ffw.shape}")
                        output_bfw = torch.cat((output_bfw, zero_pad),dim=1)[:,1:,:]
                        # print(f"output_bfw:{output_bfw.shape}")
                        t_output = self.t_out(torch.cat((output_ffw, output_bfw),dim=2)).squeeze(-1)
                        t_pred = self.m(t_output)[:,1:]
                        bi_preds = self.m(bi_output)[:,1:]
                        # print(f"bi_preds:{bi_preds.shape}")
                        sm = dcur["smasks"]
                        y = torch.masked_select(bi_preds, sm)
                        # print(f"y:{y.shape}")
                        t = torch.masked_select(rshft, sm)
                        cl_losses = self.cf_weight1 * (binary_cross_entropy(y.double(), t.double()))
                    else:
                        t_combined = torch.cat((d_output, temb), -1)
                        t_output = self.t_out(t_combined).squeeze(-1)
                        t_pred = self.m(t_output)[:,1:]
                    sm = dcur["smasks"]
                    ty = torch.masked_select(t_pred, sm)
                    # print(f"max_y:{torch.max(y)}")
                    tt = torch.masked_select(t_label, sm)
                    # print(f"cf_weight2:{self.cf_weight2}")
                    cl_losses += self.cf_weight2*binary_cross_entropy(ty.double(), tt.double())
                elif emb_type.find("bi") != -1:
                    bi_preds = self.m(bi_output)[:,1:]
                    # print(f"bi_preds:{bi_preds.shape}")
                    sm = dcur["smasks"]
                    y = torch.masked_select(bi_preds, sm)
                    # print(f"y:{y.shape}")
                    t = torch.masked_select(rshft, sm)
                    # print(f"t:{t.shape}")
                    cl_losses = self.cf_weight * (binary_cross_entropy(y.double(), t.double()))
                return preds, y2, y3, cl_losses
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

        if self.emb_type.find("sahp") != -1:
            self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)
            # self.position_emb = BiasedPositionalEmbedding(d_model=self.d_model, max_len=seq_len)
        else:
            self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data, forget_rate=None, time_step=None):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)
        if self.emb_type.find("sahp") != -1:
            # q_posemb = self.position_emb(q_embed_data, time_step)
            q_posemb = self.position_emb(q_embed_data)
        else:
            q_posemb = self.position_emb(q_embed_data)
        # print(f"q_posemb:{q_posemb.shape}")
        # print(f"q_embed_data:{q_embed_data.shape}")
        q_embed_data = q_embed_data + q_posemb
        if self.emb_type.find("sahp") != -1:
            # qa_posemb = self.position_emb(qa_embed_data, time_step)
            qa_posemb = self.position_emb(qa_embed_data)
        else:
            qa_posemb = self.position_emb(qa_embed_data)            
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed
        # print(f"x:{x.shape}")
        # print(f"y:{y.shape}")

        # encoder
        
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True,forget_rate=forget_rate) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
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


class BiasedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

        self.Wt = nn.Linear(1, d_model // 2)

    def forward(self, x, interval):
        # print(f"interval:{interval.shape}")
        interval = interval.float()
        phi = self.Wt(interval.unsqueeze(-1))
        aa = len(x.size())
        if aa > 1:
            length = x.size(1)
        else:
            length = x.size(0)

        arc = (self.position[:length] * self.div_term).unsqueeze(0)

        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe
