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

class BAKTSimpleX(nn.Module):
    def __init__(self, n_question, n_pid, 
            d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4, seq_len=200, 
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", 
            emb_path="", pretrain_dim=768,
            m = 0.2, w=0.2, topk=5
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
        self.model_name = "bakt_simplex"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.loss1 = loss1
        self.loss2 = loss2
        self.start = start
        self.topk = topk
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.m = m
        self.w = w
        embed_l = d_model
        if self.n_pid > 0:
            if emb_type.find("scalar") != -1:
                # print(f"question_difficulty is scalar")
                self.difficult_param = nn.Embedding(self.n_pid+1, 1) # 题目难度
            else:
                self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上
        
        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                    self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )

        self.c_weight = nn.Linear(d_model, d_model)

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


    def cal_ccl_loss_weili(self, scores, output, sm, t=0.2):
        res = 0
        losses = []
        for j in range(0, scores.shape[0]):
            curoutput = output[j,:,:]
            seq_len = curoutput.shape[0] 
            curscores = scores[j,:,:]
            score_copy = curscores.clone()
            sort_score, index = torch.sort(score_copy, descending=True)
            # print(f"output: {output.shape}, index: {index.shape}")
            eye = torch.eye(seq_len)
            positive = eye[index[:, 0]]
    
            #positive = torch.load("/share_v1/sunweili/pykt-toolkit-dev/examples/tmp.pt").to(device)
            #相似度矩阵
            similarity_matrix = F.cosine_similarity(curoutput.unsqueeze(0), curoutput.unsqueeze(1), dim=-1).to(device)
            #print(similarity_matrix)
            mask = positive.long().to(device)
            #这步得到它的不同类的矩阵，不同类的位置为1
            #mask_no_sim = torch.ones_like(mask) - mask
            
            similarity_matrix = torch.exp(similarity_matrix / t)
            similarity_matrix = torch.tril(similarity_matrix, diagonal=-1)  # 只保留下三角
            #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0
            # print(f"similarity_matrix: {similarity_matrix}")
            # assert False
            sim = mask * similarity_matrix    # seqlen x seqlen
            # print(f"sim: {sim[2]}, sum: {sim[2].sum()}")
            # print(f"matrix: {similarity_matrix[2]}, sum: {similarity_matrix[2].sum()}")
            # i = 2
            # print(sim[i].sum() / similarity_matrix[i].sum())
            # print(-torch.log(sim[i].sum() / similarity_matrix[i].sum()))
            # assert False
            # if j == 2:
            #     assert False
            loss = 0.0
            for i in range(1, seq_len):
                # print(f"i: {i}")
                # print(f"j: {j}, scores[j]: {scores[j][i]}")
                # print(f"sim[i]: {sim[i]}")
                # print(f"similarity_matrix[i]: {similarity_matrix[i]}")

                loss_i = sim[i].sum() / similarity_matrix[i].sum()
                loss_i = -torch.log(loss_i)# / i)  #求-log
                # print(f"bz:{j}, index:{i}, loss: {loss_i}")
                losses.append(loss_i)
                loss += loss_i
            #     if i == 5:
            #         assert False
            # assert False
                
            loss = loss / seq_len
            res += loss
            # print(f"loss: {loss}")
        # assert False
        res = torch.mean(torch.tensor(losses))
        # res = res / scores.shape[0]
        return res



    def cal_ccl_loss_topkattn(self, scores, sm, trues, preds):
        # scores: bz * seqlen * seqlen
        # negres = []
        # posres = []
        losses = torch.tensor([])
        lossn = 0
        for i in range(self.start, scores.shape[1]-1):
            curs = scores[:,i,:]
            topknum = min(self.start, self.topk)
            curtop = curs.topk(self.topk, largest = True, sorted = True)
            indexs = curtop.indices
            # values = curtop.values
            # print(indexs.tolist())
            # print(trues.shape, indexs.shape)
            curtrues = torch.gather(trues, dim=1, index=indexs)
            curpreds = torch.gather(preds, dim=1, index=indexs)
            
            curpreds[curtrues==0]=1-curpreds[curtrues==0]
            
            newp = curpreds
            predtrues = newp * (newp>=0.5).long()
            predfalses = newp * (newp<0.5).long()

            pnum = torch.count_nonzero(predtrues, dim=1)+1
            first = 1 - torch.sum(predtrues, dim=1) / pnum
            ######
            nnum = torch.count_nonzero(predfalses, dim=1)+1
            predfalses = 1-predfalses-self.m
            predfalses[predfalses>=1-self.m] = 0
            # print(predfalses)
            predfalses = torch.sum(predfalses, dim=1) / nnum 
            second = self.w * predfalses

            # print(first)
            # print(second)
            # assert False

            loss = first + second
            # print(loss)
            
            # loss = torch.nan_to_num(loss, 0)
            # print(loss)
            
            loss = torch.masked_select(loss, sm[:,i-1])
            # print(loss)
            # assert False
            lossn += torch.count_nonzero(loss)
            # print(f"lossn: {lossn}, loss: {loss}")
            # assert False
            if losses.shape[0] == 0:
                losses = loss
            else:
                losses = torch.cat([losses, loss])#.extend(loss)#.tolist())

        # print(f"losses: {losses}")
        # print(f"sum losses: {torch.sum(losses)}")
        # print(loss.requires_grad)
        loss = torch.sum(losses) / lossn
        # loss.requires_grad_(True) 
        # print(loss.requires_grad)
        # # loss = sum(losses) / lossn
        # print(f"loss: {loss}")
        # assert False
        return loss#torch.sum(torch.tensor(losses)) / lossnes

            # print(first, second)
        # losses = []
        # for i in range(self.start, scores.shape[1]-1):
        #     curs = scores[:,i,:]
        #     curtop = curs.topk(self.start, largest = True, sorted = True)
        # #     print(curtop)
        #     indexs = curtop.indices
        #     values = curtop.values
        # #     print(values.tolist())
        #     # print(indexs.tolist())
        #     for j in range(0, scores.shape[0]):
        #         if sm[j][i-1] == 0:
        #             continue
        #         curidxs = indexs[j].tolist()
        #         curtrues = trues[j][curidxs]
        #         curpreds = preds[j][curidxs]
        # #         print(curtrues, curidxs)
        #         # print(curtrues)
        #         # print(curpreds)
        #         # import copy
        #         newp = curpreds#copy.deepcopy(curpreds)

        #         newp[curtrues==0]=1-newp[curtrues==0] # 预测正确的概率
        #         # print(f"newp: {newp}")
                
        #         # 预测正确的作为正例，预测错误的作为负例，根据newp值0.5阈值判断
        #         predtrues = newp[newp>=0.5]
        #         predfalses = newp[newp<0.5]
        #         if predtrues.shape[0] == 0 or predfalses.shape[0] == 0:
        #             continue
        #         # 正例的预测，准确率越高，loss越小
        #         first = torch.sum(1 - predtrues) / predtrues.shape[0]
                
        #         # w, m = 0.2, 0.2
        #         # 错误的预测，错误率越高，loss越大
        #         tmp = 1 - predfalses - self.m
        #         tmp[tmp<0] = 0
        # #         print(tmp)
        #         second = self.w * torch.sum(tmp, dim=0) / predfalses.shape[0] 
        #         # print(f"first: {first}, second: {second}")
        #         loss = first + second
        #         losses.append(loss)#.item())
        
        # loss = sum(losses) / len(losses)#torch.mean(torch.tensor(loss))
        # print(loss)
        # assert False
        # return loss


    def cal_ccl_loss_cosine2(self, scores, q_embed_data, sm):
        # print(f"random loss!")
        # loss = torch.rand(scores.shape[0], scores.shape[1]).to(device)
        # loss = torch.masked_select(loss[:,1:], sm)
        # loss = torch.sum(loss) / loss.shape[0]
        loss = torch.sigmoid(torch.randn(1)).to(device)
        return loss

    def cal_ccl_loss_cosine(self, scores, q_embed_data, sm):
        # scores: bz * seqlen * seqlen
        # embedding cosine 
        # from torch.functional import F
        tal = 0.2
        coses = torch.cosine_similarity(q_embed_data.unsqueeze(1), q_embed_data.unsqueeze(2), dim=-1)
        coses = torch.exp(coses/tal)
        coses = torch.tril(coses,diagonal=-1) # 下三角不为0，对角线和上三角为0
        # print(f"q_embed_data: {q_embed_data.shape}, coses: {coses.shape}, start: {self.start}")
        # print(f"coses: {coses[0,2,:]}")
        # print(f"scores: {scores[0,2,:]}")
        # assert False


        # negres = []
        # posres = []

        losses = []
        res = []
        for j in range(self.start, scores.shape[1]):
            # pos
            premax = torch.max(scores[:,j,:], dim=1).indices
            bzindex = torch.arange(0,scores.shape[0],1).to(device)
            curcospos = coses[:,j,:][bzindex, premax]
            # print(f"j: {j}, coses[:,j,:]: {coses[:,j,:]}")
            # print(f"curcospos: {curcospos}")
            

            # all
            cur = coses[:,j,:]
            addnum = torch.sum(cur, dim=1)
            # print(f"curpos: {curcospos}, addnum: {addnum}")

            curloss = -1 * torch.log(curcospos / addnum)
            curloss = torch.masked_select(curloss, sm[:,j-1])

            # for i in range(0, scores.shape[0]):
            #     print(f"bz: {i}, index: {j}, curloss: {curloss[i]}")

            # if j == 3:
            #     assert False
            
            losses.extend(curloss)
            # res.append(curloss.sum() / curloss.shape[0])
            # print(f"loss1: {curloss.sum() / curloss.shape[0]}, loss2: {torch.mean(curloss)}")
            # assert False
            # res.append(torch.mean(curloss))

        # loss = torch.mean(torch.tensor(losses))
        # print(f"loss1torch: {torch.mean(torch.tensor(losses))}, loss2: {sum(losses) / len(losses)}")
        print(f"losses: {losses}")
        loss = sum(losses) / len(losses)
        # print(loss)

        # loss = sum(res) / len(res)
        # print(f"loss1: {loss}, loss2: {torch.mean(torch.tensor(res))}")
        # loss = torch.mean(torch.tensor(res))
        print(loss)
        assert False
        return loss

    def cal_ccl_loss_cosine1(self, scores, q_embed_data, sm):
        # scores: bz * seqlen * seqlen
        # embedding cosine 
        # from torch.functional import F
        # tal = 2
        coses = torch.cosine_similarity(q_embed_data.unsqueeze(1), q_embed_data.unsqueeze(2), dim=-1)
        coses = torch.exp(coses)#*tal)
        coses = torch.tril(coses,diagonal=-1) # 下三角不为0，对角线和上三角为0
        # print(f"q_embed_data: {q_embed_data.shape}, coses: {coses.shape}, start: {self.start}")
        # print(f"coses: {coses}")
        # assert False

        negres = []
        posres = []
        if self.start >= scores.shape[1]:
            return 0
        for j in range(self.start, scores.shape[1]):
            # pos
            # print(scores[:,j,:])
            # print(coses[:,j,:])
            
            premax = torch.max(scores[:,j,:], dim=1).indices
            bzindex = torch.arange(0,scores.shape[0],1).to(device)
            curcospos = coses[:,j,:][bzindex, premax]
            # print(curcospos.shape)
            posres.append(curcospos.tolist())

            # neg
            curcospos = curcospos - self.m
            cur = coses[:,j,:] - self.m
            cur[cur<0] = 0
            addnum = torch.sum(cur, dim=1)
            # print("addnum: ", addnum)
            final = (addnum - curcospos) / (j - 1)
            negres.append(final.tolist())

            # print(f"pos: {curcospos+self.m}, neg: {final}")

            # assert False
        # print(f"sm: {sm.shape}, {sm[:,self.start-1:].shape}")
        negres = torch.tensor(negres).transpose(0,1).to(device)
        negres = torch.masked_select(negres, sm[:,self.start-1:])
        posres = torch.tensor(posres).transpose(0,1).to(device)
        posres = torch.masked_select(posres, sm[:,self.start-1:])

        # print(f"negres: {negres.shape}, posres: {posres.shape}")
        
        loss = (1-posres) + negres * self.w
        loss = torch.sum(loss)/ loss.shape[0]
        # print(loss)
        # assert False
        return loss

    def cal_ccl_loss(self, scores, sm):
        # scores: bz * seqlen * seqlen
        # negreses = []
        # posreses = []
        losses = []
        for j in range(self.start, scores.shape[1]):
            premax = torch.max(scores[:,j,:], dim=1).values
            # posres.append(premax)

            premax = premax - self.m
            # print("premax: ", premax)
            # print(scores[:,j,:])
            cur = scores[:,j,:] - self.m
            cur[cur<0] = 0
        #     print("cur: ", cur)
            addnum = torch.sum(cur, dim=1)
            # print("addnum: ", addnum)
        
            final = (addnum - premax) / (j - 1)
            # negres.append(final)

            posres = torch.masked_select(premax, sm[:,j-1])
            negres = torch.masked_select(final, sm[:,j-1])

            curloss = (1-posres) + negres * self.w
            losses.extend(curloss)
            # print("final: ", final)
            # print("-="*10)

        # print(f"losses: {losses}")
        loss = sum(losses) / len(losses)
        # print(loss)
        # assert False
        return loss

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)

        sm = dcur["smasks"]

        emb_type = self.emb_type

        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        pid_embed_data = None
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
        y2 = 0
        if emb_type in ["qid", "qidaktrasch", "qid_scalar", "qid_norasch"]:
            d_output, scores = self.model(q_embed_data, qa_embed_data)
            # print(f"d_output: {d_output.shape} scores: {scores.shape}")
            # print(scores)
            
            # assert False

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)

        # y2 = self.cal_ccl_loss(scores, sm)
        y2 = self.cal_ccl_loss_topkattn(scores, sm, target, preds)
        # if self.start == -1:
        #     y2 = self.cal_ccl_loss_weili(scores, d_output, sm)
        # else:
        #     y2 = self.cal_ccl_loss_cosine(scores, d_output, sm)

        if train:
            return preds, y2
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds, q_data, pid_embed_data, q_embed_data

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

        if model_type in {'bakt_simplex'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
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
            x, scores = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
        return x, scores

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
            query2, scores = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2, scores = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query, scores


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
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

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
        values, scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = values.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)
        # scores = scores.transpose(1, 2).contiguous()\
        #     .view(bs, -1, self.d_model)
        # print(scores.shape)
        # print(scores)
        # assert False
        scores = torch.sum(scores, dim=1) / self.h

        output = self.out_proj(concat)

        return output, scores


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    # print(f"ori scores: {scores}")
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    nscores = dropout(scores)
    output = torch.matmul(nscores, v)
    # import sys
    # sys.exit()
    return output, scores


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

        return tg_emb

