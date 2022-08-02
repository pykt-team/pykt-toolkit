import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .que_base_model import QueBaseModel,QueEmb
from sklearn import metrics
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class MLP(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


class AKTQueNet(nn.Module):
    def __init__(self, num_q, num_c, emb_size, n_blocks, dropout, d_ff=256, 
            kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "akt_que"
        self.num_c = num_c
        self.dropout = dropout
        self.kq_same = kq_same
        self.num_q = num_q
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type,self.loss_mode,self.predict_mode,self.output_mode = emb_type.split("|-|")
        # embed_l = d_model
        if self.num_q > 0:
            self.difficult_param = nn.Embedding(self.num_q+1, 1) # 题目难度
            self.q_embed_diff = nn.Embedding(self.num_q+1, emb_size) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.num_q + 1, emb_size) # interaction emb, 同上
        
        if self.separate_qa: 
            self.qa_embed = nn.Embedding(2*self.num_c+1, emb_size) # interaction emb
        else: # false default
            self.qa_embed = nn.Embedding(2, emb_size)

        self.que_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type=self.emb_type,device=device,
                             emb_path=emb_path,pretrain_dim=pretrain_dim,model_name=self.model_name)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(num_q=num_q, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=emb_size, d_feature=emb_size / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type)
        
        # self.out_que_next = MLP(n_layer=1,hidden_dim=2*emb_size,output_dim=1,dpo=dropout)
        # self.out_concept_next = MLP(n_layer=1,hidden_dim=2*emb_size,output_dim=self.num_c,dpo=dropout)

        # self.out_que_all = MLP(n_layer=1,hidden_dim=emb_size,output_dim=self.num_q,dpo=dropout)
        # self.out_concept_all = MLP(n_layer=1,hidden_dim=emb_size,output_dim=self.num_c,dpo=dropout)

        self.out_que_next = nn.Sequential(
            nn.Linear(emb_size + emb_size,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.out_concept_next = nn.Sequential(
            nn.Linear(emb_size + emb_size,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, self.num_c)
        )

        self.out_que_all = nn.Sequential(
            nn.Linear(emb_size,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, self.num_q)
        )

        self.out_concept_all = nn.Sequential(
            nn.Linear(emb_size,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, self.num_c)
        )

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_q+1 and self.num_q > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q, c, r):
        if self.emb_type=="iekt":
            xemb,emb_qca,emb_qc,emb_q,emb_c = self.que_emb(q, c, r)
            q_embed_data = xemb
        else:
            q_embed_data = self.que_emb(q,c,r)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q + self.num_q * r
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(r)+q_embed_data
        return q_embed_data, qa_embed_data

    def get_avg_fusion_concepts(self,y_concept,cshft):
        """获取知识点 fusion 的预测结果
        """
        concept_mask = torch.where(cshft.long()==-1,False,True)
        concept_index = F.one_hot(torch.where(cshft!=-1,cshft,0),self.num_c)
        concept_sum = (y_concept.unsqueeze(2).repeat(1,1,4,1)*concept_index).sum(-1)
        concept_sum = concept_sum*concept_mask#remove mask
        y_concept = concept_sum.sum(-1)/torch.where(concept_mask.sum(-1)!=0,concept_mask.sum(-1),1)
        return y_concept

    # def forward(self, q_data, target, pid_data=None, qtest=False):
    def forward(self, q, c, r,data=None):
        # Batch First
        q_embed_data, qa_embed_data = self.base_emb(q,c,r)

        if self.num_q > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(q)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(q)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(
                r)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            reg_loss = (pid_embed_data ** 2.).sum() * self.l2 # rasch部分loss
        else:
            reg_loss = 0.

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        h = self.model(q_embed_data, qa_embed_data)
        h_q = torch.cat([h, q_embed_data], dim=-1)

        y_question_next = torch.sigmoid(self.out_que_next(h_q).squeeze(-1))[:,1:]
        y_concept_next = torch.sigmoid(self.out_concept_next(h_q).squeeze(-1))[:,1:]
        y_question_all = torch.sigmoid(self.out_que_all(h))[:,1:]
        y_concept_all = torch.sigmoid(self.out_concept_all(h))[:,1:]

        outputs = {"y_question_next":y_question_next,"y_concept_next":y_concept_next,"reg_loss":reg_loss,"y_question_all":y_question_all,"y_concept_all":y_concept_all}

        # all to next question
        outputs['y_concept_next'] = self.get_avg_fusion_concepts(outputs['y_concept_next'],data['cshft'])
        outputs['y_concept_all'] = self.get_avg_fusion_concepts(outputs['y_concept_all'],data['cshft'])
        outputs['y_question_all'] = (outputs["y_question_all"] * F.one_hot(data['qshft'].long(), self.num_q)).sum(-1)
        
        return outputs
       



class AKTQue(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size,n_blocks=1, dropout=0.1, emb_type='qid',kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5,d_ff=256,emb_path="", pretrain_dim=768,device='cpu',seed=0):
        model_name = "akt_que"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = AKTQueNet(num_q=num_q, num_c=num_c, emb_size=emb_size, n_blocks=n_blocks, dropout=dropout, d_ff=d_ff, 
            kq_same=kq_same, final_fc_dim=final_fc_dim, num_attn_heads=num_attn_heads, separate_qa=separate_qa, 
            l2=l2, emb_type=emb_type, emb_path=emb_path, pretrain_dim=pretrain_dim)
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type

    def predict(self,dataset,batch_size,return_ts=False,process=True):
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_pred_dict = {}
            for data in test_loader:
                new_data = self.batch_to_device(data,process=process)
                outputs,data_new = self.predict_one_step(data,return_details=True)
               
                for key in outputs:
                    # print(f"key is {key},shape is {outputs[key].shape}")
                    if not key.startswith("y") or key in ['y_qc_predict']:
                        continue
                    elif key not in y_pred_dict:
                       y_pred_dict[key] = []
                    # print(f"outputs is {outputs}")
                    y = torch.masked_select(outputs[key], new_data['sm']).detach().cpu()#get label
                    y_pred_dict[key].append(y.numpy())
                
                t = torch.masked_select(new_data['rshft'], new_data['sm']).detach().cpu()
                y_trues.append(t.numpy())
        results = y_pred_dict
        for key in results:
            results[key] = np.concatenate(results[key], axis=0)
        ts = np.concatenate(y_trues, axis=0)
        results['ts'] = ts
        return results

    def evaluate(self,dataset,batch_size,acc_threshold=0.5):
        results = self.predict(dataset,batch_size=batch_size)
        eval_result = {}
        ts = results["ts"]
        for key in results:
            if not key.startswith("y") or key in ['y_qc_predict']:
                pass
            else:
                ps = results[key]
                kt_auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
                prelabels = [1 if p >= acc_threshold else 0 for p in ps]
                kt_acc = metrics.accuracy_score(ts, prelabels)
                if key!="y":
                    eval_result["{}_kt_auc".format(key)] = kt_auc
                    eval_result["{}_kt_acc".format(key)] = kt_acc
                else:
                    eval_result["auc"] = kt_auc
                    eval_result["acc"] = kt_acc
        self.eval_result = eval_result
        return eval_result

    def get_merge_loss(self,loss_question,loss_concept,loss_mode):
        if loss_mode in ["c"]:
            loss = loss_concept
        elif loss_mode in ["q"]:
            loss = loss_question
        elif loss_mode in ["qc"]:
            loss = (loss_question+loss_concept)/2
        return loss

    def train_one_step(self,data,process=True,return_all=False):
        outputs,data_new = self.predict_one_step(data,return_details=True,process=process)
        all_loss_mode,next_loss_mode =  self.model.loss_mode.replace("_dyn","").split("_")[0].split("-")

        # predict next loss
        loss_next_question = self.get_loss(outputs['y_question_next'],data_new['rshft'],data_new['sm'])
        loss_next_concept = self.get_loss(outputs['y_concept_next'],data_new['rshft'],data_new['sm'])
        loss_next = self.get_merge_loss(loss_next_question,loss_next_concept,next_loss_mode)

        # predict all loss
        loss_all_question = self.get_loss(outputs['y_question_all'],data_new['rshft'],data_new['sm'])
        loss_all_concept = self.get_loss(outputs['y_concept_all'],data_new['rshft'],data_new['sm'])
        loss_all = self.get_merge_loss(loss_all_question,loss_all_concept,all_loss_mode)
        

        loss = loss_all + loss_next + outputs['reg_loss'] 
        print(f"loss={loss:.3f},loss_all={loss_all:.3f},loss_next={loss_next:.3f},loss_next_question={loss_next_question:.3f},loss_next_concept={loss_next_concept:.3f},loss_all_question={loss_all_question:.3f},loss_all_concept={loss_all_concept:.3f}")

        return outputs['y'],loss

    def get_merge_y(self,y_question,y_concept,predict_mode):
        if predict_mode=="c":
            y = y_concept
        elif predict_mode=="q":
            y = y_question
        elif predict_mode in ["qc"]:
            y = (y_question+y_concept)/2
        return y

    def predict_one_step(self,data,return_details=False,process=True):
        all_predict_mode,next_predict_mode = self.model.predict_mode.split("_")[0].split("-")

        data_new = self.batch_to_device(data,process=process)
        outputs = self.model(data_new['cq'].long(),data_new['cc'].long(),data_new['cr'].long(),data_new)

       
        # predict next results
        y_qc_next = self.get_merge_y(outputs['y_question_next'],outputs['y_concept_next'],next_predict_mode)
        outputs['y_qc_next'] = y_qc_next

        # predict all results
        y_qc_all = self.get_merge_y(outputs['y_question_all'],outputs['y_concept_all'],all_predict_mode)
        outputs['y_qc_all'] = y_qc_all

        y = (y_qc_all + y_qc_next)/2

        outputs['y'] = y
        if return_details:
            return outputs,data_new
        else:
            return y

class Architecture(nn.Module):
    def __init__(self, num_q,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'akt','akt_que'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
            y = block(mask=1, query=y, key=y, values=y) # yt^
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False) # False: 没有FFN, 第一层只有self attention, 对应于xt^
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
                # print(x[0,0,:])
                flag_first = True
        return x

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
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

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
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
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



