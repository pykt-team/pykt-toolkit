import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn import metrics

emb_type_list = ["qc_merge","qid","qaid","qcid_merge"]
emb_type_map = {"akt-iekt":"qc_merge",
                "iekt-qid":"qc_merge",
                "iekt_ab-qid":"qc_merge",
                "iekt-qc_merge":"qc_merge",
                "iekt_ce-qid":"qc_merge",
                "dkt_que-qid":"qaid_qc",
                "dkt_que-qcaid":"qcaid",
                "dkt_que-qcaid_h":"qcaid_h",
                }
  

class QueEmb(nn.Module):
    def __init__(self,num_q,num_c,emb_size,model_name,device='cpu',emb_type='qid',emb_path="", pretrain_dim=768):
        """_summary_

        Args:
            num_q (_type_): num of question
            num_c (_type_): num of concept
            emb_size (_type_): emb_size
            device (str, optional): device. Defaults to 'cpu'.
            emb_type (str, optional): how to encode question id. Defaults to 'qid'. qid:question_id one-hot; 
                qaid:question_id + r*question_num one-hot; qc_merge: question emb + avg(concept emb);
            emb_path (str, optional): _description_. Defaults to "".
            pretrain_dim (int, optional): _description_. Defaults to 768.
        """
        super().__init__()
        self.device = device
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        #get emb type
        tmp_emb_type = f"{model_name}-{emb_type}"
        emb_type = emb_type_map.get(tmp_emb_type,tmp_emb_type.replace(f"{model_name}-",""))
        print(f"emb_type is {emb_type}")

        self.emb_type = emb_type
        self.emb_path = emb_path
        self.pretrain_dim = pretrain_dim

        if emb_type in ["qc_merge","qaid_qc"]:
            self.concept_emb = nn.Parameter(torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True)#concept embeding
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)#question embeding
            self.que_c_linear = nn.Linear(2*self.emb_size,self.emb_size)

        if emb_type =="qaid_c":
            self.que_c_linear = nn.Linear(2*self.emb_size,self.emb_size)
        
        if emb_type in ["qcaid","qcaid_h"]:
            self.concept_emb = nn.Parameter(torch.randn(self.num_c*2, self.emb_size).to(device), requires_grad=True)#concept embeding
            self.que_inter_emb = nn.Embedding(self.num_q * 2, self.emb_size)
            self.que_c_linear = nn.Linear(2*self.emb_size,self.emb_size)


        if emb_type.startswith("qaid"):
            self.interaction_emb = nn.Embedding(self.num_q * 2, self.emb_size)

        if emb_type.startswith("qid"):
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)

        if emb_type == "qcid":#question_emb concat avg(concepts emb)
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)
            self.concept_emb = nn.Parameter(torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True)#concept embeding
            self.que_c_linear = nn.Linear(2*self.emb_size,self.emb_size)


        if emb_type == "iekt":
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)#question embeding
            # self.que_emb.weight.requires_grad = False
            self.concept_emb = nn.Parameter(torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True)#concept embeding
            self.que_c_linear = nn.Linear(2*self.emb_size,self.emb_size)
        
        self.output_emb_dim = emb_size

    def get_avg_skill_emb(self,c):
        # add zero for padding
        concept_emb_cat = torch.cat(
            [torch.zeros(1, self.emb_size).to(self.device), 
            self.concept_emb], dim=0)
        # shift c

        related_concepts = (c+1).long()
        #[batch_size, seq_len, emb_dim]
        concept_emb_sum = concept_emb_cat[related_concepts, :].sum(
            axis=-2)

        #[batch_size, seq_len,1]
        concept_num = torch.where(related_concepts != 0, 1, 0).sum(
            axis=-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg

    def forward(self,q,c,r=None):
        emb_type = self.emb_type
        if "qc_merge" in emb_type:
            concept_avg = self.get_avg_skill_emb(c)#[batch,max_len-1,emb_size]
            que_emb = self.que_emb(q)#[batch,max_len-1,emb_size]
            # print(f"que_emb shape is {que_emb.shape}")
            que_c_emb = torch.cat([concept_avg,que_emb],dim=-1)#[batch,max_len-1,2*emb_size]
            
        if emb_type == "qaid":
            x = q + self.num_q * r
            xemb = self.interaction_emb(x)#[batch,max_len-1,emb_size]
            # print("qid")
        elif emb_type == "qid":
            xemb = self.que_emb(q)#[batch,max_len-1,emb_size]
        elif emb_type == "qaid+qc_merge":
            x = q + self.num_q * r
            xemb = self.interaction_emb(x)#[batch,max_len-1,emb_size]
            que_c_emb = self.que_c_linear(que_c_emb)#[batch,max_len-1,emb_size]
            xemb = xemb + que_c_emb
            # print("qid+qc_merge")
        elif emb_type=="qc_merge":
            # print("qc_merge")
            xemb = que_c_emb
        elif emb_type =="qaid_qc":
            x = q + self.num_q * r
            emb_q = self.interaction_emb(x)
            emb_c = self.get_avg_skill_emb(c)#[batch,max_len-1,emb_size]
            xemb = torch.cat([emb_q,emb_c],dim=-1)
            xemb = self.que_c_linear(xemb)
        elif emb_type in ["qcaid","qcaid_h"]:
            x_q = q + self.num_q * r
            gate = torch.where(c==-1,0,1)
            x_c = c + self.num_c * r.unsqueeze(-1).repeat(1,1,4)*gate
            emb_q = self.que_inter_emb(x_q)
            emb_c = self.get_avg_skill_emb(x_c)
            xemb = torch.cat([emb_q,emb_c],dim=-1)
            xemb = self.que_c_linear(xemb)
            return xemb,emb_q,emb_c
        elif emb_type in ["qcid","qaid_h"]:
            emb_c = self.get_avg_skill_emb(c)#[batch,max_len-1,emb_size]
            emb_q = self.que_emb(q)#[batch,max_len-1,emb_size]
            que_c_emb = torch.cat([emb_q,emb_c],dim=-1)#[batch,max_len-1,2*emb_size]
            xemb = self.que_c_linear(xemb)
            return xemb,emb_q,emb_c
        elif emb_type == "iekt":
            emb_c = self.get_avg_skill_emb(c)#[batch,max_len-1,emb_size]
            emb_q = self.que_emb(q)#[batch,max_len-1,emb_size]
            emb_qc = torch.cat([emb_q,emb_c],dim=-1)#[batch,max_len-1,2*emb_size]
            xemb = self.que_c_linear(emb_qc)
            # print(f"emb_qc shape is {emb_qc.shape}")
            # print(f"r shape is {r.shape}")
            # print(f"(1-r).unsqueeze(-1).repeat(1,1, self.emb_size * 2) shape is {(1-r).unsqueeze(-1).repeat(1,1, self.emb_size * 2).shape}")
            emb_qca = torch.cat([emb_qc.mul((1-r).unsqueeze(-1).repeat(1,1, self.emb_size * 2)),
                                emb_qc.mul((r).unsqueeze(-1).repeat(1,1, self.emb_size * 2))], dim = -1)# s_t 扩展，分别对应正确的错误的情况
            return xemb,emb_qca,emb_qc,emb_q,emb_c

        return xemb

from pykt.utils import set_seed
class QueBaseModel(nn.Module):
    def __init__(self,model_name,emb_type,emb_path,pretrain_dim,device,seed=0):
        super().__init__()
        self.model_name = model_name
        self.emb_type = emb_type
        self.emb_path = emb_path
        self.pretrain_dim = pretrain_dim
        self.device = device
        # set_seed(seed)

    def compile(self, optimizer,lr=0.001,
                loss='binary_crossentropy',
                metrics=None):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        ref from https://github.com/shenweichen/DeepCTR-Torch/blob/2cd84f305cb50e0fd235c0f0dd5605c8114840a2/deepctr_torch/models/basemodel.py
        """
        self.lr = lr
        # self.metrics_names = ["loss"]
        self.opt = self._get_optimizer(optimizer)
        self.loss_func = self._get_loss_func(loss)
        # self.metrics = self._get_metrics(metrics)

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _get_optimizer(self,optimizer):
        if isinstance(optimizer, str):
            if optimizer == 'gd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            elif optimizer == 'adagrad':
                optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr)
            elif optimizer == 'adadelta':
                optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.lr)
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            else:
                raise ValueError("Unknown Optimizer: " + self.optimizer)
        return optimizer

    def train_one_step(self,data,process=True):
        raise NotImplemented()

    def predict_one_step(self,data,process=True):
        raise NotImplemented()
        
    def get_loss(self, ys,rshft,sm):
        y_pred = torch.masked_select(ys, sm)
        y_true = torch.masked_select(rshft, sm)
        loss = self.loss_func(y_pred.double(), y_true.double())
        return loss

    def _save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, self.model.emb_type+"_model.ckpt"))

    def load_model(self,save_dir):
        net = torch.load(os.path.join(save_dir, self.emb_type+"_model.ckpt"))
        self.model.load_state_dict(net)
    
    def batch_to_device(self,data,process=True):
        if not process:
            return data
        dcur = data
        # q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]
        # qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
        # m, sm = dcur["masks"], dcur["smasks"]
        data_new = {}
        data_new['cq'] = torch.cat((dcur["qseqs"][:,0:1], dcur["shft_qseqs"]), dim=1)
        data_new['cc'] = torch.cat((dcur["cseqs"][:,0:1],  dcur["shft_cseqs"]), dim=1)
        data_new['cr'] = torch.cat((dcur["rseqs"][:,0:1], dcur["shft_rseqs"]), dim=1)
        data_new['ct'] = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        data_new['q'] = dcur["qseqs"]
        data_new['c'] = dcur["cseqs"]
        data_new['r'] = dcur["rseqs"]
        data_new['t'] = dcur["tseqs"]
        data_new['qshft'] = dcur["shft_qseqs"]
        data_new['cshft'] = dcur["shft_cseqs"]
        data_new['rshft'] = dcur["shft_rseqs"]
        data_new['tshft'] = dcur["shft_tseqs"]
        data_new['m'] = dcur["masks"]
        data_new['sm'] = dcur["smasks"]
        return data_new

    def train(self,train_dataset, valid_dataset,batch_size=16,valid_batch_size=None,num_epochs=32, test_loader=None, test_window_loader=None,save_dir="tmp",save_model=False,patient=10,shuffle=True,process=True):
        self.save_dir = save_dir
        os.makedirs(self.save_dir,exist_ok=True)

        if valid_batch_size is None:
            valid_batch_size = batch_size

        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=shuffle)
        
        max_auc, best_epoch = 0, -1
        train_step = 0
        
        for i in range(1, num_epochs + 1):
            loss_mean = []
            for data in train_loader:
                train_step += 1
                self.model.train()
                y,loss = self.train_one_step(data,process=process)
                self.opt.zero_grad()
                loss.backward()#compute gradients 
                self.opt.step()#update model’s parameters
                loss_mean.append(loss.detach().cpu().numpy())
               
            loss_mean = np.mean(loss_mean)
            eval_result = self.evaluate(valid_dataset,batch_size=valid_batch_size)
            auc, acc = eval_result['auc'],eval_result['acc']
            print(f"eval_result is {eval_result}")
            if auc > max_auc+1e-3:
                if save_model:
                    self._save_model()
                max_auc = auc
                best_epoch = i
                testauc, testacc = -1, -1
                window_testauc, window_testacc = -1, -1
                validauc, validacc = auc, acc
            print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {self.model.emb_type}, model: {self.model.model_name}, save_dir: {self.save_dir}")
            print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")

            if i - best_epoch >= patient:
                break
        return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch


    def evaluate(self,dataset,batch_size,acc_threshold=0.5):
        ps,ts = self.predict(dataset,batch_size=batch_size)
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= acc_threshold else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        eval_result = {"auc":auc,"acc":acc}
        return eval_result
        # return auc,acc

    def _parser_row(self,row,data_config,ob_portions=0.5):
        max_concepts = data_config["max_concepts"]
        max_len = data_config["maxlen"]
        start_index,seq_len = self._get_multi_ahead_start_index(row['concepts'],ob_portions)
        questions = [int(x) for x in row["questions"].split(",")]
        responses = [int(x) for x in row["responses"].split(",")]
        concept_list = []
        for concept in row["concepts"].split(","):
            if concept == "-1":
                skills = [-1] * max_concepts
            else:
                skills = [int(_) for _ in concept.split("_")]
                skills = skills +[-1]*(max_concepts-len(skills))
            concept_list.append(skills)
        cq_full = torch.tensor(questions).to(self.device)
        cc_full = torch.tensor(concept_list).to(self.device)
        cr_full = torch.tensor(responses).to(self.device)

        history_start_index = max(start_index - max_len,0)
        hist_q = cq_full[history_start_index:start_index].unsqueeze(0)
        hist_c = cc_full[history_start_index:start_index].unsqueeze(0)
        hist_r = cr_full[history_start_index:start_index].unsqueeze(0)
        return hist_q,hist_c,hist_r,cq_full,cc_full,cr_full,seq_len,start_index
        
    
    def _get_multi_ahead_start_index(self,cc,ob_portions=0.5):
        """_summary_

        Args:
            cc (str): the concept sequence
            ob_portions (float, optional): _description_. Defaults to 0.5.

        Returns:
            _type_: _description_
        """
        filter_cc = [x for x in cc.split(",") if x != "-1"]
        seq_len = len(filter_cc)
        start_index = int(seq_len * ob_portions)
        if start_index == 0:
            start_index = 1
        if start_index == seq_len:
            start_index = seq_len - 1
        return start_index,seq_len

   
    def _evaluate_multi_ahead_accumulative(self,data_config,batch_size=1,ob_portions=0.5,acc_threshold=0.5):
       
        testf = os.path.join(data_config["dpath"], "test.csv")
        df = pd.read_csv(testf)
        print("total sequence length is {}".format(len(df)))

        y_pred_list = []
        y_true_list = []
        for i, row in df.iterrows():
            hist_q,hist_c,hist_r,cq_full,cc_full,cr_full,seq_len,start_index = self._parser_row(row,data_config=data_config,ob_portions=ob_portions)
            if i%10==0:
                print(f"predict step {i}")

            seq_y_pred_hist = [cr_full[start_index]]
            for i in range(start_index,seq_len):
                cur_q = cq_full[start_index:i+1].unsqueeze(0)
                cur_c = cc_full[start_index:i+1].unsqueeze(0)
                cur_r = torch.tensor(seq_y_pred_hist).unsqueeze(0).to(self.device)
                # print(f"cur_q is {cur_q} shape is {cur_q.shape}")
                # print(f"cur_r is {cur_r} shape is {cur_r.shape}")
                cq = torch.cat([hist_q,cur_q],axis=1)
                cc = torch.cat([hist_c,cur_c],axis=1)
                cr = torch.cat([hist_r,cur_r],axis=1)
                # print(f"cc_full is {cc_full}")
                # print(f"cr is {cr} shape is {cr.shape}")
                # print(f"cq is {cq} shape is {cq.shape}")
                data = [cq,cc,cr]
                cq,cc,cr = [x.to(self.device) for x in data]#full sequence,[1,n]
                q,c,r = [x[:,:-1].to(self.device) for x in data]#[0,n-1]
                qshft,cshft,rshft = [x[:,1:].to(self.device) for x in data]#[1,n]
                data = {"cq":cq,"cc":cc,"cr":cr,"q":q,"c":c,"r":r,"qshft":qshft,"cshft":cshft,"rshft":rshft}
                y_last_pred = self.predict_one_step(data,process=False)[:,-1][0]
                seq_y_pred_hist.append(1 if y_last_pred>acc_threshold else 0)
            
                y_true_list.append(cr_full[i].item())
                y_pred_list.append(y_last_pred.item())

        print(f"num of y_pred_list is {len(y_pred_list)}")
        print(f"num of y_true_list is {len(y_true_list)}")

        y_pred_list = np.array(y_pred_list)
        y_true_list = np.array(y_true_list)
        auc = metrics.roc_auc_score(y_true_list, y_pred_list)
        acc = metrics.accuracy_score(y_true_list, [1 if p >= acc_threshold else 0 for p in y_pred_list])

        return auc,acc


    def _evaluate_multi_ahead_help(self,data_config,batch_size,ob_portions=0.5,acc_threshold=0.5):
        """generate multi-ahead dataset

        Args:
            data_config (_type_): data_config
            ob_portions (float, optional): portions of observed student interactions. . Defaults to 0.5.

        Returns:
            dataset: new dataset for multi-ahead prediction
        """
        testf = os.path.join(data_config["dpath"], "test.csv")
        df = pd.read_csv(testf)
        print("total sequence length is {}".format(len(df)))
        y_pred_list = []
        y_true_list = []
        for i, row in df.iterrows():
            hist_q,hist_c,hist_r,cq_full,cc_full,cr_full,seq_len,start_index = self._parser_row(row,data_config=data_config,ob_portions=ob_portions)
            if i%10==0:
                print(f"predict step {i}")
            cq_list = []
            cc_list = []
            cr_list = []
            
            for i in range(start_index,seq_len):
                cur_q = cq_full[i:i+1].unsqueeze(0)
                cur_c = cc_full[i:i+1].unsqueeze(0)
                cur_r = cr_full[i:i+1].unsqueeze(0)
                cq_list.append(torch.cat([hist_q,cur_q],axis=1))
                cc_list.append(torch.cat([hist_c,cur_c],axis=1))
                cr_list.append(torch.cat([hist_r,cur_r],axis=1))
                y_true_list.append(cr_full[i].item())
            # print(f"cq_list is {len(cq_list)}")
            cq_ahead = torch.cat(cq_list,axis=0)
            cc_ahead = torch.cat(cc_list,axis=0)
            cr_ahead = torch.cat(cr_list,axis=0)
            # print(f"cq_ahead shape is {cq_ahead.shape}")

            tensor_dataset = TensorDataset(cq_ahead,cc_ahead,cr_ahead)
            dataloader = DataLoader(dataset=tensor_dataset,batch_size=batch_size) 

            for data in dataloader:
                cq,cc,cr = [x.to(self.device) for x in data]#full sequence,[1,n]
                q,c,r = [x[:,:-1].to(self.device) for x in data]#[0,n-1]
                qshft,cshft,rshft = [x[:,1:].to(self.device) for x in data]#[1,n]
                data = {"cq":cq,"cc":cc,"cr":cr,"q":q,"c":c,"r":r,"qshft":qshft,"cshft":cshft,"rshft":rshft}
                y = self.predict_one_step(data,process=False)[:,-1].detach().cpu().numpy().flatten()
                y_pred_list.extend(list(y))
        
        print(f"num of y_pred_list is {len(y_pred_list)}")
        print(f"num of y_true_list is {len(y_true_list)}")

        y_pred_list = np.array(y_pred_list)
        y_true_list = np.array(y_true_list)
        auc = metrics.roc_auc_score(y_true_list, y_pred_list)
        acc = metrics.accuracy_score(y_true_list, [1 if p >= acc_threshold else 0 for p in y_pred_list])

        return auc,acc

    def evaluate_multi_ahead(self,data_config,batch_size,ob_portions=0.5,acc_threshold=0.5,accumulative=False):
        """Predictions in the multi-step ahead prediction scenario

        Args:
            data_config (_type_): data_config
            batch_size (int): batch_size
            ob_portions (float, optional): portions of observed student interactions. Defaults to 0.5.
            accumulative (bool, optional): `True` for accumulative prediction and `False` for non-accumulative prediction. Defaults to False.
            acc_threshold (float, optional): threshold for accuracy. Defaults to 0.5.

        Returns:
            metrics: auc,acc
        """
        self.model.eval()
        with torch.no_grad():
            if accumulative:
                print("predict use accumulative")
                auc,acc = self._evaluate_multi_ahead_accumulative(data_config,batch_size=batch_size,ob_portions=ob_portions,acc_threshold=acc_threshold)
            else:
                print("predict use no accumulative")
                auc,acc = self._evaluate_multi_ahead_help(data_config,batch_size=batch_size,ob_portions=ob_portions,acc_threshold=acc_threshold)
        return {"auc":auc,"acc":acc}
        
 

    def predict(self,dataset,batch_size,return_ts=False,process=True):
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_scores = []
            for data in test_loader:
                new_data = self.batch_to_device(data,process=process)
                y = self.predict_one_step(data)
                y = torch.masked_select(y, new_data['sm']).detach().cpu()
                t = torch.masked_select(new_data['rshft'], new_data['sm']).detach().cpu()
                y_trues.append(t.numpy())
                y_scores.append(y.numpy())
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        return ps,ts
