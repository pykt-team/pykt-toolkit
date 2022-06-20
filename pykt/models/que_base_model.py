from tkinter import Y
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics


class QueEmb(nn.Module):
    def __init__(self,num_q,num_c,emb_size,device='cpu',emb_type='qid',emb_path="", pretrain_dim=768):
        super().__init__()
        self.device = device
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.emb_type = emb_type
        self.emb_path = emb_path
        self.pretrain_dim = pretrain_dim

        if "q_c_merge" in emb_type:
            self.concept_emb = nn.Parameter(torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True)#concept embeding
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)#question embeding
            self.que_c_linear = nn.Linear(2*self.emb_size,self.emb_size)

        if emb_type.startswith("qid"):
            self.interaction_emb = nn.Embedding(self.num_q * 2, self.emb_size)
        
        
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

    def forward(self,q,c,r):
        emb_type = self.emb_type
        if "q_c_merge" in emb_type:
            concept_avg = self.get_avg_skill_emb(c)#[batch,max_len-1,emb_size]
            que_emb = self.que_emb(q)#[batch,max_len-1,emb_size]
            # print(f"que_emb shape is {que_emb.shape}")
            que_c_emb = torch.cat([concept_avg,que_emb],dim=-1)#[batch,max_len-1,2*emb_size]
            que_c_emb = self.que_c_linear(que_c_emb)#[batch,max_len-1,emb_size]

        if emb_type == "qid":
            x = q + self.num_q * r
            xemb = self.interaction_emb(x)#[batch,max_len-1,emb_size]
            # print("qid")
        elif emb_type == "qid+q_c_merge":
            x = q + self.num_q * r
            xemb = self.interaction_emb(x)#[batch,max_len-1,emb_size]
            xemb = xemb + que_c_emb
            # print("qid+q_c_merge")
        elif emb_type=="q_c_merge":
            # print("q_c_merge")
            xemb = que_c_emb
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
        set_seed(seed)


        
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

    def train_one_step(self,data):
        raise NotImplemented()

    def predict_one_step(self,data):
        raise NotImplemented()
        
    def get_loss(self, ys,rshft,sm):
        y_pred = torch.masked_select(ys, sm)
        y_true = torch.masked_select(rshft, sm)
        loss = self.loss_func(y_pred.double(), y_true.double())
        return loss

    def _save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, self.model.emb_type+"_model.ckpt"))

    def load_model(self,ckpt_path):
        net = torch.load(os.path.join(ckpt_path, self.emb_type+"_model.ckpt"))
        self.model.load_state_dict(net)
    
    def batch_to_device(self,data):
        dcur = data
        q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
        m, sm = dcur["masks"], dcur["smasks"]

        cq = torch.cat((q[:,0:1], qshft), dim=1)
        cc = torch.cat((c[:,0:1], cshft), dim=1)
        cr = torch.cat((r[:,0:1], rshft), dim=1)
        ct = torch.cat((t[:,0:1], tshft), dim=1)
        return q, c, r, t, qshft, cshft, rshft, tshft, m, sm, cq, cc, cr, ct

    def train(self,train_dataset, valid_dataset,batch_size=16,valid_batch_size=None,num_epochs=32, test_loader=None, test_window_loader=None,ckpt_path="",save_model=False,patient=10,shuffle=True):
        self.ckpt_path = ckpt_path
        if valid_batch_size is None:
            valid_batch_size = batch_size

        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=shuffle)
        
        max_auc, best_epoch = 0, -1
        train_step = 0
        
        for i in range(1, num_epochs + 1):
            loss_mean = []
            for data in train_loader:
                q, c, r, t, qshft, cshft, rshft, tshft, m, sm, cq, cc, cr, ct = self.batch_to_device(data)
                train_step += 1
                self.model.train()
                y = self.train_one_step(data)
                loss = self.get_loss(y,rshft,sm)#get loss
                self.opt.zero_grad()
                loss.backward()#compute gradients 
                self.opt.step()#update model’s parameters
                loss_mean.append(loss.detach().cpu().numpy())
               
            loss_mean = np.mean(loss_mean)
            auc, acc = self.evaluate(valid_dataset,batch_size=valid_batch_size)

            if auc > max_auc:
                if save_model:
                    self._save_model()
                max_auc = auc
                best_epoch = i
                testauc, testacc = -1, -1
                window_testauc, window_testacc = -1, -1

                validauc, validacc = round(auc, 4), round(acc, 4)#model.evaluate(valid_dataset, emb_type)
                testauc, testacc, window_testauc, window_testacc = round(testauc, 4), round(testacc, 4), round(window_testauc, 4), round(window_testacc, 4)
                max_auc = round(max_auc, 4)
            print(f"Epoch: {i}, validauc: {validauc}, validacc: {validacc}, best epoch: {best_epoch}, best auc: {max_auc}, loss: {loss_mean}, emb_type: {self.model.emb_type}, model: {self.model.model_name}, save_dir: {self.ckpt_path}")
            print(f"            testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

            if i - best_epoch >= patient:
                break
        return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch


    def evaluate(self,dataset,batch_size,acc_throld=0.5):
        ps,ts = self.predict(dataset,batch_size=batch_size)
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= acc_throld else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        # eval_result = {"auc":auc,"acc":acc}
        # return eval_result
        return auc,acc


    def predict(self,dataset,batch_size,return_ts=False):
        test_loader = DataLoader(dataset, batch_size=batch_size,shuffle=False)
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_scores = []
            for data in test_loader:
                q, c, r, t, qshft, cshft, rshft, tshft, m, sm, cq, cc, cr, ct = self.batch_to_device(data)
                y = self.predict_one_step(data)
                y = torch.masked_select(y, sm).detach().cpu()
                t = torch.masked_select(rshft, sm).detach().cpu()
                y_trues.append(t.numpy())
                y_scores.append(y.numpy())
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        return ps,ts