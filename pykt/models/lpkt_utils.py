#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from sklearn import metrics
from ..datasets.lpkt_utils import generate_time2idx

device = "cpu" if not torch.cuda.is_available() else "cuda"

def generate_qmatrix(data_config, gamma=0.0):
    df_train = pd.read_csv(os.path.join(data_config["dpath"], "train_valid.csv"))
    df_test = pd.read_csv(os.path.join(data_config["dpath"], "test.csv"))
    df = pd.concat([df_train, df_test])

    problem2skill = dict()
    for i, row in df.iterrows():
        cids = [int(_) for _ in row["concepts"].split(",")]
        qids = [int(_) for _ in row["questions"].split(",")]
        for q,c in zip(qids, cids):
            if q in problem2skill:
                problem2skill[q].append(c)
            else:
                problem2skill[q] = [c]
    n_problem, n_skill = data_config["num_q"], data_config["num_c"]
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p in problem2skill.keys():
        for c in problem2skill[p]:
            q_matrix[p][c] = 1
    np.savez(os.path.join(data_config["dpath"], "qmatrix.npz"), matrix = q_matrix)
    return q_matrix

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
        data_new['cit'] = torch.cat((dcur["itseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
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

def predict_one_step(model,data,return_details=False,process=False,return_raw=False):
    # data_new = batch_to_device(data,process=process)
    outputs = model(data['cq'].long(),data['cr'].long(),data['cit'].long())
    if return_details:
        return outputs,data
    else:
        return outputs

def _parser_row(row,data_config,ob_portions=0.5):
    max_concepts = data_config["max_concepts"]
    max_len = data_config["maxlen"]
    start_index,seq_len = _get_multi_ahead_start_index(row['concepts'],ob_portions)
    questions = [int(x) for x in row["questions"].split(",")]
    responses = [int(x) for x in row["responses"].split(",")]
    
    times = [] if "timestamps" not in row else row["timestamps"].split(",")
    if times != []:
        times = [int(x) for x in times]
        # shft_times = [0] + times[:-1]
        shft_times = times[:1] + times[:-1]
        it_times = np.maximum(np.minimum((np.array(times) - np.array(shft_times)) // 1000 // 60, 43200),-1)
        # print(f"it_times:{it_times}")
    else:
        it_times = np.ones(len(questions)).astype(int)
    
    at2idx, it2idx = generate_time2idx(data_config)
    # it_times = [it2idx.get(str(t),len(it2idx)) for t in it_times]
    it_times = [it2idx.get(str(t)) for t in it_times]
    # print(f"it_times:{it_times}")

    concept_list = []
    for concept in row["concepts"].split(","):
        if concept == "-1":
            skills = [-1] * max_concepts
        else:
            skills = [int(_) for _ in concept.split("_")]
            skills = skills +[-1]*(max_concepts-len(skills))
        concept_list.append(skills)
    cq_full = torch.tensor(questions).to(device)
    cc_full = torch.tensor(concept_list).to(device)
    cr_full = torch.tensor(responses).to(device)
    cit_full = torch.tensor(it_times).to(device)

    history_start_index = max(start_index - max_len,0)
    hist_q = cq_full[history_start_index:start_index].unsqueeze(0)
    hist_c = cc_full[history_start_index:start_index].unsqueeze(0)
    hist_r = cr_full[history_start_index:start_index].unsqueeze(0)
    hist_it = cit_full[history_start_index:start_index].unsqueeze(0)
    return hist_q,hist_c,hist_r,hist_it,cq_full,cc_full,cr_full,cit_full,seq_len,start_index
    

def _get_multi_ahead_start_index(cc,ob_portions=0.5):
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


def _evaluate_multi_ahead_accumulative(model,data_config,batch_size=1,ob_portions=0.5,acc_threshold=0.5,max_len=200):
    
    testf = os.path.join(data_config["dpath"], "test_quelevel.csv")
    df = pd.read_csv(testf)
    print("total sequence length is {}".format(len(df)))
    # max_len = data_config["maxlen"]
    y_pred_list = []
    y_true_list = []
    for i, row in df.iterrows():
        hist_q,hist_c,hist_r,hist_it,cq_full,cc_full,cr_full,cit_full,seq_len,start_index = _parser_row(row,data_config=data_config,ob_portions=ob_portions)
        if i%10==0:
            print(f"predict step {i}")

        seq_y_pred_hist = [cr_full[start_index]]
        for i in range(start_index,seq_len):
            cur_q = cq_full[start_index:i+1].unsqueeze(0)
            cur_c = cc_full[start_index:i+1].unsqueeze(0)
            cur_r = torch.tensor(seq_y_pred_hist).unsqueeze(0).to(device)
            cur_it = cit_full[start_index:i+1].unsqueeze(0)
            # print(f"cur_q is {cur_q} shape is {cur_q.shape}")
            # print(f"cur_r is {cur_r} shape is {cur_r.shape}")
            cq = torch.cat([hist_q,cur_q],axis=1)[:,-max_len:]
            cc = torch.cat([hist_c,cur_c],axis=1)[:,-max_len:]
            cr = torch.cat([hist_r,cur_r],axis=1)[:,-max_len:]
            cit = torch.cat([hist_it,cur_it],axis=1)[:,-max_len:]
            # print(f"cc_full is {cc_full}")
            # print(f"cr is {cr} shape is {cr.shape}")
            # print(f"cq is {cq} shape is {cq.shape}")
            data = [cq,cc,cr,cit]
            # print(f"cq.shape is {cq.shape}")
            cq,cc,cr,cit = [x.to(device) for x in data]#full sequence,[1,n]
            q,c,r,it = [x[:,:-1].to(device) for x in data]#[0,n-1]
            qshft,cshft,rshft,itshft = [x[:,1:].to(device) for x in data]#[1,n]
            data = {"cq":cq,"cc":cc,"cr":cr,"cit":cit,"q":q,"c":c,"r":r,"it":it,"qshft":qshft,"cshft":cshft,"rshft":rshft,"itshft":itshft}
            y_last_pred = predict_one_step(model,data,process=False)[:,-1][0]
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


def _evaluate_multi_ahead_help(model,data_config,batch_size,ob_portions=0.5,acc_threshold=0.5):
    """generate multi-ahead dataset

    Args:
        data_config (_type_): data_config
        ob_portions (float, optional): portions of observed student interactions. . Defaults to 0.5.

    Returns:
        dataset: new dataset for multi-ahead prediction
    """
    testf = os.path.join(data_config["dpath"], "test_quelevel.csv")
    df = pd.read_csv(testf)
    print("total sequence length is {}".format(len(df))) 
    y_pred_list = []
    y_true_list = []
    for i, row in df.iterrows():
        hist_q,hist_c,hist_r,hist_it,cq_full,cc_full,cr_full,cit_full,seq_len,start_index = _parser_row(row,data_config=data_config,ob_portions=ob_portions)
        if i%10==0:
            print(f"predict step {i}")
        cq_list = []
        cc_list = []
        cr_list = []
        cit_list = []
        
        for i in range(start_index,seq_len):
            cur_q = cq_full[i:i+1].unsqueeze(0)
            cur_c = cc_full[i:i+1].unsqueeze(0)
            cur_r = cr_full[i:i+1].unsqueeze(0)
            cur_it = cit_full[i:i+1].unsqueeze(0)
            cq_list.append(torch.cat([hist_q,cur_q],axis=1))
            cc_list.append(torch.cat([hist_c,cur_c],axis=1))
            cit_list.append(torch.cat([hist_it,cur_it],axis=1))
            cr_list.append(torch.cat([hist_r,cur_r],axis=1))
            y_true_list.append(cr_full[i].item())
        # print(f"cq_list is {len(cq_list)}")
        cq_ahead = torch.cat(cq_list,axis=0)
        cc_ahead = torch.cat(cc_list,axis=0)
        cr_ahead = torch.cat(cr_list,axis=0)
        cit_ahead = torch.cat(cit_list,axis=0)
        # print(f"cq_ahead shape is {cq_ahead.shape}")

        tensor_dataset = TensorDataset(cq_ahead,cc_ahead,cr_ahead,cit_ahead)
        dataloader = DataLoader(dataset=tensor_dataset,batch_size=batch_size) 

        for data in dataloader:
            cq,cc,cr,cit = [x.to(device) for x in data]#full sequence,[1,n]
            q,c,r,it = [x[:,:-1].to(device) for x in data]#[0,n-1]
            qshft,cshft,rshft,itshft = [x[:,1:].to(device) for x in data]#[1,n]
            data = {"cq":cq,"cc":cc,"cr":cr,"cit":cit,"q":q,"c":c,"r":r,"it":it,"qshft":qshft,"cshft":cshft,"rshft":rshft,"itshft":itshft}
            y = predict_one_step(model,data,process=False)[:,-1].detach().cpu().numpy().flatten()
            y_pred_list.extend(list(y))
    
    print(f"num of y_pred_list is {len(y_pred_list)}")
    print(f"num of y_true_list is {len(y_true_list)}")

    y_pred_list = np.array(y_pred_list)
    y_true_list = np.array(y_true_list)
    auc = metrics.roc_auc_score(y_true_list, y_pred_list)
    acc = metrics.accuracy_score(y_true_list, [1 if p >= acc_threshold else 0 for p in y_pred_list])

    return auc,acc

def lpkt_evaluate_multi_ahead(model,data_config,batch_size,ob_portions=0.5,acc_threshold=0.5,accumulative=False,max_len=200):
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
    model.eval()
    with torch.no_grad():
        if accumulative:
            print("predict use accumulative")
            auc,acc = _evaluate_multi_ahead_accumulative(model,data_config,batch_size=batch_size,ob_portions=ob_portions,acc_threshold=acc_threshold,max_len=max_len)
        else:
            print("predict use no accumulative")
            auc,acc = _evaluate_multi_ahead_help(model,data_config,batch_size=batch_size,ob_portions=ob_portions,acc_threshold=acc_threshold)
    return {"auc":auc,"acc":acc}


