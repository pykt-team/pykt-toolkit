from audioop import cross
from multiprocessing import reduction
import os, sys
from re import L
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def polyloss(logits, labels, epsilon=1.0):
#     # pt, CE, and Poly1 have shape [batch].
#     pt = tf.reduce_sum(labels * tf.nn.softmax(logits), axis=-1)
#     CE = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
#     Poly1 = CE + epsilon * (1 - pt)
#     return Poly1

def SoftMax(x, g=0, t=0.1):
    # x = np.array(x).cpu()
    log_x = (x + g) / t
    exp_x = log_x.exp()
    softmax_x = exp_x / torch.sum(exp_x)
    return softmax_x

def polyloss(model, logits, labels, epsilon=1.0, reduction="mean"):
    import torch.nn.functional as F
    labels_onehot = F.one_hot(labels, num_classes=model.num_c).to(device=logits.device, dtype=logits.dtype)
    pt = torch.sum(labels_onehot * SoftMax(F.softmax(logits, dim=-1)), dim=-1)
    # pt = torch.sum(labels_onehot * SoftMax(logits), dim=-1)
    CE = F.cross_entropy(input=logits, target=labels, reduction='none')
    poly1 = CE + epsilon * (1 - pt)
    if reduction == "mean":
        poly1 = poly1.mean()
    elif reduction == "sum":
        poly1 = poly1.sum()
    return poly1

# def binary_polyloss(logits, labels, epsilon=1.0, reduction="mean"):
#     # import torch.nn.functional as F
#     # labels_onehot = F.one_hot(labels, num_classes=model.num_c).to(device=logits.device, dtype=logits.dtype)
#     # pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
    
#     pt = torch.sum(logits)
#     CE = binary_cross_entropy(logits, labels, reduction="none")

#     # CE = F.cross_entropy(input=logits, target=labels, reduction='none')
#     poly1 = CE + epsilon * (1 - pt)
#     if reduction == "mean":
#         poly1 = poly1.mean()
#     elif reduction == "sum":
#         poly1 = poly1.sum()
#     return poly1

def cal_loss(model, ys, r, rshft, sm, preloss=[], epoch=0):
    model_name = model.model_name

    if model_name in ["cdkt", "cakt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # print(f"loss1: {y.shape}")
        loss1 = binary_cross_entropy(y.double(), t.double())

        # 1.2
        loss2 = 0
        if model.emb_type.endswith("mergetwo"):
            # if epoch <= 3:
            loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            # elif epoch <= 6:
            #     loss = model.l1*loss1+model.l2/2*ys[1]+model.l3/2*ys[2]
            # else:
            #     loss = loss1
        elif model.emb_type.find("predcurc") != -1:
            loss = model.l1*loss1+model.l2*ys[1]
        # if model.emb_type.find("predcurc") != -1:
        #     mask = sm == 1
        #     loss2 = cross_entropy(ys[1][mask], ys[2][mask])
        #     loss = model.l1*loss1+model.l2*loss2
        elif model.emb_type.endswith("addcc"):
            # print(f"loss1: {loss1}, loss2: {ys[1]}")
            loss = model.l1*loss1+model.l2*0.05*ys[1]
        elif model.emb_type.endswith("seq2seq") or model.emb_type.endswith("transpretrain"):
            # print(f"loss1: {model.l1*loss1}, loss2: {model.l2*ys[1]}")
            loss = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.endswith("predfuture") or model.emb_type.endswith("three"):
            # print(f"loss1: {model.l1*loss1}, loss2: {model.l2*ys[1]}")
            loss = model.l1*loss1+model.l2*ys[1]
        else:
            loss = loss1

        if model_name == "cakt":
            loss = loss + preloss[0]
        
        # # concept predict loss  ## 1.3
        # loss2 = 0        
        # # mask = sm == 1
        # # # print(ys[1][mask][0:5])
        # # # print(f"y: {y.shape}, ys[1][mask]: {ys[1][mask].shape}")
        # # assert y.shape[0] == ys[1][mask].shape[0]
        # # loss2 = polyloss(model, ys[1][mask], ys[2][mask])
        # y2 = (ys[1] * one_hot(ys[2].long(), model.num_c)).sum(-1)
        # y2 = torch.masked_select(y2, sm)
        # t2 = torch.ones(y2.shape[0]).to(device)
        # loss2 = binary_cross_entropy(y2.double(), t2.double())
        # loss = 0.5*loss1 + 0.5*loss2

        
    #elif model_name in ["dkt", "dkt_forget", "dkvmn", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes"]:
    elif model_name in ["dkt", "dkt_forget", "dkvmn", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt", "skvmn", "hawkes", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx"]:

        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())
    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y_next.double(), r_next.double())

        loss_r = binary_cross_entropy(y_curr.double(), r_curr.double()) # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
    elif model_name in ["akt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
    elif model_name == "lpkt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        criterion = nn.BCELoss(reduction='none')        
        loss = criterion(y, t).sum()
    
    return loss


def model_forward(model, data, epoch):
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    if model_name in ["dkt_forget"]:
        dcur, dgaps = data
    else:
        dcur = data
    q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]
    qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
    m, sm = dcur["masks"], dcur["smasks"]

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    if model_name in ["hawkes"]:
        ct = torch.cat((t[:,0:1], tshft), dim=1)

    # if model_name in ["cdkt"]: ## 1.3
    #     y, y2 = model(c.long(), r.long(), train=True)
    #     y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
    #     y2 = (y2 * one_hot(cshft.long(), model.num_c)).sum(-1)
    #     ys = [y, y2] # first: yshft
    if model_name in ["cdkt"]:
        # is_repeat = dcur["is_repeat"]
        y, y2, y3 = model(dcur, train=True)
        if model.emb_type.find("bkt") == -1:
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        # y2 = (y2 * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3] # first: yshft
    elif model_name in ["cakt"]:
        y, reg_loss, y2, y3 = model(dcur, train=True)#(cc.long(), cr.long(), cq.long(), train=True)
        ys = [y[:,1:], y2, y3]
        # ys = [y[:,1:], y2[:,1:], cshft] if model.emb_type.endswith("predcurc") else [y[:,1:]]
        preloss.append(reg_loss)
    elif model_name in ["lpkt"]:
        # cat = torch.cat((d["at_seqs"][:,0:1], dshft["at_seqs"]), dim=1)
        cit = torch.cat((dcur["itseqs"][:,0:1], dcur["shft_itseqs"]), dim=1)
    if model_name in ["dkt"]:
        y = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y) # first: yshft
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), dgaps)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn", "skvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:,1:])
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name in ["akt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx"]:               
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm, epoch)
        # at
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        pred_res, _ = model(c.long(), r.long(), p_adv)
        # second loss
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm, epoch)
        loss = loss + model.beta * adv_loss
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)  
    # cal loss
    elif model_name == "lpkt":

        # y = model(cq.long(), cr.long(), cat, cit.long())
        y = model(cq.long(), cr.long(), cit.long())
        ys.append(y[:, 1:])  
    elif model_name == "hawkes":
        # ct = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
        # csm = torch.cat((dcur["smasks"][:,0:1], dcur["smasks"]), dim=1)
        # y = model(cc[0:1,0:5].long(), cq[0:1,0:5].long(), ct[0:1,0:5].long(), cr[0:1,0:5].long(), csm[0:1,0:5].long())
        y = model(cc.long(), cq.long(), ct.long(), cr.long())#, csm.long())
        ys.append(y[:, 1:])
    elif model_name == "iekt":
        y,loss = model.train_one_step(data)
    if model_name not in ["atkt", "atktfix","iekt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss, epoch)
    return loss
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False):
    max_auc, best_epoch = 0, -1
    train_step = 0
    if model.model_name=='lpkt':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step+=1
            if model.model_name=='iekt':
                model.model.train()
            else:
                model.train()
            loss = model_forward(model, data, i)
            opt.zero_grad()
            loss.backward()#compute gradients 
            opt.step()#update model’s parameters
                
            loss_mean.append(loss.detach().cpu().numpy())
            if model.model_name == "gkt" and train_step%10==0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text = text,fuc_name="train_model")
        if model.model_name=='lpkt':
            scheduler.step()#update each epoch
        loss_mean = np.mean(loss_mean)
        auc, acc = evaluate(model, valid_loader, model.model_name)
        ### atkt 有diff， 以下代码导致的
        ### auc, acc = round(auc, 4), round(acc, 4)

        if auc > max_auc:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            # window_testauc, window_testacc = -1, -1
            validauc, validacc = round(auc, 4), round(acc, 4)#model.evaluate(valid_loader, emb_type)
            # trainauc, trainacc = model.evaluate(train_loader, emb_type)
            testauc, testacc, window_testauc, window_testacc = round(testauc, 4), round(testacc, 4), round(window_testauc, 4), round(window_testacc, 4)
            max_auc = round(max_auc, 4)
        print(f"Epoch: {i}, validauc: {validauc}, validacc: {validacc}, best epoch: {best_epoch}, best auc: {max_auc}, loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(f"            testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

        if i - best_epoch >= 10:
            break
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch
