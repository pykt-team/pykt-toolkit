import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot
from sklearn import metrics

import pandas as pd

device = "cpu" if not torch.cuda.is_available() else "cuda"

def save_cur_predict_result(dres, q, r, d, t, m, sm, p):
    # dres, q, r, qshft, rshft, m, sm, y
    results = []
    for i in range(0, t.shape[0]):
        cps = torch.masked_select(p[i], sm[i]).detach().cpu()
        cts = torch.masked_select(t[i], sm[i]).detach().cpu()
    
        cqs = torch.masked_select(q[i], m[i]).detach().cpu()
        crs = torch.masked_select(r[i], m[i]).detach().cpu()

        cds = torch.masked_select(d[i], sm[i]).detach().cpu()

        qs, rs, ts, ps, ds = [], [], [], [], []
        for cq, cr in zip(cqs.int(), crs.int()):
            qs.append(cq.item())
            rs.append(cr.item())
        for ct, cp, cd in zip(cts.int(), cps, cds.int()):
            ts.append(ct.item())
            ps.append(cp.item())
            ds.append(cd.item())
        try:
            auc = metrics.roc_auc_score(
                y_true=np.array(ts), y_score=np.array(ps)
            )
            
        except Exception as e:
            # print(e)
            auc = -1
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        dres[len(dres)] = [qs, rs, ds, ts, ps, prelabels, auc, acc]
        results.append(str([qs, rs, ds, ts, ps, prelabels, auc, acc]))
    return "\n".join(results)

def evaluate(model, test_loader, model_name, save_path=""):
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")
    with torch.no_grad():
        y_trues = []
        y_scores = []
        dres = dict()
        for data in test_loader:
            if model_name in ["dkt_forget"]:
                q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
            else:
                q, c, r, qshft, cshft, rshft, m, sm = data
            q, c, r, qshft, cshft, rshft, m, sm = q.to(device), c.to(device), r.to(device), qshft.to(device), cshft.to(device), rshft.to(device), m.to(device), sm.to(device)

            model.eval()

            # print(f"before y: {y.shape}")
            cq = torch.cat((q[:,0:1], qshft), dim=1)
            cc = torch.cat((c[:,0:1], cshft), dim=1)
            cr = torch.cat((r[:,0:1], rshft), dim=1)
            if model_name in ["dkt", "dkt+"]:
                y = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["dkt_forget"]:
                y = model(c.long(), r.long(), d, dshft)
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["dkvmn"]:
                y = model(cc.long(), cr.long())
                y = y[:,1:]
            elif model_name in ["kqn", "sakt"]:
                y = model(c.long(), r.long(), cshft.long())
            elif model_name == "saint":
                y = model(cq.long(), cc.long(), r.long())
                y = y[:, 1:]
            elif model_name == "akt":                                
                y, reg_loss = model(cc.long(), cr.long(), cq.long())
                y = y[:,1:]
            elif model_name in ["atkt", "atktfix"]:
                y, _ = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name == "gkt":
                y = model(cc.long(), cr.long())
                
            # print(f"after y: {y.shape}")
            # save predict result
            if save_path != "":
                result = save_cur_predict_result(dres, c, r, cshft, rshft, m, sm, y)
                fout.write(result+"\n")

            y = torch.masked_select(y, sm).detach().cpu()
            t = torch.masked_select(rshft, sm).detach().cpu()

            y_trues.append(t.numpy())
            y_scores.append(y.numpy())
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)

        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
    # if save_path != "":
    #     pd.to_pickle(dres, save_path+".pkl")
    return auc, acc

def early_fusion(curhs, model, model_name):
    if model_name == "dkvmn":
        p = model.p_layer(model.dropout_layer(curhs[0]))
        p = torch.sigmoid(p)
        p = p.squeeze(-1)
    elif model_name == "akt":
        output = model.out(curhs[0]).squeeze(-1)
        m = nn.Sigmoid()
        p = m(output)
    elif model_name == "saint":
        p = model.out(model.dropout(curhs[0]))
        p = torch.sigmoid(p).squeeze(-1)
    elif model_name == "sakt":
        p = torch.sigmoid(model.pred(model.dropout_layer(curhs[0]))).squeeze(-1)
    elif model_name == "kqn":
        logits = torch.sum(curhs[0] * curhs[1], dim=1) # (batch_size, max_seq_len)
        p = model.sigmoid(logits)
    return p

def late_fusion(dcur, curdf, fusion_type=["mean", "vote", "all"]):
    if "mean" in fusion_type:
        dcur.setdefault("late_mean", [])
        dcur["late_mean"].append(round(curdf["preds"].mean().astype(float), 4))
    if "vote" in fusion_type:
        dcur.setdefault("late_vote", [])
        labeldf = curdf["preds"] >= 0.5
        counts = labeldf.value_counts()
        poslen = 0 if True not in counts else counts[True]
        dcur["late_vote"].append(1.0 if poslen / curdf.shape[0] >= 0.5 else 0.0)
    if "all" in fusion_type:
        dcur.setdefault("late_all", [])
        labeldf = curdf["preds"] >= 0.5
        counts = labeldf.value_counts()
        dcur["late_all"].append(1.0 if True in counts and False not in counts else 0.0)
    return 

def effective_fusion(df, model, model_name, fusion_type):
    dres = dict()
    df = df.groupby("qidx", as_index=True, sort=True)#.mean()

    curhs, curr = [[], []], []
    dcur = {"late_trues": [], "qidxs": [], "questions": [], "concepts": [], "row": [], "concept_preds": []}
    for ui in df:
        # 一题一题处理
        curdf = ui[1]
        if model_name in ["dkvmn", "akt", "saint", "sakt"]:
            curhs[0].append(curdf["hidden"].mean().astype(float))
        elif model_name == "kqn":
            curhs[0].append(curdf["ek"].mean().astype(float))
            curhs[1].append(curdf["es"].mean().astype(float))
        else:
            # print(f"model: {model_name} has no early fusion res!")
            pass

        curr.append(curdf["response"].mean().astype(int))
        dcur["late_trues"].append(curdf["response"].mean().astype(int))
        dcur["qidxs"].append(ui[0])
        dcur["row"].append(curdf["row"].mean().astype(int))
        dcur["questions"].append(",".join([str(int(s)) for s in curdf["questions"].tolist()]))
        dcur["concepts"].append(",".join([str(int(s)) for s in curdf["concepts"].tolist()]))
        late_fusion(dcur, curdf)
        # save original predres in concepts
        dcur["concept_preds"].append(",".join([str(round(s, 4)) for s in (curdf["preds"].tolist())]))

    for key in dcur:
        dres.setdefault(key, [])
        dres[key].append(np.array(dcur[key]))
    # early fusion
    if "early_fusion" in fusion_type and model_name in ["dkvmn", "akt", "saint", "sakt", "kqn"]:
        curhs = [torch.tensor(curh).float().to(device) for curh in curhs]
        curr = torch.tensor(curr).long().to(device)
        p = early_fusion(curhs, model, model_name)
        dres.setdefault("early_trues", [])
        dres["early_trues"].append(curr.cpu().numpy())
        dres.setdefault("early_preds", [])
        dres["early_preds"].append(p.cpu().numpy())
    return dres

def group_fusion(dmerge, model, model_name, fusion_type, fout):
    hs, sms, cq, cc, rs, ps, qidxs, rests, orirows = dmerge["hs"], dmerge["sm"], dmerge["cq"], dmerge["cc"], dmerge["cr"], dmerge["y"], dmerge["qidxs"], dmerge["rests"], dmerge["orirow"]
    if cq.shape[1] == 0:
        cq = cc
    
    alldfs, drest = [], dict() # not predict infos!
    # print(f"real bz in group fusion: {rs.shape[0]}")
    realbz = rs.shape[0]
    for bz in range(rs.shape[0]):
        cursm = ([0] + sms[bz].cpu().tolist())
        curqidxs = ([-1] + qidxs[bz].cpu().tolist())
        currests = ([-1] + rests[bz].cpu().tolist())
        currows = ([-1] + orirows[bz].cpu().tolist())
        curps = ([-1] + ps[bz].cpu().tolist())
        # print(f"qid: {len(curqidxs)}, select: {len(cursm)}, response: {len(rs[bz].cpu().tolist())}, preds: {len(curps)}")
        df = pd.DataFrame({"qidx": curqidxs, "rest": currests, "row": currows, "select": cursm, 
                "questions": cq[bz].cpu().tolist(), "concepts": cc[bz].cpu().tolist(), "response": rs[bz].cpu().tolist(), "preds": curps})
        if model_name in ["dkvmn", "akt", "saint", "sakt"]:
            df["hidden"] = [np.array(a) for a in hs[0][bz].cpu().tolist()]
        elif model_name == "kqn":
            df["ek"] = [np.array(a) for a in hs[0][bz].cpu().tolist()]
            df["es"] = [np.array(a) for a in hs[1][bz].cpu().tolist()]

        df = df[df["select"] != 0]
        alldfs.append(df)
    
    effective_dfs, rest_start = [], -1
    flag = False
    for i in range(len(alldfs) - 1, -1, -1):
        df = alldfs[i]
        counts = (df["rest"] == 0).value_counts()
        if not flag and False not in counts: # has no question rest > 0
            flag =True
            effective_dfs.append(df)
            rest_start = i + 1
        elif flag:
            effective_dfs.append(df)
    if rest_start == -1:
        rest_start = 0
    # merge rest
    for key in dmerge.keys():
        if key == "hs":
            drest[key] = []
            if model_name in ["dkvmn", "akt", "saint", "sakt"]:
                drest[key] = [dmerge[key][0][rest_start:]]
            elif model_name == "kqn":
                drest[key] = [dmerge[key][0][rest_start:], dmerge[key][1][rest_start:]]                
        else:
            drest[key] = dmerge[key][rest_start:] 
    restlen = drest["cr"].shape[0]

    dfs = dict()
    for df in effective_dfs:
        for i, row in df.iterrows():
            for key in row.keys():
                dfs.setdefault(key, [])
                dfs[key].extend([row[key]])
    df = pd.DataFrame(dfs)
    print(f"real bz: {realbz}, effective_dfs: {len(effective_dfs)}, rest_start: {rest_start}, drestlen: {restlen}, predict infos: {df.shape}")

    if df.shape[0] == 0:
        return {}, drest

    dres = effective_fusion(df, model, model_name, fusion_type)
            
    dfinal = dict()
    for key in dres:
        dfinal[key] = np.concatenate(dres[key], axis=0)
    early = False
    if model_name in ["dkvmn", "akt", "saint", "sakt", "kqn"]:
        early = True
    save_question_res(dfinal, fout, early)
    return dfinal , drest

def save_question_res(dres, fout, early=False):
    # print(f"dres: {dres.keys()}")
    # qidxs, late_trues, late_mean, late_vote, late_all, early_trues, early_preds
    for i in range(0, len(dres["qidxs"])):
        row, qidx, qs, cs, lt, lm, lv, la = dres["row"][i], dres["qidxs"][i], dres["questions"][i], dres["concepts"][i], \
            dres["late_trues"][i], dres["late_mean"][i], dres["late_vote"][i], dres["late_all"][i]
        conceptps = dres["concept_preds"][i]
        curres = [row, qidx, qs, cs, conceptps, lt, lm, lv, la]
        if early:
            et, ep = dres["early_trues"][i], dres["early_preds"][i]
            curres = curres + [et, ep]
        curstr = "\t".join([str(round(s, 4)) if type(s) == type(0.1) or type(s) == np.float32 else str(s) for s in curres])
        fout.write(curstr + "\n")

def evaluate_question(model, test_loader, model_name, fusion_type=["early_fusion", "late_fusion"], save_path=""):
    # dkt / dkt+ / dkt_forget / atkt: give past -> predict all. has no early fusion!!!
    # dkvmn / akt / saint: give cur -> predict cur
    # sakt: give past+cur -> predict cur
    # kqn: give past+cur -> predict cur
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")
        if model_name in ["dkvmn", "akt", "saint", "sakt", "kqn"]:
            fout.write("\t".join(["orirow", "qidx", "questions", "concepts", "concept_preds", "late_trues", "late_mean", "late_vote", "late_all", "early_trues", "early_preds"]) + "\n")
        else:
            fout.write("\t".join(["orirow", "qidx", "questions", "concepts", "concept_preds", "late_trues", "late_mean", "late_vote", "late_all"]) + "\n")
    with torch.no_grad():
        dinfos = dict()
        dhistory = dict()
        history_keys = ["hs", "sm", "cq", "cc", "cr", "y", "qidxs", "rests", "orirow"]
        # for key in history_keys:
        #     dhistory[key] = []
        y_trues, y_scores = [], []
        lenc = 0
        for data in test_loader:
            if model_name in ["dkt_forget"]:
                q, c, r, qshft, cshft, rshft, m, sm, d, dshft, dqtest = data
            else:
                q, c, r, qshft, cshft, rshft, m, sm, dqtest = data
            q, c, r, qshft, cshft, rshft, m, sm = q.to(device), c.to(device), r.to(device), qshft.to(device), cshft.to(device), rshft.to(device), m.to(device), sm.to(device)
            qidxs, rests, orirow = dqtest["qidxs"], dqtest["rests"], dqtest["orirow"]
            lenc += q.shape[0]
            print("="*20)
            print(f"start predict seqlen: {lenc}")
            model.eval()

            # print(f"before y: {y.shape}")
            cq = torch.cat((q[:,0:1], qshft), dim=1)
            cc = torch.cat((c[:,0:1], cshft), dim=1)
            cr = torch.cat((r[:,0:1], rshft), dim=1)
            dcur = dict()
            if model_name in ["dkvmn"]:
                y, h = model(cc.long(), cr.long(), True)
                y = y[:,1:]
            elif model_name == "akt":
                y, reg_loss, h = model(cc.long(), cr.long(), cq.long(), True)
                y = y[:,1:]
            elif model_name == "saint":
                y, h = model(cq.long(), cc.long(), r.long(), True)
                y = y[:,1:]
            elif model_name == "sakt":
                y, h = model(c.long(), r.long(), cshft.long(), True)
                start_hemb = torch.tensor([-1] * (h.shape[0] * h.shape[2])).reshape(h.shape[0], 1, h.shape[2]).to(device)
                # print(start_hemb.shape, h.shape)
                h = torch.cat((start_hemb, h), dim=1) # add the first hidden emb
            elif model_name == "kqn":
                y, ek, es = model(c.long(), r.long(), cshft.long(), True)
                # print(f"ek: {ek.shape},  es: {es.shape}")
                start_hemb = torch.tensor([-1] * (ek.shape[0] * ek.shape[2])).reshape(ek.shape[0], 1, ek.shape[2]).to(device)
                ek = torch.cat((start_hemb, ek), dim=1) # add the first hidden emb
                es = torch.cat((start_hemb, es), dim=1) # add the first hidden emb  
            elif model_name in ["dkt", "dkt+"]:
                y = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["dkt_forget"]:
                y = model(c.long(), r.long(), d, dshft)
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["atkt", "atktfix"]:
                y, _ = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name == "gkt":
                y = model(cc.long(), cr.long())
            

            concepty = torch.masked_select(y, sm).detach().cpu()
            conceptt = torch.masked_select(rshft, sm).detach().cpu()

            y_trues.append(conceptt.numpy())
            y_scores.append(concepty.numpy())

            # hs, sms, rs, ps, qidxs, model, model_name, fusion_type
            hs = []
            if model_name == "kqn":
                hs = [ek, es]
            elif model_name in ["dkvmn", "akt", "saint", "sakt"]:
                hs = [h]
            dcur["hs"], dcur["sm"], dcur["cq"], dcur["cc"], dcur["cr"], dcur["y"], dcur["qidxs"], dcur["rests"], dcur["orirow"] = hs, sm, cq, cc, cr, y, qidxs, rests, orirow
            # merge history
            dmerge = dict()
            for key in history_keys:
                if len(dhistory) == 0:
                    dmerge[key] = dcur[key]
                else:
                    if key == "hs":
                        dmerge[key] = []
                        if model_name == "kqn":
                            dmerge[key] = [[], []]
                            dmerge[key][0] = torch.cat((dhistory[key][0], dcur[key][0]), dim=0)
                            dmerge[key][1] = torch.cat((dhistory[key][1], dcur[key][1]), dim=0)
                        elif model_name in ["dkvmn", "akt", "saint", "sakt"]:
                            dmerge[key] = [torch.cat((dhistory[key][0], dcur[key][0]), dim=0)]                            
                    else:
                        dmerge[key] = torch.cat((dhistory[key], dcur[key]), dim=0)
                
            dcur, dhistory = group_fusion(dmerge, model, model_name, fusion_type, fout)
            for key in dcur:
                dinfos.setdefault(key, [])
                dinfos[key].append(dcur[key])

            if "early_fusion" in dinfos and "late_fusion" in dinfos:
                assert dinfos["early_trues"][-1].all() == dinfos["late_trues"][-1].all()
            # import sys
            # sys.exit()
        # ori concept eval
        aucs, accs = dict(), dict()
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        aucs["concepts"] = auc
        accs["concepts"] = acc

        print(f"dinfos: {dinfos.keys()}")
        for key in dinfos:
            if key not in ["late_mean", "late_vote", "late_all", "early_preds"]:
                continue
            ts = np.concatenate(dinfos['late_trues'], axis=0) # early_trues == late_trues
            ps = np.concatenate(dinfos[key], axis=0)
            print(f"key: {key}, ts.shape: {ts.shape}, ps.shape: {ps.shape}")
            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
            prelabels = [1 if p >= 0.5 else 0 for p in ps]
            acc = metrics.accuracy_score(ts, prelabels)
            aucs[key] = auc
            accs[key] = acc
    return aucs, accs


def log2(t):
    import math
    return round(math.log(t+1, 2))

def calC(row, data_config):
    repeated_gap, sequence_gap, past_counts = [], [], []
    uid = row["uid"]
    # default: concepts
    skills = row["concepts"].split(",")
    timestamps = row["timestamps"].split(",")
    dlastskill, dcount = dict(), dict()
    pret = None
    idx = -1
    for s, t in zip(skills, timestamps):
        idx += 1
        s, t = int(s), int(t)
        if s not in dlastskill or s == -1:
            curRepeatedGap = 0
        else:
            curRepeatedGap = log2((t - dlastskill[s]) / 1000 / 60) + 1 # minutes
        dlastskill[s] = t

        repeated_gap.append(curRepeatedGap)
        if pret == None or t == -1:
            curLastGap = 0
        else:
            curLastGap = log2((t - pret) / 1000 / 60) + 1
        pret = t
        sequence_gap.append(curLastGap)

        dcount.setdefault(s, 0)
        ccount = log2(dcount[s])
        ccount = data_config["num_pcount"] - 1 if ccount >= data_config["num_pcount"] else ccount
        past_counts.append(ccount)
        
        dcount[s] += 1
    return repeated_gap, sequence_gap, past_counts           

def get_info_dkt_forget(row, data_config):
    dforget = dict()
    rgap, sgap, pcount = calC(row, data_config)

    ## TODO
    dforget["rgaps"], dforget["sgaps"], dforget["pcounts"] = rgap, sgap, pcount
    return dforget

def evaluate_splitpred_question(model, data_config, testf, model_name, save_path="", use_pred=False, train_ratio=0.2, atkt_pad=False):
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")
    with torch.no_grad():
        y_trues = []
        y_scores = []
        dres = dict()
        idx = 0
        df = pd.read_csv(testf)
        dcres, dqres = {"trues": [], "preds": []}, {"trues": [], "late_mean": [], "late_vote": [], "late_all": []}
        for i, row in df.iterrows():
            # print(f"idx: {idx}")
            # if idx == 2:
            #     import sys
            #     sys.exit()
            model.eval()

            dforget = dict() if model_name != "dkt_forget" else get_info_dkt_forget(row, data_config)

            concepts, responses = row["concepts"].split(","), row["responses"].split(",")
            curl = len(responses)

            # print("="*20)
            is_repeat = ["0"] * curl if "is_repeat" not in row else row["is_repeat"].split(",")
            is_repeat = [int(s) for s in is_repeat]
            questions = [] if "questions" not in row else row["questions"].split(",")

            qlen, qtrainlen, ctrainlen = get_cur_teststart(is_repeat, train_ratio)
            # print(f"idx: {idx}, qlen: {qlen}, qtrainlen: {qtrainlen}, ctrainlen: {ctrainlen}")
            # print(concepts)
            # print(responses)
            cq = torch.tensor([int(s) for s in questions]).to(device)
            cc = torch.tensor([int(s) for s in concepts]).to(device)
            cr = torch.tensor([int(s) for s in responses]).to(device)
            # print(f"cc: {cc[0:ctrainlen]}")
            curcin, currin = cc[0:ctrainlen].unsqueeze(0), cr[0:ctrainlen].unsqueeze(0)
            curqin = cq[0:ctrainlen].unsqueeze(0) if cq.shape[0] > 0 else cq
            curdforget = dict()
            for key in dforget:
                dforget[key] = torch.tensor(dforget[key]).to(device)
                curdforget[key] = dforget[key][0:ctrainlen].unsqueeze(0)
            # print(f"curcin: {curcin}")
            t = ctrainlen

            ### 如果不用预测结果，可以从这里并行了
            
            if not use_pred:
                uid, end = row["uid"], curl
                qidx = qtrainlen
                qidxs, ctrues, cpreds = predict_each_group2(curdforget, dforget, is_repeat, qidx, uid, idx, curqin, curcin, currin, model_name, model, t, cq, cc, cr, end, fout, atkt_pad)
                # 计算
                save_currow_question_res(idx, dcres, dqres, qidxs, ctrues, cpreds, uid, fout)
            else:
                qidx = qtrainlen
                while t < curl:
                    rtmp = [t]
                    for k in range(t+1, curl):
                        if is_repeat[k] != 0:
                            rtmp.append(k)
                        else:
                            break
                    # dfshape = curdforget["rgaps"].shape
                    # print(f"currin: {currin.shape}, curdforget: {dfshape}, rtmp: {rtmp}")
                    # print(f"rtmp: {rtmp}")
                    end = rtmp[-1]+1
                    uid = row["uid"]
                    # if use_pred:
                    curqin, curcin, currin, curdforget, ctrues, cpreds = predict_each_group(curdforget, dforget, is_repeat, qidx, uid, idx, curqin, curcin, currin, model_name, model, t, cq, cc, cr, end, fout, atkt_pad)
                    # else:
                    #     ctrues, cpreds = [], []
                    #     for m in range(t, end):
                    #         nextqin, nextcin, nextrin, nextdforget, ctrue, cpred = predict_each_group(curdforget, dforget, is_repeat, qidx, uid, idx, curqin, curcin, currin, model_name, model, m, cq, cc, cr, m+1, fout)
                    #         ctrues.extend(ctrue)
                    #         cpreds.extend(cpred)
                    
                    late_mean, late_vote, late_all = save_each_question_res(dcres, dqres, ctrues, cpreds)    
                    # print("\t".join([str(idx), str(uid), str(k), str(qidx), str(late_mean), str(late_vote), str(late_all)]))    
                    fout.write("\t".join([str(idx), str(uid), str(qidx), str(late_mean), str(late_vote), str(late_all)]) + "\n")      
                    t = end
                    qidx += 1
            idx += 1

        dfinal = cal_predres(dcres, dqres)
        for key in dfinal:
            fout.write(key + "\t" + str(dfinal[key]) + "\n")
    return dfinal

def get_cur_teststart(is_repeat, train_ratio):
    curl = len(is_repeat)
    # print(is_repeat)
    qlen = is_repeat.count(0)
    qtrainlen = int(qlen * train_ratio)
    qtrainlen = 1 if qtrainlen == 0 else qtrainlen
    qtrainlen = qtrainlen - 1 if qtrainlen == qlen else qtrainlen
    # get real concept len
    ctrainlen, qidx = 0, 0
    i = 0
    while i < curl:
        if is_repeat[i] == 0:
            qidx += 1
        # print(f"i: {i}, curl: {curl}, qidx: {qidx}, qtrainlen: {qtrainlen}")
        # qtrainlen = 7 if qlen>7 else qtrainlen
        if qidx == qtrainlen:
            break
        i += 1
    for j in range(i+1, curl):
        if is_repeat[j] == 0:
            ctrainlen = j
            break
    return qlen, qtrainlen, ctrainlen

def predict_each_group(curdforget, dforget, is_repeat, qidx, uid, idx, curqin, curcin, currin, model_name, model, t, cq, cc, cr, end, fout, atkt_pad=False, maxlen=200):
    """use the predict result as next question input
    """
    nextcin, nextrin = curcin, currin
    import copy
    nextdforget = copy.deepcopy(curdforget)
    ctrues, cpreds = [], []
    for k in range(t, end):
        qin, cin, rin = curqin, curcin, currin
        # 输入长度大于200时，截断
        # print("cin: ", cin)
        start = 0
        cinlen = cin.shape[1]
        if cinlen >= maxlen - 1:
            start = cinlen - maxlen + 1
        
        cin, rin = cin[:,start:], rin[:,start:]
        # print(f"start: {start}, cin: {cin.shape}")
        if cq.shape[0] > 0:
            qin = qin[:, start:]
        # print(f"start: {start}, cin: {cin.shape}")
        cout, true = cc.long()[k], cr.long()[k] # 当前预测的是第k个
        qout = None if cq.shape[0] == 0 else cq.long()[k]
        if model_name in ["dkt", "dkt+"]:
            y = model(cin.long(), rin.long())
            # print(y)
            pred = y[0][-1][cout.item()]
        elif model_name == "dkt_forget":
            din = dict()
            for key in curdforget:
                din[key] = curdforget[key][:,start:]
            dcur = dict()
            for key in dforget:
                curd = torch.tensor([[dforget[key][k]]]).long().to(device)
                dcur[key] = torch.cat((din[key][:,1:], curd), axis=1)
                # print(f"cin: {cin.shape}, dcur key: {dcur[key].shape}")
            # if idx == 13:
            #     print(f"input to dktforget ! cin: {cin.shape}, k: {k}")
            #     for key in dcur:
            #         print(key, dcur[key].shape, din[key].shape, dcur[key], din[key])
            y = model(cin.long(), rin.long(), din, dcur)
            pred = y[0][-1][cout.item()]
        elif model_name in ["kqn", "sakt"]:
            curc = torch.tensor([[cout.item()]]).to(device)
            cshft = torch.cat((cin[:,1:],curc), axis=1)
            y = model(cin.long(), rin.long(), cshft.long())
            pred = y[0][-1]
        elif model_name == "saint":
            #### 输入有question！
            if qout != None:
                curq = torch.tensor([[qout.item()]]).to(device)
                qin = torch.cat((qin, curq), axis=1)
            curc = torch.tensor([[cout.item()]]).to(device)
            cin = torch.cat((cin, curc), axis=1)
            
            y = model(qin.long(), cin.long(), rin.long())
            pred = y[0][-1]
        elif model_name in ["atkt", "atktfix"]:
            if atkt_pad == True:
                oricinlen = cin.shape[1]
                padlen = maxlen-1-oricinlen
                # print(f"padlen: {padlen}, cin: {cin.shape}")
                pad = torch.tensor([0]*(padlen)).unsqueeze(0).to(device)
                # curc = torch.tensor([[cout.item()]]).to(device)
                # cshft = torch.cat((cin[:,1:],curc), axis=1)
                cin = torch.cat((cin, pad), axis=1)
                rin = torch.cat((rin, pad), axis=1)
            y, _ = model(cin.long(), rin.long())
            # print(f"y: {y}")
            if atkt_pad == True:
                # print(f"use idx: {oricinlen-1}")
                pred = y[0][oricinlen-1][cout.item()]
            else:
                pred = y[0][-1][cout.item()]
        elif model_name in ["dkvmn"]:
            curc, curr = torch.tensor([[cout.item()]]).to(device), torch.tensor([[true.item()]]).to(device)
            cin, rin = torch.cat((cin, curc), axis=1), torch.cat((rin, curr), axis=1)
            # print(f"cin: {cin.shape}, curc: {curc.shape}")
            # 应该用预测的r更新memory value，但是这里一个知识点一个知识点预测，所以curr不起作用！
            y = model(cin.long(), rin.long())
            pred = y[0][-1]
        elif model_name == "akt":  
            #### 输入有question！     
            if qout != None:
                curq = torch.tensor([[qout.item()]]).to(device)
                qin = torch.cat((qin, curq), axis=1)
            # curr不起作用！当前预测不用curr，实际用的历史的也不一定是true，用的是rin，可能是预测，可能是历史
            curc, curr = torch.tensor([[cout.item()]]).to(device), torch.tensor([[1]]).to(device)
            cin, rin = torch.cat((cin, curc), axis=1), torch.cat((rin, curr), axis=1)

            y, reg_loss = model(cin.long(), rin.long(), qin.long())
            pred = y[0][-1]
        elif model_name == "gkt":
            curc, curr = torch.tensor([[cout.item()]]).to(device), torch.tensor([[1]]).to(device)
            cin, rin = torch.cat((cin, curc), axis=1), torch.cat((rin, curr), axis=1)
            y = model(cin.long(), rin.long())
            # print(f"y.shape is {y.shape},cin shape is {cin.shape}")
            pred = y[0][-1]
        
        predl = 1 if pred.item() >= 0.5 else 0
        cpred = torch.tensor([[predl]]).to(device)

        nextqin = cq[0:k+1].unsqueeze(0) if cq.shape[0] > 0 else qin
        nextcin = cc[0:k+1].unsqueeze(0)
        nextrin = torch.cat((nextrin, cpred), axis=1)### change!!

        # print(f"nextqin: {nextqin.shape}")
        # update nextdforget
        if model_name == "dkt_forget":
            for key in nextdforget:
                curd = torch.tensor([[dforget[key][k]]]).long().to(device)
                nextdforget[key] = torch.cat((nextdforget[key], curd), axis=1)
        # print(f"bz: {bz}, t: {t}, pred: {pred}, true: {true}")

        # save pred res
        ctrues.append(true.item())
        cpreds.append(pred.item())

        # output
        clist, rlist = cin.squeeze(0).long().tolist()[0:k], rin.squeeze(0).long().tolist()[0:k]
        # print("\t".join([str(idx), str(uid), str(k), str(qidx), str(is_repeat[t:end]), str(len(clist)), str(clist), str(rlist), str(cout.item()), str(true.item()), str(pred.item()), str(predl)]))
        fout.write("\t".join([str(idx), str(uid), str(k), str(qidx), str(is_repeat[t:end]), str(len(clist)), str(clist), str(rlist), str(cout.item()), str(true.item()), str(pred.item()), str(predl)]) + "\n")
    # nextcin, nextrin = nextcin.unsqueeze(0), nextrin.unsqueeze(0)
    return nextqin, nextcin, nextrin, nextdforget, ctrues, cpreds

def save_each_question_res(dcres, dqres, ctrues, cpreds):
    # save res
    high, low = [], []
    for true, pred in zip(ctrues, cpreds):
        dcres["trues"].append(true)
        dcres["preds"].append(pred)
        if pred >= 0.5:
            high.append(pred)
        else:
            low.append(pred)
    cpreds = np.array(cpreds)
    late_mean = np.mean(cpreds)
    correctnum = list(cpreds>=0.5).count(True)
    late_vote = np.mean(high) if correctnum / len(cpreds) >= 0.5 else np.mean(low)
    late_all = np.mean(high) if correctnum == len(cpreds) else np.mean(low)
    assert len(set(ctrues)) == 1
    dqres["trues"].append(dcres["trues"][-1])
    dqres["late_mean"].append(late_mean)
    dqres["late_vote"].append(late_vote)
    dqres["late_all"].append(late_all)
    return late_mean, late_vote, late_all

def cal_predres(dcres, dqres):
    dres = dict()#{"concept": [], "late_mean": [], "late_vote": [], "late_all": []}

    ctrues, cpreds = np.array(dcres["trues"]), np.array(dcres["preds"])
    # print(f"key: concepts, ts.shape: {ctrues.shape}, ps.shape: {cpreds.shape}")
    auc = metrics.roc_auc_score(y_true=ctrues, y_score=cpreds)
    prelabels = [1 if p >= 0.5 else 0 for p in cpreds]
    acc = metrics.accuracy_score(ctrues, prelabels)

    dres["concepts"] = [len(cpreds), auc, acc]

    qtrues = np.array(dqres["trues"])
    for key in dqres:
        if key == "trues":
            continue
        preds = np.array(dqres[key])
        # print(f"key: {key}, ts.shape: {qtrues.shape}, ps.shape: {preds.shape}")
        auc = metrics.roc_auc_score(y_true=qtrues, y_score=preds)
        prelabels = [1 if p >= 0.5 else 0 for p in preds]
        acc = metrics.accuracy_score(qtrues, prelabels)
        dres[key] = [len(preds), auc, acc]
    return dres

def prepare_data(model_name, is_repeat, qidx, curqin, curcin, currin, curdforget, cq, cc, cr, dforget, t, end, maxlen=200):
    dqshfts, dcshfts, drshfts, dds, ddshfts = [], [], [], dict(), dict()
    dqs, dcs, drs = [], [], []
    qidxs = []
    qstart = qidx-1
    for k in range(t, end):
        if is_repeat[k] == 0:
            qstart += 1
            qidxs.append(qstart)
        else:
            qidxs.append(qstart)
        # get start
        start = 0
        cinlen = curcin.shape[1]
        if cinlen >= maxlen - 1:
            start = cinlen - maxlen + 1

        curc, curr = cc.long()[k], cr.long()[k]
        curc, curr = torch.tensor([[curc.item()]]).to(device), torch.tensor([[curr.item()]]).to(device)
        dcs.append(curcin[:, start:])
        drs.append(currin[:, start:])

        curc, curr = torch.cat((curcin[:, start+1:], curc), axis=1), torch.cat((currin[:, start+1:], curr), axis=1)
        dcshfts.append(curc)
        drshfts.append(curr)
        if cq.shape[0] > 0:
            curq = cq.long()[k]
            curq = torch.tensor([[curq.item()]]).to(device)

            dqs.append(curqin[:, start:])
            curq = torch.cat((curqin[:, start+1:], curq), axis=1)
            dqshfts.append(curq)

        d, dshft = dict(), dict()
        if model_name == "dkt_forget":
            for key in curdforget:
                d[key] = curdforget[key][:,start:]
                dds.setdefault(key, [])
                dds[key].append(d[key])
            for key in dforget:
                curd = torch.tensor([[dforget[key][k]]]).long().to(device)
                dshft[key] = torch.cat((d[key][:,1:], curd), axis=1)
                ddshfts.setdefault(key, [])
                ddshfts[key].append(dshft[key])
        
    finalcs, finalrs = torch.cat(dcs, axis=0), torch.cat(drs, axis=0)
    finalqs, finalqshfts = torch.tensor([]), torch.tensor([])
    if cq.shape[0] > 0:
        finalqs = torch.cat(dqs, axis=0)
        finalqshfts = torch.cat(dqshfts, axis=0)
    finalcshfts, finalrshfts = torch.cat(dcshfts, axis=0), torch.cat(drshfts, axis=0)
    finald, finaldshft = dict(), dict()
    for key in dds:
        finald[key] = torch.cat(dds[key], axis=0)
        finaldshft[key] = torch.cat(ddshfts[key], axis=0)
    # print(f"qidx: {len(qidxs)}, finalqs: {finalqs.shape}, finalcs: {finalcs.shape}, finalrs: {finalrs.shape}")
    # print(f"qidx: {len(qidxs)}, finalqshfts: {finalqshfts.shape}, finalcshfts: {finalcshfts.shape}, finalrshfts: {finalrshfts.shape}")
    return qidxs, finalqs, finalcs, finalrs, finalqshfts, finalcshfts, finalrshfts, finald, finaldshft     

def predict_each_group2(curdforget, dforget, is_repeat, qidx, uid, idx, curqin, curcin, currin, model_name, model, t, cq, cc, cr, end, fout, atkt_pad=False, maxlen=200):
    """not use the predict result
    """
    nextcin, nextrin = curcin, currin
    import copy
    nextdforget = copy.deepcopy(curdforget)
    ctrues, cpreds = [], []
    # 以下这些用的是同一个历史,可以并行
    # 不用预测结果
    qidxs, finalqs, finalcs, finalrs, finalqshfts, finalcshfts, finalrshfts, finald, finaldshft = prepare_data(model_name, is_repeat, qidx, curqin, curcin, currin, curdforget, cq, cc, cr, dforget, t, end)
    bidx, bz = 0, 128
    while bidx < finalcs.shape[0]:
        curc, curr = finalcs[bidx: bidx+bz], finalrs[bidx: bidx+bz]
        curcshft, currshft = finalcshfts[bidx: bidx+bz], finalrshfts[bidx: bidx+bz]
        curqidxs = qidxs[bidx: bidx+bz]
        curq, curqshft = torch.tensor([[]]), torch.tensor([[]])
        if finalqs.shape[0] > 0:
            curq = finalqs[bidx: bidx+bz]
            curqshft = finalqshfts[bidx: bidx+bz]
        curd, curdshft = dict(), dict()
        if model_name == "dkt_forget":
            for key in finald:
                curd[key] = finald[key][bidx: bidx+bz]
                curdshft[key] = finaldshft[key][bidx: bidx+bz]
        ## start predict
        ccq = torch.cat((curq[:,0:1], curqshft), dim=1)
        ccc = torch.cat((curc[:,0:1], curcshft), dim=1)
        ccr = torch.cat((curr[:,0:1], currshft), dim=1)
        if model_name in ["dkt", "dkt+"]:
            y = model(curc.long(), curr.long())
            y = (y * one_hot(curcshft.long(), model.num_c)).sum(-1)
        elif model_name in ["dkt_forget"]:
            y = model(curc.long(), curr.long(), curd, curdshft)
            y = (y * one_hot(curcshft.long(), model.num_c)).sum(-1)
        elif model_name in ["dkvmn"]:
            y = model(ccc.long(), ccr.long())
            y = y[:,1:]
        elif model_name in ["kqn", "sakt"]:
            y = model(curc.long(), curr.long(), curcshft.long())
        elif model_name == "saint":
            y = model(ccq.long(), ccc.long(), curr.long())
            y = y[:, 1:]
        elif model_name == "akt":                                
            y, reg_loss = model(ccc.long(), ccr.long(), ccq.long())
            y = y[:,1:]
        elif model_name in ["atkt", "atktfix"]:
            # print(f"atkt_pad: {atkt_pad}")
            if atkt_pad == True:
                oricurclen = curc.shape[1]
                padlen = maxlen-1-oricurclen
                # print(f"padlen: {padlen}, curc: {curc.shape}")
                pad = torch.tensor([0]*padlen).unsqueeze(0).expand(curc.shape[0], padlen).to(device)
                curc = torch.cat((curc, pad), axis=1)
                curr = torch.cat((curr, pad), axis=1)
                curcshft = torch.cat((curcshft, pad), axis=1)
            y, _ = model(curc.long(), curr.long())
            y = (y * one_hot(curcshft.long(), model.num_c)).sum(-1)
        elif model_name == "gkt":
            y = model(ccc.long(), ccr.long())
            # print(f"y: {y}")
            # y = y[:, t-1:t]
        if model_name in ["atkt", "atktfix"] and atkt_pad == True:
            # print(f"use idx: {oricurclen-1}")
            pred = y[:, oricurclen-1].tolist()
            # assert ccr[:, t] == curcshft[:, t-1]
            true = currshft[:, oricurclen-1].tolist()
            # print(true)
            # true = curcshft[:, t-1].tolist()
        else:
            pred = y[:, -1].tolist()
            true = ccr[:, -1].tolist()
        
        # print(f"pred: {len(pred)}, true: {true}")

        # save pred res
        ctrues.extend(true)
        cpreds.extend(pred)

        # output
        
        for i in range(0, curc.shape[0]):
            clist, rlist = curc[i].long().tolist()[0:t], curr[i].long().tolist()[0:t]
            cshftlist, rshftlist = curcshft[i].long().tolist()[0:t], currshft[i].long().tolist()[0:t]
            qidx = curqidxs[i]
            predl = 1 if pred[i] >= 0.5 else 0
            # print("\t".join([str(idx), str(uid), str(bidx+i), str(qidx), str(len(clist)), str(clist), str(rlist), str(cshftlist), str(rshftlist), str(true[i]), str(pred[i]), str(predl)]))
            fout.write("\t".join([str(idx), str(uid), str(bidx+i), str(qidx), str(len(clist)), str(clist), str(rlist), str(cshftlist), str(rshftlist), str(true[i]), str(pred[i]), str(predl)]) + "\n")

        bidx += bz
    return qidxs, ctrues, cpreds

def save_currow_question_res(idx, dcres, dqres, qidxs, ctrues, cpreds, uid, fout):
    # save res
    dqidx = dict()
    # dhigh, dlow = dict(), dict()
    for i in range(0, len(qidxs)):
        true, pred = ctrues[i], cpreds[i]
        qidx = qidxs[i]
        dqidx.setdefault(qidx, {"trues": [], "preds": []})
        dqidx[qidx]["trues"].append(true)
        dqidx[qidx]["preds"].append(pred)

    for qidx in dqidx:
        ctrues, cpreds = dqidx[qidx]["trues"], dqidx[qidx]["preds"]
        late_mean, late_vote, late_all = save_each_question_res(dcres, dqres, ctrues, cpreds)
        # print("\t".join([str(idx), str(uid), str(qidx), str(late_mean), str(late_vote), str(late_all)]))
        fout.write("\t".join([str(idx), str(uid), str(qidx), str(late_mean), str(late_vote), str(late_all)]) + "\n")
