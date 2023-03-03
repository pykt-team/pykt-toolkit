import os
import argparse
import json

import torch
torch.set_num_threads(4) 
from torch.optim import SGD, Adam
import copy

from pykt.models import train_model,evaluate,init_model
from pykt.utils import debug_print,set_seed
from pykt.datasets import init_dataset4train
import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)

def addF2AKT(model, train_loader, valid_loader, test_loader):
    def prepare(data, cs, rs, sms):
        dcur = data
        q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
        m, sm = dcur["masks"], dcur["smasks"]
        curcs = torch.cat([c[:, 0:1], cshft], dim=1)
        currs = torch.cat([r[:, 0:1], rshft], dim=1)
        cs = torch.cat([cs, curcs], dim=0)
        rs = torch.cat([rs, currs], dim=0)
        sms = torch.cat([sms, sm], dim=0)
        return cs, rs, sms
    cs, rs, sms = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
    
    for data in train_loader:
        cs, rs, sms = prepare(data, cs, rs, sms)
    
    # cs, rs, sms = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
    # model.calSkillF(cs.long(), rs.long(), sms.long(), istrain=True)
    # for data in valid_loader:
    #     cs, rs, sms = prepare(data, cs, rs, sms)
    # for data in test_loader:
    #     cs, rs, sms = prepare(data, cs, rs, sms)
    model.calSkillF(cs.long(), rs.long(), sms.long())

    print(f"cs: {cs.shape}, rs: {rs.shape}")
    print(f"dF: {len(model.dF)}")

def main(params):
    if "use_wandb" not in params:
        params['use_wandb'] = 1

    if params['use_wandb']==1:
        import wandb
        wandb.init()

    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]
        
    debug_print(text = "load config files.",fuc_name="main")
    
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        if model_name in ["dkvmn", "deep_irt", "skvmn", "sakt", "csakt", "saint","saint++", "akt", "cakt", "atkt", "lpkt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["bakt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["bakt", "bakt_peiyou", "bakt_time", "bakt_simplex"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16 
        if model_name in ["qdkt","qikt"] and dataset_name in ['algebra2005','bridge2algebra2006']:
            train_config["batch_size"] = 32 
        if model_name in ["akt_peiyou", "iekt_peiyou"]:
            train_config["batch_size"] = 64
        model_config = copy.deepcopy(params)
        for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed",
                "insert_ratio", "crop_ratio", "mask_ratio", "reorder_ratio", "aug", "K"]:
            if key not in model_config:
                continue
            del model_config[key]
        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']
        # model_config = {"d_model": params["d_model"], "n_blocks": params["n_blocks"], "dropout": params["dropout"], "d_ff": params["d_ff"]}
    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
        train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]

    print("Start init data")
    print(dataset_name, model_name, data_config, fold, batch_size)

    aug = False
    if "aug" in params:
        aug = params["aug"]
    print(f"aug: {aug}")
    if aug:
        data_config[dataset_name]["insert"] = params["insert_ratio"]
        data_config[dataset_name]["crop"] = params["crop_ratio"]
        data_config[dataset_name]["mask"] = params["mask_ratio"]
        data_config[dataset_name]["reorder"] = params["reorder_ratio"]
        data_config[dataset_name]["K"] = params["K"]
        data_config[dataset_name]["aug"] = True
    
    debug_print(text="init_dataset",fuc_name="main")
    train_loader, valid_loader = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size)

    params_str = "_".join([str(v) for k,v in params.items() if not k in ['other_config']])

    print(f"params: {params}, params_str: {params_str}")
    if params['add_uuid'] == 1 and params["use_wandb"] == 1:
        import uuid
        # if not model_name in ['saint','saint++']:
        params_str = params_str+f"_{ str(uuid.uuid4())}"
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"train_config: {train_config}")

    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
    learning_rate = params["learning_rate"]
    for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
        if remove_item in model_config:
            del model_config[remove_item]
    if model_name in ["saint","saint++", "sakt", "cdkt", "bakt", "bakt_time", "bakt_simplex"]:
        model_config["seq_len"] = seq_len
        
    debug_print(text = "init_model",fuc_name="main")
    print(f"model_name:{model_name}")
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    print(f"model is {model}")
    if model_name == "hawkes":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
    elif model_name in ["iekt", "iekt_peiyou"]:
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    else:
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)
   
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True
    
    debug_print(text = "train model",fuc_name="main")
    
    testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model)
    
    if save_model:
        best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
        net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
        best_model.load_state_dict(net)

    print("fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch")
    print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(testauc, 4)) + "\t" + str(round(testacc, 4)) + "\t" + str(round(window_testauc, 4)) + "\t" + str(round(window_testacc, 4)) + "\t" + str(validauc) + "\t" + str(validacc) + "\t" + str(best_epoch))
    model_save_path = os.path.join(ckpt_path, emb_type+"_model.ckpt")
    print(f"end:{datetime.datetime.now()}")
    
    if params['use_wandb']==1:
        wandb.log({ 
                    "validauc": validauc, "validacc": validacc, "best_epoch": best_epoch,"model_save_path":model_save_path})
