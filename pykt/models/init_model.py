import torch
import numpy as np
import os

from .dkt import DKT
from .dkt_plus import DKTPlus
from .dkvmn import DKVMN
from .deep_irt import DeepIRT
from .sakt import SAKT
from .saint import SAINT
from .kqn import KQN
from .atkt import ATKT
from .dkt_forget import DKTForget
from .akt import AKT
from .gkt import GKT
from .gkt_utils import get_gkt_graph
from .lpkt import LPKT
from .lpkt_utils import generate_qmatrix
from .skvmn import SKVMN
from .hawkes import HawkesKT
from .iekt import IEKT
from .cdkt import CDKT
from .bakt import BAKT
from .bakt_time import BAKTTime
from .bakt_simplex import BAKTSimpleX
from .qdkt import QDKT
from .qikt import QIKT
from .akt_peiyou import AKTPeiyou
from .iekt_peiyou import IEKTPeiyou
from .qdkt_peiyou import QDKTPeiyou
from .bakt_peiyou import BAKTPeiyou 

device = "cpu" if not torch.cuda.is_available() else "cuda"

import pandas as pd
def save_qcemb(model, emb_save, ckpt_path):
    fs = []
    for k in ckpt_path.split("/"):
        if k.strip() != "":
            fs.append(k)
    fname = "_".join(fs)
    for n, p in model.question_emb.named_parameters():
        pd.to_pickle(p, os.path.join(emb_save, fname+"qemb_from_cdkt.pkl"))
    for n, p in model.concept_emb.named_parameters():
        pd.to_pickle(p, os.path.join(emb_save, fname+"cemb_from_cdkt.pkl"))
    for n, p in model.interaction_emb.named_parameters():
        pd.to_pickle(p, os.path.join(emb_save, fname+"xemb_from_cdkt.pkl"))

def init_model(model_name, model_config, data_config, emb_type):
    print(f"in init_model, model_name: {model_name}")
    if model_name == "cdkt":
        model = CDKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "cakt":
        model = CAKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "bakt":
        model = BAKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "bakt_simplex":
        model = BAKTSimpleX(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt":
        model = DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "deep_irt":
        model = DeepIRT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt":
        model = SAKT(data_config["num_c"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "saint":
        model = SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_forget":
        model = DKTForget(data_config["num_c"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config).to(device)
    elif model_name == "akt":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "kqn":
        model = KQN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "atkt":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=False).to(device)
    elif model_name == "atktfix":
        model = ATKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], fix=True).to(device)
    elif model_name == "gkt":
        graph_type = model_config['graph_type']
        fname = f"gkt_graph_{graph_type}.npz"
        graph_path = os.path.join(data_config["dpath"], fname)
        if os.path.exists(graph_path):
            graph = torch.tensor(np.load(graph_path, allow_pickle=True)['matrix']).float()
        else:
            graph = get_gkt_graph(data_config["num_c"], data_config["dpath"], 
                    data_config["train_valid_original_file"], data_config["test_original_file"], graph_type=graph_type, tofile=fname)
            graph = torch.tensor(graph).float()
        model = GKT(data_config["num_c"], **model_config,graph=graph,emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "lpkt":
        qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
        if os.path.exists(qmatrix_path):
            q_matrix = np.load(qmatrix_path, allow_pickle=True)['matrix']
        else:
            q_matrix = generate_qmatrix(data_config)
        q_matrix = torch.tensor(q_matrix).float().to(device)
        model = LPKT(data_config["num_at"], data_config["num_it"], data_config["num_q"], data_config["num_c"], **model_config, q_matrix=q_matrix, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "skvmn":
        model = SKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)   
    elif model_name == "hawkes":
        if data_config["num_q"] == 0 or data_config["num_c"] == 0:
            print(f"model: {model_name} needs questions ans concepts! but the dataset has no both")
            return None
        model = HawkesKT(data_config["num_c"], data_config["num_q"], **model_config)
        model = model.double()
        # print("===before init weights"+"@"*100)
        # model.printparams()
        model.apply(model.init_weights)
        # print("===after init weights")
        # model.printparams()
        model = model.to(device)
    elif model_name == "iekt":
        model = IEKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)   
    elif model_name == "iekt_peiyou":
        dpretrain = {
            "kc_embs": [os.path.join(data_config["dpath"], data_config["kc_embs"]), 768],
            "ana_embs": [os.path.join(data_config["dpath"], data_config["ana_embs"]), 768],
            "que_embs": [os.path.join(data_config["dpath"], data_config["que_embs"]), 768]
        }
        model = IEKTPeiyou(num_q=data_config['num_q'], num_croutes=data_config["num_croutes"], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=dpretrain, device=device).to(device)   
    elif model_name == "qdkt":
        model = QDKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "qdkt_peiyou":
        dpretrain = {
            "kc_embs": [os.path.join(data_config["dpath"], data_config["kc_embs"]), 768],
            "ana_embs": [os.path.join(data_config["dpath"], data_config["ana_embs"]), 768],
            "que_embs": [os.path.join(data_config["dpath"], data_config["que_embs"]), 768]
        }
        model = QDKTPeiyou(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=dpretrain,device=device).to(device)
    elif model_name == "qikt":
        model = QIKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "cdkt":
        model = CDKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "bakt_time":
        model = BAKTTime(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    # elif model_name == "bakt":
    #     model = BAKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    # elif model_name == "bakt_simplex":
    #     model = BAKTSimpleX(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "bakt_peiyou":
        dpretrain = {
            "kc_embs": [os.path.join(data_config["dpath"], data_config["kc_embs"]), 768],
            "ana_embs": [os.path.join(data_config["dpath"], data_config["ana_embs"]), 768],
            "que_embs": [os.path.join(data_config["dpath"], data_config["que_embs"]), 768]
        }
        model = BAKTPeiyou(data_config["num_croutes"], data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, dpretrain=dpretrain).to(device)
    elif model_name == "akt_peiyou":
        dpretrain = {
            "kc_embs": [os.path.join(data_config["dpath"], data_config["kc_embs"]), 768],
            "ana_embs": [os.path.join(data_config["dpath"], data_config["ana_embs"]), 768],
            "que_embs": [os.path.join(data_config["dpath"], data_config["que_embs"]), 768]
        }
        model = AKTPeiyou(data_config["num_croutes"], data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, dpretrain=dpretrain).to(device)
    else:
        print(f"The wrong model name: {model_name} was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    # infs = model_name.split("_")
    # save = False
    # if len(infs) == 2:
    #     model_name, save = infs[0], True
    print(f"in load model! model name: {model_name}")#, save: {save}")
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    model.load_state_dict(net)
    if model_name == "cdkt" and save:
        save_qcemb(model, data_config["emb_save"], ckpt_path)
    '''
    from torchstat import stat
    stat(model)#, (3, 500, 500))
    '''
    return model
