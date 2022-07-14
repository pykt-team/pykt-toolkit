import torch
import numpy as np
import os

from .dkt import DKT
from .dkt_plus import DKTPlus
from .dkvmn import DKVMN
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
from .dkt_rasch import DKTRasch
from .akt_vector import AKTVec

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_model(model_name, model_config, data_config, emb_type):
    if model_name == "dkt":
        model = DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt":
        model = SAKT(data_config["num_c"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "saint":
        model = SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_forget":
        model = DKTForget(data_config["num_c"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config).to(device)
    elif model_name == "akt":
        if emb_type.startswith("relation") or emb_type.startswith("yplus"):
            qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
            if os.path.exists(qmatrix_path):
                q_matrix = torch.tensor(np.load(qmatrix_path, allow_pickle=True)['matrix']).float()
            else:
                q_matrix = generate_qmatrix(data_config)
                q_matrix = torch.tensor(q_matrix).float()
            model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], qmatrix=q_matrix).to(device)
        else:
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
            q_matrix = torch.tensor(np.load(qmatrix_path, allow_pickle=True)['matrix']).float()
        else:
            q_matrix = generate_qmatrix(data_config)
            q_matrix = torch.tensor(q_matrix).float()
        model = LPKT(data_config["num_at"], data_config["num_it"], data_config["num_q"], data_config["num_c"], **model_config, q_matrix=q_matrix, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "skvmn":
        model = SKVMN(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_rasch":
        model = DKTRasch(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "dkt_interac":
        model = DKTRasch(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], use_interac=True).to(device)
    elif model_name == "akt_vector":
        if emb_type.startswith("relation") or emb_type.startswith("yplus"):
            qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
            if os.path.exists(qmatrix_path):
                q_matrix = torch.tensor(np.load(qmatrix_path, allow_pickle=True)['matrix']).float()
            else:
                q_matrix = generate_qmatrix(data_config)
                q_matrix = torch.tensor(q_matrix).float()
            model = AKTVec(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], qmatrix=q_matrix).to(device)
        else:
            model = AKTVec(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "aktvec_raschx":
        model = AKTVec(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], use_rasch=True, rasch_x=True).to(device)
    elif model_name == "akt_norasch":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], use_rasch=False).to(device)
    elif model_name == "akt_raschx":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], rasch_x=True).to(device)
    elif model_name == "akt_raschy":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], rasch_y=True).to(device)
    elif model_name == "akt_mono":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], monotonic=False).to(device)
    elif model_name == "aktmono_pos":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], monotonic=False, use_pos=True).to(device)
    elif model_name == "akt_attn":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], use_rasch=False, monotonic=False).to(device)
    elif model_name == "aktattn_pos":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], use_rasch=False, monotonic=False, use_pos=True).to(device)
    elif model_name == "dkt_qmatrix":
        qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
        if os.path.exists(qmatrix_path):
            q_matrix = torch.tensor(np.load(qmatrix_path, allow_pickle=True)['matrix']).float()
        else:
            q_matrix = generate_qmatrix(data_config)
            q_matrix = torch.tensor(q_matrix).float()
        model = DKTRasch(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], use_interac=False, use_qmatrix=True, qmatrix=q_matrix).to(device)
    elif model_name == "dkt_mastery":
        qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
        if os.path.exists(qmatrix_path):
            q_matrix = torch.tensor(np.load(qmatrix_path, allow_pickle=True)['matrix']).float()
        else:
            q_matrix = generate_qmatrix(data_config)
            q_matrix = torch.tensor(q_matrix).float()
        model = DKTRasch(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], use_interac=False, use_qmatrix=True, qmatrix=q_matrix, kt_state=True).to(device)
    else:
        print("The wrong model name was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    model.load_state_dict(net)
    return model
