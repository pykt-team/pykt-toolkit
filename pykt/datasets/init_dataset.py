import os, sys
import json

from torch.utils.data import DataLoader
import numpy as np
from .data_loader import KTDataset
from .dkt_forget_dataloader import DktForgetDataset
from .lpkt_dataloader import LPKTDataset
from .lpkt_utils import generate_time2idx

def init_test_datasets(data_config, model_name, batch_size):
    test_question_loader, test_question_window_loader = None, None
    if model_name in ["dkt_forget"]:
        test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                        data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)
    if model_name in ["lpkt"]:
        test_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), at2idx, it2idx, data_config["input_type"], {-1})
        test_window_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), at2idx, it2idx, data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), at2idx, it2idx, data_config["input_type"], {-1}, True)
            test_question_window_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), at2idx, it2idx, data_config["input_type"], {-1}, True)
    else:

        test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    if "test_question_file" in data_config:
        print(f"has test_question_file!")
        test_question_loader = DataLoader(test_question_dataset, batch_size=batch_size, shuffle=False)
        test_question_window_loader = DataLoader(test_question_window_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_window_loader, test_question_loader, test_question_window_loader

def update_gap(max_rgap, max_sgap, max_pcount, cur):
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_sgap = cur.max_sgap if cur.max_sgap > max_sgap else max_sgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    return max_rgap, max_sgap, max_pcount

def init_dataset4train(dataset_name, model_name, data_config, i, batch_size):
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"])
    if model_name == "dkt_forget":
        max_rgap, max_sgap, max_pcount = 0, 0, 0
        curvalid = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curtrain)
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curvalid)
    elif model_name == "lpkt":
        at2idx, it2idx = generate_time2idx(data_config)
        json_str = json.dumps(at2idx)
        # with open('at2idx.json', 'w') as json_file:
        #     json_file.write(json_str)
        # json_str_2 = json.dumps(it2idx)
        # with open('it2idx.json', 'w') as json_file2:
        #     json_file2.write(json_str_2)
        curvalid = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), at2idx, it2idx, data_config["input_type"], {i})
        curtrain = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), at2idx, it2idx, data_config["input_type"], all_folds - {i})
    else:
        curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})
    train_loader = DataLoader(curtrain, batch_size=batch_size)
    valid_loader = DataLoader(curvalid, batch_size=batch_size)
    
    if model_name == "dkt_forget":
        test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        # test_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
        #                                 data_config["input_type"], {-1})
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, test_dataset)
    elif model_name == "lpkt":
        test_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), at2idx, it2idx, data_config["input_type"], {-1})
        # test_window_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), at2idx, it2idx, data_config["input_type"], {-1})
    else:
        test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        # test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
    
    if model_name == "dkt_forget":
        data_config["num_rgap"] = max_rgap + 1
        data_config["num_sgap"] = max_sgap + 1
        data_config["num_pcount"] = max_pcount + 1
    if model_name == "lpkt":
        print(f"num_at:{len(at2idx)}")
        print(f"num_it:{len(it2idx)}")
        data_config["num_at"] = len(at2idx) + 1
        data_config["num_it"] = len(it2idx) + 1
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = None
    return train_loader, valid_loader, test_loader, test_window_loader
