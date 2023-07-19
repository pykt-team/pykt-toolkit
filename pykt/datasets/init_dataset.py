import os, sys
import json

from torch.utils.data import DataLoader
import numpy as np
from .data_loader import KTDataset
from .dkt_forget_dataloader import DktForgetDataset
from .parkt_dataloader import ParktDataset
# from .cdkt_dataloader import CDKTDataset
from .lpkt_dataloader import LPKTDataset
from .lpkt_utils import generate_time2idx
from .que_data_loader import KTQueDataset
from .que_data_loader_cl import KTQueDataset4CL
from .que_data_loader_time import KTQueDataset4PT
from pykt.config import que_type_models
# from .simplekt_cl_dataloader import CL4KTDataset
from .cl_utils import sort_samples
from .cl_dataloader import CL4KTDataset

def init_test_datasets(data_config, model_name, batch_size,i,win200=""):
    print(f"model_name is {model_name}")
    test_question_loader, test_question_window_loader = None, None
    if model_name in ["dkt_forget", "bakt_time"]:
        test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                        data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)
    elif model_name in ["lpkt"]:
        print(f"model_name in lpkt")
        at2idx, it2idx = generate_time2idx(data_config)
        test_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]), at2idx, it2idx, data_config["input_type"], {-1})
        test_window_dataset = LPKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]), at2idx, it2idx, data_config["input_type"], {-1})
        test_question_dataset = None
        test_question_window_dataset= None
    elif model_name in ["parkt", "mikt"]:
        # print(f"model_name in parkt")
        test_dataset = ParktDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = ParktDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                        data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = ParktDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = ParktDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)  
    elif model_name in que_type_models:
        print(f"model_name:{model_name}")
        if model_name not in ["gpt4kt"]:
            test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                            input_type=data_config["input_type"], folds=[-1], 
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
            test_window_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                            input_type=data_config["input_type"], folds=[-1], 
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
        else:
            # test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel_pretrain"]),
            #                 input_type=data_config["input_type"], folds=[-1], 
            #                 concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
            test_dataset = None
            dataset = data_config['dpath'].split("/")[-1]
            if win200:
                if dataset in ["assist2009", "algebra2005", "bridge2algebra2006", "nips_task34", "ednet", "peiyou", "ednet5w"]:
                    test_window_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel_pretrain_w200"]),
                                input_type=data_config["input_type"], folds=[-1], 
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
                else:
                    test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
                    test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})                              
            else:
                if dataset in ["assist2009", "algebra2005", "bridge2algebra2006", "nips_task34", "ednet", "peiyou", "ednet5w"]:
                    test_window_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel_pretrain_w200"]),
                                input_type=data_config["input_type"], folds=[-1], 
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
                else:
                    test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
                    test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})  
        test_question_dataset = None
        test_question_window_dataset= None
    elif model_name in ["cdkt"]:
        test_dataset = CDKTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        test_window_dataset = CDKTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = CDKTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = CDKTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)
    else:
        test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        all_folds = set(data_config["folds"])
        print(f"test_window_file:{data_config['test_window_file']}")
        print(f"input_type:{data_config['input_type']}")
        test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, True)
            test_question_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    # if "test_question_file" in data_config:
    #     print(f"has test_question_file!")
    #     test_question_loader,test_question_window_loader = None,None
    #     if not test_question_dataset is None:
    #         test_question_loader = DataLoader(test_question_dataset, batch_size=batch_size, shuffle=False)
    #     if not test_question_window_dataset is None:
    #         test_question_window_loader = DataLoader(test_question_window_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_window_loader, test_question_loader, test_question_window_loader

def update_gap(max_rgap, max_sgap, max_pcount, max_it, cur):
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_sgap = cur.max_sgap if cur.max_sgap > max_sgap else max_sgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    max_it = cur.max_it if cur.max_it > max_it else max_it
    return max_rgap, max_sgap, max_pcount, max_it

def init_dataset4train(dataset_name, model_name, emb_type, data_config, i, batch_size, args=None):
    print(f"dataset_name:{dataset_name}")
    print(f"data_conf:{data_config}")
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"])
    if emb_type.find("cl") != -1:
        # train_valid_path = os.path.join(data_config["dpath"], data_config["train_valid_file"])
        # cl_dpath = sort_samples(train_valid_path, data_config["dpath"])
        # # print(f"cl_dpath:{cl_dpath}")
        # curvalid = KTDataset(cl_dpath, data_config["input_type"], {i})
        # # print(f"curvalid:{len(curvalid)}")
        # curtrain = KTDataset(cl_dpath, data_config["input_type"], all_folds - {i})  
        # print(f"curtrain:{len(curtrain)}")
        if model_name != "gpt4kt":
            train_valid_path = os.path.join(data_config["dpath"], data_config["train_valid_file"])
            sorted_df = sort_samples(train_valid_path)
            curvalid = CL4KTDataset(train_valid_path, sorted_df, data_config["input_type"], {i})
            # print(f"curvalid:{len(curvalid)}")
            curtrain = CL4KTDataset(train_valid_path, sorted_df, data_config["input_type"], all_folds - {i})  
            # print(f"curtrain:{len(curtrain)}")
        else:
            train_valid_path = os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"])
            sorted_df = sort_samples(train_valid_path)
            curvalid = KTQueDataset4CL(train_valid_path, sorted_df, data_config["input_type"], {i}, concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
            curtrain = KTQueDataset4CL(train_valid_path, sorted_df, data_config["input_type"], all_folds - {i}, concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])  
    elif model_name in ["dkt_forget", "bakt_time"]:
        max_rgap, max_sgap, max_pcount, max_it = 0, 0, 0, 0
        curvalid = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})
        max_rgap, max_sgap, max_pcount, max_it = update_gap(max_rgap, max_sgap, max_pcount, max_it, curtrain)
        max_rgap, max_sgap, max_pcount, max_it = update_gap(max_rgap, max_sgap, max_pcount, max_it, curvalid)
    elif model_name == "lpkt":
        at2idx, it2idx = generate_time2idx(data_config)
        # json_str = json.dumps(at2idx)
        # with open('at2idx.json', 'w') as json_file:
        #     json_file.write(json_str)
        # json_str_2 = json.dumps(it2idx)
        # with open('it2idx.json', 'w') as json_file2:
        #     json_file2.write(json_str_2)
        curvalid = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]), at2idx, it2idx, data_config["input_type"], {i})
        curtrain = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]), at2idx, it2idx, data_config["input_type"], all_folds - {i})
    elif model_name in ["rkt"] and dataset_name in ["statics2011", "assist2015", "poj"]:
        curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})
    elif model_name in que_type_models:
        if emb_type.find("pt") != -1:
            max_rgap, max_sgap, max_pcount, max_it = 0, 0, 0, 0
            curvalid = KTQueDataset4PT(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                            input_type=data_config["input_type"], folds={i}, 
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
            curtrain = KTQueDataset4PT(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                            input_type=data_config["input_type"], folds=all_folds - {i}, 
                            concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])  
            max_sgap = curtrain.max_sgap if curtrain.max_sgap > max_sgap else max_sgap
            max_sgap = curvalid.max_sgap if curvalid.max_sgap > max_sgap else max_sgap        
        else:
            if model_name == "gpt4kt":
                seq_len = args.seq_len
                curvalid = KTQueDataset(os.path.join(data_config["dpath"], f"train_valid_sequences_quelevel_{seq_len}.csv"),
                                input_type=data_config["input_type"], folds={i}, 
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
                curtrain = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds=all_folds - {i}, 
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
            else:        
                curvalid = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds={i}, 
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
                curtrain = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds=all_folds - {i}, 
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
    elif model_name in ["cdkt"]:
        curvalid = CDKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = CDKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})
    elif model_name in ["simplekt_sr"]:
        curvalid = CL4KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], data_config["num_c"], data_config["num_q"], {i}, args = args)
        curtrain = CL4KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], data_config["num_c"], data_config["num_q"], all_folds - {i}, args = args) 
    elif model_name in ["parkt", "mikt"]:
        if emb_type.find("cl") != -1 or emb_type.find("uid") != -1:
            curvalid = CL4KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], data_config["num_c"], data_config["num_q"], {i}, args = args)
            curtrain = CL4KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], data_config["num_c"], data_config["num_q"], all_folds - {i}, args = args) 
        elif emb_type.find("time")!= -1:
            # at2idx, it2idx = generate_time2idx(data_config)
            # curvalid = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), at2idx, it2idx, data_config["input_type"], {i})
            # curtrain = LPKTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), at2idx, it2idx, data_config["input_type"], all_folds - {i})
            max_rgap, max_sgap, max_pcount, max_it = 0, 0, 0, 0
            curvalid = ParktDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
            curtrain = ParktDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})
            max_rgap, max_sgap, max_pcount, max_it = update_gap(max_rgap, max_sgap, max_pcount, max_it, curtrain)
            max_rgap, max_sgap, max_pcount, max_it = update_gap(max_rgap, max_sgap, max_pcount, max_it, curvalid)
        else:
            curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
            curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})   
    else:
        curvalid = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], {i})
        curtrain = KTDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config["input_type"], all_folds - {i})
    if emb_type.find("cl") != -1:
        # train_loader = None
        train_loader = DataLoader(curtrain, batch_size=batch_size)
        valid_loader = DataLoader(curvalid, batch_size=batch_size)
    else:
        print(f"curvalid:{len(curvalid)}")
        print(f"curtrain:{len(curtrain)}")
        train_loader = DataLoader(curtrain, batch_size=batch_size)
        valid_loader = DataLoader(curvalid, batch_size=batch_size)
    
    # try:
    if model_name in ["dkt_forget", "bakt_time"]:
        test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        # test_window_dataset = DktForgetDataset(os.path.join(data_config[n"dpath"], data_config["test_window_file"]),
        #                                 data_config["input_type"], {-1})
        max_rgap, max_sgap, max_pcount, max_it = update_gap(max_rgap, max_sgap, max_pcount, max_it, test_dataset)
    elif model_name in ["parkt", "mikt"]:
        test_dataset = ParktDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config["input_type"], {-1})
        max_rgap, max_sgap, max_pcount, max_it = update_gap(max_rgap, max_sgap, max_pcount, max_it, test_dataset)
        if dataset_name in ["assist2009", "assist2015"]:
            test_window_dataset = ParktDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config["input_type"], {-1})
            max_rgap, max_sgap, max_pcount, max_it = update_gap(max_rgap, max_sgap, max_pcount, max_it, test_window_dataset) 
            if "test_question_file" in data_config:
                test_question_dataset = ParktDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config["input_type"], {-1}, qtest=True)
                max_rgap, max_sgap, max_pcount, max_it = update_gap(max_rgap, max_sgap, max_pcount, max_it, test_question_dataset) 
                test_question_window_dataset = ParktDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config["input_type"], {-1}, qtest=True)
                max_rgap, max_sgap, max_pcount, max_it = update_gap(max_rgap, max_sgap, max_pcount, max_it, test_question_window_dataset) 
        print(f"update_data:max_rgap:{max_rgap}, max_sgap:{max_sgap}, max_pcount:{max_pcount}, max_it:{max_it}")     
    # except:
    #     pass
    
    if model_name in ["dkt_forget", "bakt_time"] or emb_type.find("time") != -1:
        data_config["num_rgap"] = max_rgap + 1
        data_config["num_sgap"] = max_sgap + 1
        data_config["num_pcount"] = max_pcount + 1
        data_config["num_it"] = max_it + 1
    if model_name in ["lpkt"] :
        print(f"num_at:{len(at2idx)}")
        print(f"num_it:{len(it2idx)}")
        data_config["num_at"] = len(at2idx) + 1
        data_config["num_it"] = len(it2idx) + 1
    if model_name in ["gpt4kt"] and emb_type.find("pt") != -1:
        data_config["num_sgap"] = max_sgap + 1
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # # test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    # test_window_loader = None
    # print(f"curtrain:{len(curtrain)}")
    return train_loader, valid_loader, curtrain#, test_loader, test_window_loader
