import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykt.models.gpt_mt import GPTMT as GPT
from pykt.models.gpt_mt import GPTConfig
from pykt.datasets.que_data_loader import KTQueMultiDataset
from torch.utils.data import DataLoader
import json
import pandas as pd
import numpy as np
from sklearn import metrics
from pykt.utils import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_config = json.load(open('../configs/data_config.json'))



all_folds = set(data_config['assist2009']["folds"])
input_type = data_config['assist2009']["input_type"]


source_list = ['assist2009','nips_task34']

config_list = [data_config[source] for source in source_list]
file_path_list = [os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]) for data_config in  config_list]

max_concepts = max([x['max_concepts'] for x in config_list])
concept_num = max([x['num_c'] for x in config_list])

i = 0
train_dataset = KTQueMultiDataset(file_path_list=file_path_list,source_list=source_list, folds=all_folds - {i}, 
                        input_type=input_type,
                        concept_num=concept_num, max_concepts=max_concepts)

valid_dataset = KTQueMultiDataset(file_path_list=file_path_list,source_list=source_list,
                        input_type=input_type, folds={i}, 
                        concept_num=concept_num, max_concepts=max_concepts)

# test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
#                         input_type=data_config["input_type"], folds=[-1], 
#                         concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
# test_win_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
#                         input_type=data_config["input_type"], folds=[-1], 
#                         concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])

set_seed(3407)


model = GPT(num_q=-1, num_c=-1, emb_size=128, dropout=0.1,
            emb_type='qid', n_head=2, device=device, seed=0, n_layer=2,
            dataconfig_list=config_list,
            source_list=source_list
            )
model = model.to(device)

model.compile(optimizer='adam', lr=0.001)

model.train(train_dataset, valid_dataset, batch_size=64,
            num_epochs=300, patient=10, shuffle=False,save_model=True,save_dir='tmp/dkt_test')

# Epoch: 1,validauc: 0.7479, validacc: 0.7219, best epoch: 1, best auc: 0.7479, train loss: 0.6252192616645054, emb_type: iekt, model: dkt_que, save_dir: tmp/dkt_test

model.load_model(model.save_dir)