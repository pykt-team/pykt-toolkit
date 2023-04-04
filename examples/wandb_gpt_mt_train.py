import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykt.models.gpt_mt import GPTMT as GPT
from pykt.datasets.que_data_loader import KTQueDataset
from torch.utils.data import DataLoader
import json
import pandas as pd
import numpy as np
from sklearn import metrics
from pykt.utils import set_seed
from torch.utils.data import DataLoader
import uuid
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(params):
    if params['use_wandb']==1:
        import wandb
        wandb.init()
    
    set_seed(params["seed"])#set seed
    params['share_output'] = params['share_output']==1
    params['aux_tasks'] = params['aux_tasks'].split(',')

    params['source_list'] = params['source_list'].split(',')
    source_list = params['source_list']
    data_config = json.load(open('../configs/data_config.json'))
    all_folds = set(data_config['assist2009']["folds"])
    

    config_list = [data_config[source] for source in source_list]
    max_concepts = max([x['max_concepts'] for x in config_list])
    max_c_num = max([x['num_c'] for x in config_list])
    max_q_num = max([x['num_q'] for x in config_list])


    train_dataset_list = []
    valid_dataset_list = []

    i = params['fold']
    train_file_key = "train_valid_file_quelevel" 
    for source, data_config in zip(source_list, config_list):        
        train_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config[train_file_key]),
                                    input_type=data_config["input_type"], folds=all_folds - {
                                        i},
                                    concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])

        valid_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config[train_file_key]),
                                    input_type=data_config["input_type"], folds={
                                        i},
                                    concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
        train_dataset_list.append(train_dataset)
        valid_dataset_list.append(valid_dataset)

    save_dir = params['save_dir']
    if params['add_uuid']==1:
        save_dir = os.path.join(save_dir, str(uuid.uuid4()))
    
    os.makedirs(save_dir, exist_ok=True)

    # save params
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(params, f)
    print(f"params are {params}")

    model_args = copy.deepcopy(params)
    for key in ['batch_size','num_epochs','patient','use_wandb','add_uuid','fold','save_dir','learning_rate']:
        del model_args[key]
    # define model
    model = GPT(num_q=max_q_num, 
                num_c=max_c_num, 
                device=device, 
                return_dict=True,
                dataconfig_list=config_list,
                **model_args
                )

    model = model.to(device)
    model.compile(optimizer='adam', lr=params['learning_rate'])

    # train model
    model_result = model.train(train_dataset_list, valid_dataset_list,source_list,
                batch_size=params['batch_size'],
                num_epochs=params['num_epochs'], 
                patient=params['patient'], 
                shuffle=False,save_model=True,save_dir=save_dir)

    model_result['model_save_dir'] = save_dir

    if params['use_wandb']==1:
        wandb.log(model_result)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_list", type=str, default="assist2009,nips_task34,bridge2algebra2006,algebra2005")
    parser.add_argument("--model_name", type=str, default="gpt_mt")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patient", type=int, default=10)
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    #input to model
    parser.add_argument("--source_weight_t", type=float, default=1)
    parser.add_argument("--emb_type", type=str, default="iekt")
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--share_output", type=int, default=1)
    parser.add_argument("--aux_tasks", type=str, default='')
    parser.add_argument("--mlp_layer_num", type=int, default=1)
    

    
   
   
    args = parser.parse_args()

    params = vars(args)
    main(params)


# python wandb_gpt_mt_train.py --dropout=0.05 --emb_size=64 --emb_type=qid --fold=0 --learning_rate=0.001 --model_name=gpt_mt --n_head=8 --n_layer=12 --save_dir=models/gpt_mt --seed=42 --share_output=1 --source_list=assist2009,nips_task34,bridge2algebra2006,algebra2005 --source_weight_t=1