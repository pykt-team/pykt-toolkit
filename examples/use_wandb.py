from pykt.utils.wandb_utils import WandbUtils
import pandas as pd
import os
from tqdm import tqdm_notebook
import yaml
import pandas as pd

import os
wandb_key = "07ec58f8e8315aa083a202292092d7e2ee9d43b4"
#wandb_key = "6c89ba6dac93ead57358a63bf337f8876f54bc0f"
os.environ['WANDB_API_KEY'] = wandb_key

import sys
dataset_names=sys.argv[1]
model_names=sys.argv[2]
emb_types=sys.argv[3]
extract_type=sys.argv[4]

wandb_api = WandbUtils(user='shyann', project_name=f'gpt4kt')
#wandb_api = WandbUtils(user='pykt-team', project_name=f'rerun_lpkt')

# 生成config
if extract_type == str(1):
    dataset_names = dataset_names.split(",")
    model_names = model_names.split(",")
    emb_types = emb_types.split(",")
    for dataset_name in dataset_names:
        for model_name in model_names:
            for emb_type in emb_types:
                df = wandb_api.get_best_run(dataset_name,model_name,emb_type)
                wandb_api.extract_best_models(df, dataset_name, model_name,emb_type, fpath="../examples/seedwandb/predict.yaml",wandb_key=wandb_key)

else:
    dataset_names = dataset_names.split(",")
    model_names = model_names.split(",")
    emb_types = emb_types.split(",")
# dataset_names = ["statics2011","algebra2005", "assist2009","nips_task34"]
    for dataset_name in dataset_names:
        for model_name in model_names:
            for emb_type in emb_types:
                print(wandb_api.extract_prediction_results(dataset_name, model_name, emb_type=emb_type,print_std=True))
                
