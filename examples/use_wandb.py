from pykt.utils.wandb_utils import WandbUtils
import pandas as pd
import os
from tqdm import tqdm_notebook
import yaml
import pandas as pd

import os
# wandb_key = "07ec58f8e8315aa083a202292092d7e2ee9d43b4"
wandb_key = "6c89ba6dac93ead57358a63bf337f8876f54bc0f"
os.environ['WANDB_API_KEY'] = wandb_key

import sys
dataset_name=sys.argv[1]
model_name=sys.argv[2]
emb_type=sys.argv[3]
wandb_api = WandbUtils(user='pykt-team', project_name=f'sparse_attn_shyann')

df = wandb_api.get_best_run(dataset_name,model_name,emb_type)
# 生成config
wandb_api.extract_best_models(df, dataset_name, model_name,emb_type,
                              fpath="../examples/seedwandb/predict.yaml",wandb_key=wandb_key)
