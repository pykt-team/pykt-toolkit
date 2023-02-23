from pykt.utils.wandb_utils import WandbUtils
import pandas as pd
import os
from tqdm import tqdm_notebook
import yaml
import pandas as pd

import os
wandb_key = "6c89ba6dac93ead57358a63bf337f8876f54bc0f"
os.environ['WANDB_API_KEY'] = wandb_key

import sys
dname=sys.argv[1]
model_name=sys.argv[2]
emb_type=sys.argv[3]
project_name=sys.argv[4]
#wandb_api = WandbUtils(user='kitty3', project_name='bakt_newseed')#f'nips2022-{dataset_name}')
wandb_api = WandbUtils(user='pykt-team', project_name=project_name)#f'nips2022-{dataset_name}')
for dataset_name in dname.split(","):
    print(f"dataset_name: {dataset_name}, model_name: {model_name}, emb_type: {emb_type}")
    typ = int(sys.argv[5])
    if typ == 0:
        '''
        fout = open("xrpre.sh", "w")
        yamlname = os.path.join("pred_wandbs", "_".join([dataset_name, model_name, emb_type]) + "_fold_first_predict.yaml")
        fyaml = open(yamlname, "w")
        i = 0
        '''
        logf = "log.xr"+"_"+dataset_name
        fout = open(logf, "w")
        for etype in emb_type.split(","):
            print(f"dataset_name: {dataset_name}, model_name: {model_name}, emb_type: {etype}")
            df = wandb_api.get_best_run(dataset_name,model_name,emb_type=etype)
            # 生成config
            wandb_api.extract_best_models(df, dataset_name, model_name,
                                          fpath="../examples/seedwandb/predict.yaml",wandb_key=wandb_key,emb_type=etype)
            os.system("sh start_predict.sh >> " + logf + " 2>&1")
        '''
            with open(os.path.join("pred_wandbs", "_".join([dataset_name, model_name, etype]) + "_fold_first_predict.yaml")) as fin:
                for line in fin.readlines():
                    if i == 0 and line.find("best_model_path") == -1 and line.find("program") == -1 and line.find("name") == -1:
                        fyaml.write(line)
                    if i ==0 and line.find("name") != -1:
                        fyaml.write(line.replace(etype, "xiaorong"))
                    if line.find("best_model_path") != -1:
                        fyaml.write(line)
            i += 1
        fyaml.write('program: "./wandb_predict.py"')
        fout.write("WANDB_API_KEY=dc95dfebd33a883d3bdd540b4396189aad828f59 wandb sweep " + yamlname + " -p " + project_name)
        fout.close()
        fyaml.close()
        '''
    else:
        for etype in emb_type.split(","):
            print(f"dataset_name: {dataset_name}, model_name: {model_name}, emb_type: {etype}")
            wandb_api.extract_prediction_results(dataset_name, model_name, print_std=True,emb_type=etype)
            print("="*20)

