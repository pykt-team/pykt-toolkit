# CGMKT 预处理管线 notebook 与各数据集预处理产物(kc_embeddings / ques_skill_gcn_adj /
# question_concept_map / sbm 图)托管在 Google Drive:
# https://drive.google.com/drive/folders/1OqTSZKni1sqF6_dq3s0oxn_wao8G4cIZ?usp=sharing
# 用法:下载 <dataset_name>/ 文件夹,内容合并到 data/<dataset_name>/ 后即可训练/预测
import argparse

import torch

from wandb_train import main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="cgmkt")
    parser.add_argument("--emb_type", type=str, default="qid_as09")
    parser.add_argument("--save_dir", type=str, default="saved_model/cgmkt")
    parser.add_argument("--fold", type=int, default=0)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dropout_qk", type=float, default=0.05)
    parser.add_argument("--dropout_kk", type=float, default=0.4)
    parser.add_argument("--num_clusters", type=int, default=4)
    parser.add_argument("--num_gcn_layers", type=int, default=1)
    parser.add_argument("--mastery_update_hidden", type=int, default=64)
    parser.add_argument("--mastery_step", type=float, default=0.1)
    parser.add_argument("--modulation_type", type=str, default="gate")
    parser.add_argument("--spread_type", type=str, default="sbm")
    parser.add_argument("--spread_rate_init", type=float, default=0.1)
    parser.add_argument("--spread_rate_max", type=float, default=1.0)
    parser.add_argument("--mastery_bound", type=float, default=5.0)

    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)

    args = parser.parse_args()

    main(vars(args))
