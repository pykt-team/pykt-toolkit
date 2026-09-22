# CGMKT 预处理管线 notebook 与各数据集预处理产物(kc_embeddings / ques_skill_gcn_adj /
# question_concept_map / sbm 图)托管在 Google Drive:
# https://drive.google.com/drive/folders/1OqTSZKni1sqF6_dq3s0oxn_wao8G4cIZ?usp=sharing
# 用法:下载 <dataset_name>/ 文件夹,内容合并到 data/<dataset_name>/ 后即可训练/推理
import argparse

import torch

from wandb_train import main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# assist2009 五折调参最优超参(迁移自 pykt-moekt examples/models/cadkt_tiaocan_assist2009,
# 每折取验证集最优 run 的 config.json)。
# 五折公共取值:emb_size=256, num_gcn_layers=1, modulation_type="gate", spread_type="sbm",
# spread_rate_max=1.0, mastery_bound=5.0, batch_size=128, num_epochs=200
FOLD_BEST_PARAMS = {
    0: {"learning_rate": 1e-4, "seed": 3407, "dropout": 0.5, "dropout_qk": 0.1, "dropout_kk": 0.4,
        "num_clusters": 3, "mastery_update_hidden": 256, "mastery_step": 0.2, "spread_rate_init": 0.1},
    1: {"learning_rate": 5e-4, "seed": 3407, "dropout": 0.5, "dropout_qk": 0.05, "dropout_kk": 0.2,
        "num_clusters": 5, "mastery_update_hidden": 32, "mastery_step": 0.05, "spread_rate_init": 0.1},
    2: {"learning_rate": 1e-4, "seed": 42, "dropout": 0.4, "dropout_qk": 0.1, "dropout_kk": 0.4,
        "num_clusters": 4, "mastery_update_hidden": 32, "mastery_step": 0.03, "spread_rate_init": 0.1},
    3: {"learning_rate": 1e-4, "seed": 3407, "dropout": 0.4, "dropout_qk": 0.1, "dropout_kk": 0.4,
        "num_clusters": 7, "mastery_update_hidden": 128, "mastery_step": 0.2, "spread_rate_init": 0.05},
    4: {"learning_rate": 1e-4, "seed": 42, "dropout": 0.2, "dropout_qk": 0.05, "dropout_kk": 0.4,
        "num_clusters": 9, "mastery_update_hidden": 32, "mastery_step": 0.05, "spread_rate_init": 0.1},
}

# --use_fold_best 0(或 fold 不在表内)时的兜底默认值
DEFAULT_PARAMS = {
    "learning_rate": 1e-3, "seed": 3407, "dropout": 0.3, "dropout_qk": 0.05, "dropout_kk": 0.4,
    "num_clusters": 4, "mastery_update_hidden": 64, "mastery_step": 0.1, "spread_rate_init": 0.1,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="cgmkt")
    parser.add_argument("--emb_type", type=str, default="qid_as09")
    parser.add_argument("--save_dir", type=str, default="saved_model/cgmkt")
    parser.add_argument("--fold", type=int, default=0)

    # 下面这组超参默认 None:按 --fold 查 FOLD_BEST_PARAMS 补全(命令行显式传入的值优先);
    # 查不到时用 DEFAULT_PARAMS 兜底
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--dropout_qk", type=float, default=None)
    parser.add_argument("--dropout_kk", type=float, default=None)
    parser.add_argument("--num_clusters", type=int, default=None)
    parser.add_argument("--num_gcn_layers", type=int, default=1)
    parser.add_argument("--mastery_update_hidden", type=int, default=None)
    parser.add_argument("--mastery_step", type=float, default=None)
    parser.add_argument("--modulation_type", type=str, default="gate")
    parser.add_argument("--spread_type", type=str, default="sbm")
    parser.add_argument("--spread_rate_init", type=float, default=None)
    parser.add_argument("--spread_rate_max", type=float, default=1.0)
    parser.add_argument("--mastery_bound", type=float, default=5.0)

    parser.add_argument("--use_fold_best", type=int, default=1,
                        help="1 时按 --fold 应用 FOLD_BEST_PARAMS 该折最优超参(仅 assist2009,命令行显式值优先)")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)

    args = parser.parse_args()

    fill = {}
    if args.use_fold_best == 1 and args.dataset_name == "assist2009":
        fill = FOLD_BEST_PARAMS.get(args.fold, {})
    for key, value in {**DEFAULT_PARAMS, **fill}.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    main(vars(args))
