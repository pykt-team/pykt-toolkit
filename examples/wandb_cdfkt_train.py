# CDFKT 依赖 mamba-ssm 与 causal-conv1d(训练环境需自行安装),
# 且需要 data/<dataset_name>/question_concept_map.npy。
# 下列默认值为原始调参记录中 nips_task34 fold2 的最优超参数。
import argparse

from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="nips_task34")
    parser.add_argument("--model_name", type=str, default="cdfkt")
    parser.add_argument("--emb_type", type=str, default="NIPS34")
    parser.add_argument("--save_dir", type=str, default="saved_model/cdfkt")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--dropout", type=float, default=0.02479219844695262)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=4.995497603743331e-06)
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.0004233643821974351)
    parser.add_argument("--base_intensity", type=float, default=0.0035787294084155263)
    parser.add_argument("--alpha_init_scale", type=float, default=0.01237861633441871)
    parser.add_argument("--min_half_life", type=float, default=1.0)
    parser.add_argument("--max_half_life", type=float, default=16.0)
    parser.add_argument("--sa_layerscale_init", type=float, default=0.5243660665573915)
    parser.add_argument("--sa_dropout", type=float, default=0.11921708954769812)
    parser.add_argument("--sa_gate_bias", type=float, default=3.157777130289173)
    parser.add_argument("--fusion_dropout", type=float, default=0.22306120985336225)
    parser.add_argument("--d_state", type=int, default=48)
    parser.add_argument("--d_conv", type=int, default=2)
    parser.add_argument("--expand", type=int, default=1)

    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=0)

    args = parser.parse_args()

    main(vars(args))
