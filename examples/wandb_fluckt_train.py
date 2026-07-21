import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--model_name", type=str, default="fluckt")

    # qid: 原始的AKT
    # qid_conv_ker_noexp

    parser.add_argument("--emb_type", type=str, default="qid_conv_ker_noexp")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.05)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--num_buckets", type=int, default=256)  # t5位置编码的参数
    parser.add_argument("--max_distance", type=int, default=256) # t5位置编码的参数

    parser.add_argument("--bar_d", type=int, default=64)
    parser.add_argument("--max_local_shift", type=float, default=0.0)

    parser.add_argument("--n_hat", type=int, default=0) # longrope

    # mamba parameters
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=2)
    parser.add_argument("--expand", type=int, default=1)
    # causal conv
    parser.add_argument("--kernel_size", type=int, default=5)
    
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
   
    args = parser.parse_args()

    params = vars(args)
    main(params)
