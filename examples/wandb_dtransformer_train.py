import argparse
from wandb_train import main

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--model_name", type=str, default="dtransformer")
    parser.add_argument("--emb_type", type=str, default="qid_cl")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    
    parser.add_argument("--n_know", type=int, default=16)


    parser.add_argument("--lambda_cl", type=float, default=0.1)
    parser.add_argument("--window",type=int, default=1)

    parser.add_argument("--proj",type=str2bool, default=True)
    parser.add_argument("--hard_neg", type=str2bool,default=False)
    
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)

    args = parser.parse_args()

    params = vars(args)
    main(params)