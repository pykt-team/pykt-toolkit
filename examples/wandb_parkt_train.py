import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--model_name", type=str, default="parkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--final_fc_dim", type=int, default=256)
    parser.add_argument("--final_fc_dim2", type=int, default=256)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--lambda_w1", type=float, default=0.1, help="weight of contrastive learning task")
    parser.add_argument("--lambda_w2", type=float, default=0.05, help="weight of contrastive learning task")

    parser.add_argument("--lamdba_guess", type=float, default=0.3, help="weight of contrastive learning task")
    parser.add_argument("--lamdba_slip", type=float, default=0.5, help="weight of contrastive learning task")

    # parser.add_argument("--augment_type", default="random", type=str, help="default data augmentation types. Chosen from: mask, crop, reorder, substitute, insert, random, combinatorial_enumerate (for multi-view).")
    # parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
    # parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
    # parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator")
    # parser.add_argument("--n_views", default=2, type=int, metavar="N", help="Number of augmented data for each sequence - not studied.")
    # parser.add_argument("--cf_weight", type=float, default=0.1, help="weight of contrastive learning task")
    # parser.add_argument("--seq_representation_instancecl_type", default="mean", type=str, help="operate of item representation overtime. Support types: mean, concatenate")
    # parser.add_argument("--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied.")

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    args = parser.parse_args()

    params = vars(args)
    main(params, args)
