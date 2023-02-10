import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--model_name", type=str, default="atdkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_attn_heads", type=int, default=5)
    parser.add_argument("--l1", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.5)
    parser.add_argument("--l3", type=float, default=0.5)
    parser.add_argument("--start", type=int, default=50)
    
    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    args = parser.parse_args()

    params = vars(args)
    main(params)
