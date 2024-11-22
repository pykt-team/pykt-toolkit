import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--model_name", type=str, default="rekt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--d", type=int, default=128)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    
    parser.add_argument("--learning_rate", type=float, default=0.002)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    args = parser.parse_args()

    params = vars(args)
    main(params)
#CUDA_VISIBLE_DEVICES=0 python wandb_simplekt_train.py >> wandb_simplekt_train.log 2>&1 &