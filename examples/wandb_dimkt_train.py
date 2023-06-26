import argparse
import warnings
warnings.filterwarnings("ignore")
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2012")
    parser.add_argument("--model_name", type=str, default="dimkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    # parser.add_argument("--num_epochs", type=int, default=1000)#max epochs
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    #model params
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--emb_size", type=int, default=128)
    #parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=199)
    parser.add_argument("--difficult_levels", type=int, default=100)#Cqs,Ckc=100 in data preprocess
    
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    args = parser.parse_args()

    params = vars(args)
    main(params)
