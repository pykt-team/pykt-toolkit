import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # algebra2005_dkt_forget_qid_dkt_tiaocan_algebra2005_200_0.001_0.25_224_0/
    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--model_name", type=str, default="cfdkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=224)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.25)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_attn_heads", type=int, default=5)
    parser.add_argument("--l1", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.5)
    parser.add_argument("--l3", type=float, default=0.5)
    parser.add_argument("--start", type=int, default=50)
    
    parser.add_argument("--emb_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    # algebra2006_dkt_forget_qid_dkt_tiaocan_algebra2006_100_0.001_0.4_3407_0/
    # parser.add_argument("--dataset_name", type=str, default="bridge2algebra2006")
    # parser.add_argument("--model_name", type=str, default="cfdkt")
    # parser.add_argument("--emb_type", type=str, default="qid")
    # parser.add_argument("--save_dir", type=str, default="saved_model")
    # # parser.add_argument("--learning_rate", type=float, default=1e-5)
    # parser.add_argument("--seed", type=int, default=3407)
    # parser.add_argument("--fold", type=int, default=0)
    # parser.add_argument("--dropout", type=float, default=0.4)

    # parser.add_argument("--num_layers", type=int, default=2)
    # parser.add_argument("--num_attn_heads", type=int, default=5)
    # parser.add_argument("--l1", type=float, default=0.5)
    # parser.add_argument("--l2", type=float, default=0.5)
    # parser.add_argument("--l3", type=float, default=0.5)
    # parser.add_argument("--start", type=int, default=50)
    
    # parser.add_argument("--emb_size", type=int, default=100)
    # parser.add_argument("--learning_rate", type=float, default=1e-3)

    # parser.add_argument("--use_wandb", type=int, default=1)
    # parser.add_argument("--add_uuid", type=int, default=1)
    # saved_model_300_0.01_0.5_42_0
    # parser.add_argument("--dataset_name", type=str, default="nips_task34")
    # parser.add_argument("--model_name", type=str, default="cfdkt")
    # parser.add_argument("--emb_type", type=str, default="qid")
    # parser.add_argument("--save_dir", type=str, default="saved_model")
    # # parser.add_argument("--learning_rate", type=float, default=1e-5)
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--fold", type=int, default=0)
    # parser.add_argument("--dropout", type=float, default=0.5)

    # parser.add_argument("--num_layers", type=int, default=2)
    # parser.add_argument("--num_attn_heads", type=int, default=5)
    # parser.add_argument("--l1", type=float, default=0.5)
    # parser.add_argument("--l2", type=float, default=0.5)
    # parser.add_argument("--l3", type=float, default=0.5)
    # parser.add_argument("--start", type=int, default=50)
    
    # parser.add_argument("--emb_size", type=int, default=300)
    # parser.add_argument("--learning_rate", type=float, default=1e-2)

    # parser.add_argument("--use_wandb", type=int, default=1)
    # parser.add_argument("--add_uuid", type=int, default=1)
   
    args = parser.parse_args()

    params = vars(args)
    main(params)
