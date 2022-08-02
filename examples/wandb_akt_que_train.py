import os
import json
import torch
from pykt.models.akt_que import AKTQue
from pykt.utils import debug_print,set_seed
from pykt.datasets.que_data_loader import KTQueDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(data_config,fold):
    all_folds = set(data_config["folds"])
    train_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds=all_folds - {
                                    fold},
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])

    valid_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["train_valid_file_quelevel"]),
                                input_type=data_config["input_type"], folds={fold},
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])

    test_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                                input_type=data_config["input_type"], folds=[-1],
                                concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])

    test_win_dataset = KTQueDataset(os.path.join(data_config["dpath"], data_config["test_window_file_quelevel"]),
                                    input_type=data_config["input_type"], folds=[-1],
                                    concept_num=data_config['num_c'], max_concepts=data_config['max_concepts'])
    return train_dataset, valid_dataset, test_dataset, test_win_dataset

def main(params):
    if args.use_wandb==1:
        import wandb
        wandb.init()

    print(params)
    print(f"params is {type(params)}")
    set_seed(params['seed'])

    data_config = json.load(open('../configs/data_config.json'))[params['dataset_name']]
    train_dataset, valid_dataset, test_dataset, test_win_dataset = load_data(data_config,args.fold)

    save_dir = os.path.join(params['save_dir'],params['dataset_name'],params['model_name'])
    if params['add_uuid'] == 1 and params["use_wandb"] == 1:
        import uuid
        save_dir = save_dir+"_"+str(uuid.uuid4())
    os.makedirs(save_dir,exist_ok=True)
       
    emb_type = f"{params['emb_type']}|-|{params['loss_mode']}|-|{params['predict_mode']}|-|{params['output_mode']}"
   
    model = AKTQue(num_q=data_config['num_q'], 
                   num_c=data_config['num_c'],
                   num_attn_heads=params['num_attn_heads'],
                   n_blocks=params['n_blocks'],
                   d_ff=params['d_ff'],
                   emb_size=params['emb_size'], 
                   device=device, emb_type=emb_type
                   )

                   
    model.compile(optimizer='adam', lr = params['learning_rate'])

    model.train(train_dataset, valid_dataset, batch_size=params['batch_size'],
                num_epochs=params['num_epochs'], patient=5, shuffle=False, save_model=True, save_dir=save_dir)
    model.load_model(model.save_dir)

    eval_result = model.evaluate(test_dataset,batch_size=params['batch_size'])
    auc,acc = eval_result['auc'],eval_result['acc']
    win_eval_result = model.evaluate(test_win_dataset,batch_size=params['batch_size'])
    auc_win,acc_win = win_eval_result['auc'],win_eval_result['acc']

    print(f"auc: {auc}, acc: {acc}, auc_win: {auc_win}, acc_win: {acc_win}")
    model_report = {"testauc":auc,"testacc":acc,"window_testauc":auc_win,"window_testacc":acc_win,"save_dir":save_dir}
    
    if args.use_wandb==1:
        wandb.log(model_report)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="akt_que")
    parser.add_argument("--emb_type", type=str, default="iekt")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)

    #log
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    #model config
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    #merge modes
    parser.add_argument("--predict_mode", type=str, default='qc-c')
    parser.add_argument("--output_mode", type=str, default='an',help="all/next/an")

    
    args = parser.parse_args()
    args.loss_mode = args.predict_mode 
    params = vars(args)
    main(params)
