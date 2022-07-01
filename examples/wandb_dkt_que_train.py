import os
import json
import torch
from pykt.models.dkt_que import DKTQue
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
       
    emb_type = f"{params['emb_type']}|-|{params['loss_mode']}|-|{params['predict_mode']}"
   
    model = DKTQue(num_q=data_config['num_q'], num_c=data_config['num_c'],
                   emb_size=params['emb_size'], device=device, emb_type=emb_type,
                   dropout=params['dropout'])

    model.compile(optimizer='adam', lr = params['learning_rate'])

    model.train(train_dataset, valid_dataset, batch_size=128,
                num_epochs=params['num_epochs'], patient=5, shuffle=False, save_model=True, save_dir=save_dir)
    model.load_model(model.save_dir)

    auc,acc = model.evaluate(test_dataset,batch_size=64)
    auc_win,acc_win = model.evaluate(test_win_dataset,batch_size=64)

    print(f"auc: {auc}, acc: {acc}, auc_win: {auc_win}, acc_win: {acc_win}")
    
    model_report = {"testauc":auc,"testacc":acc,"window_testauc":auc_win,"window_testacc":acc_win,"save_dir":save_dir}
    
    
    if args.use_wandb==1:
        wandb.log(model_report)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="dkt_que")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=200)
    
    
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--emb_size", type=int, default=200)
    parser.add_argument("--loss_mode", type=str, default='q')
    parser.add_argument("--predict_mode", type=str, default='q')
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    args = parser.parse_args()

    params = vars(args)
    main(params)
