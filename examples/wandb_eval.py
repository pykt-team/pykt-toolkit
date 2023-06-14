import os
import argparse
import json
import copy
from pykt.config import que_type_models
import torch
torch.set_num_threads(2)

from pykt.models import evaluate_splitpred_question, load_model, lpkt_evaluate_multi_ahead

def main(params):
    if params['use_wandb'] ==1:
        import wandb
        wandb.init()
    save_dir, use_pred, ratio = params["save_dir"], params["use_pred"], params["train_ratio"]

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])

        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]

        trained_params = config["params"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
        seq_len = config["train_config"]["seq_len"]
        if model_name in ["saint", "sakt", "atdkt"]:
            model_config["seq_len"] = seq_len
        data_config = config["data_config"]



    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")
    use_pred = True if use_pred == 1 else False

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    # for use_pred in [True, False]:
    #     for ratio in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print(f"Start predict use_pred: {use_pred}, ratio: {ratio}...")
    atkt_pad = True if params["atkt_pad"] == 1 else False
    if model_name == "atkt":
        save_test_path = os.path.join(save_dir, model.emb_type+"_test_ratio"+str(ratio)+"_"+str(use_pred)+"_"+str(atkt_pad)+"_predictions.txt")
    else:
        save_test_path = os.path.join(save_dir, model.emb_type+"_test_ratio"+str(ratio)+"_"+str(use_pred)+"_predictions.txt")
    # model, testf, model_name, save_path="", use_pred=False, train_ratio=0.2
    # testauc, testacc = evaluate_splitpred(model, test_loader, model_name, save_test_path)
    testf = os.path.join(data_config["dpath"], params["test_filename"])
    if model_name in que_type_models and model_name != "lpkt":
        dfinal = model.evaluate_multi_ahead(data_config,batch_size=16,ob_portions=ratio,accumulative=use_pred)
    elif model_name in ["lpkt"]:
        dfinal = lpkt_evaluate_multi_ahead(model, data_config,batch_size=64,ob_portions=ratio,accumulative=use_pred)
    else:
        dfinal = evaluate_splitpred_question(model, data_config, testf, model_name, save_test_path, use_pred, ratio, atkt_pad)
    for key in dfinal:
        print(key, dfinal[key])
    dfinal.update(config["params"])
    if params['use_wandb'] ==1:
        wandb.log(dfinal)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--test_filename", type=str, default="test.csv")
    parser.add_argument("--use_pred", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--atkt_pad", type=int, default=0)
    parser.add_argument("--use_wandb", type=int, default=1)

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
