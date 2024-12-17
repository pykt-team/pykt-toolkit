import os
import argparse
import json
import copy
import torch
import numpy as np

from pykt.models import evaluate, evaluate_question, load_model4promptkt
from pykt.datasets import init_test_datasets

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)


def main(params, args):
    if params["use_wandb"] == 1:
        import wandb

        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")

    save_dir, batch_size, fusion_type = (
        params["save_dir"],
        params["bz"],
        params["fusion_type"].split(","),
    )

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in [
            "use_wandb",
            "learning_rate",
            "add_uuid",
            "l2",
            "num_workers",
            "pretrain_epoch",
            "project_name",
            "local_rank",
            "train_mode",
        ]:
            if remove_item in model_config:
                del model_config[remove_item]
        trained_params = config["params"]
        model_name, dataset_name, emb_type, fold = (
            trained_params["model_name"],
            trained_params["dataset_name"],
            trained_params["emb_type"],
            trained_params["fold"],
        )
        if model_name in ["saint", "sakt", "cdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len
        # if model_name in ["promptkt"]:
        #     model_config["final_fc_dim"] = 512
        #     model_config["final_fc_dim2"] = 1024
        #     model_config["nheads"] = 16
        #     model_config["d_model"] = 512
        #     model_config["d_ff"] = 1024
        #     model_config["num_attn_heads"] = 16
        #     model_config["n_blocks"] = 24
        if model_name in ["stosakt"]:
            train_args = argparse.ArgumentParser()
            print(f"train_args;{train_args}")
            # args_dict = vars(train_args)
            # args_dict.update(config["train_args"])
            # print(f"train_args:{train_args.hidden_size}")
        if model_name in ["unikt"]:
            for remove_item in [
                "num_layers",
                "nheads",
                "loss1",
                "loss2",
                "loss3",
                "start",
                "cf_weight",
                "t_weight",
            ]:
                if remove_item in model_config:
                    del model_config[remove_item]

    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        if model_name in ["promptkt", "unikt"]:
            dataset_name = params["dataset_name"]
            re_mapping = trained_params["re_mapping"]
            if re_mapping == "0":
                re_mapping = False
            if re_mapping == "1":
                re_mapping = True
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"] or emb_type.find("time") != -1:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name in ["lpkt"]:
            print("running  prediction")
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]
        elif model_name in ["promptkt", "unikt"]:
            data_config["num_q"] = config["data_config"]["num_q"]
            data_config["num_c"] = config["data_config"]["num_c"]

    dataet_name = args.dataset_name
    (
        test_loader,
        test_window_loader,
        test_question_loader,
        test_question_window_loader,
    ) = init_test_datasets(
        data_config, model_name, batch_size, fold, args=args, re_mapping=re_mapping
    )

    print(
        f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}"
    )
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")


    for remove_item in [
        "use_wandb",
        "learning_rate",
        "add_uuid",
        "l2",
        "global_bs",
        "num_gpus",
        "pretrain_path",
        "num_workers",
        "pretrain_epoch",
    ]:
        if remove_item in model_config:
            del model_config[remove_item]
    if params["load_finetune"] == "1":
        model = load_model4promptkt(
            model_name,
            model_config,
            data_config,
            emb_type,
            save_dir,
            mode="test",
            finetune=False,
            args=args,
        )
    else:
        model = load_model4promptkt(
            model_name,
            model_config,
            data_config,
            emb_type,
            save_dir,
            args,
            mode="test",
        )

    testauc, testacc = -1, -1

    # save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt")
    # testauc, testacc = evaluate(model, test_loader, model_name, save_test_path)
    # print(f"testauc: {testauc}, testacc: {testacc}")

    window_testauc, window_testacc = -1, -1
    if dataset_name in ["statics2011", "assist2015", "poj"] or model_name in [
        "lpkt",
        "gnn4kt",
    ]:
        # init_weight = torch.zeros(data_config["num_c"]+1,data_config["num_c"]+1)
        # np.save(f"/root/autodl-nas/huangshuyan/attention/pykt-toolkit/examples/new_attn_distribution/{dataset_name}_{fold}_{emb_type}", init_weight)
        # attn_cnt_path = f"/root/autodl-nas/huangshuyan/attention/pykt-toolkit/examples/new_attn_distribution/{dataset_name}_{fold}_{emb_type}.npy"
        # attn_cnt_path = ""
        # if attn_cnt_path == "":
        save_test_window_path = os.path.join(
            save_dir, model.emb_type + "_test_window_predictions.txt"
        )
        # else:
        #     save_test_window_path = os.path.join(save_dir, model.emb_type+"_train_predictions.txt")
        # print(f"test_window_loader:{test_window_loader}")
        window_testauc, window_testacc = evaluate(
            model,
            test_window_loader,
            model_name,
            save_test_window_path,
            dataset_name,
            fold,
        )
        # print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
        print(f"window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    if model_name in ["promptkt"]:
        save_test_window_path = os.path.join(
            save_dir, model.emb_type + "_test_window_predictions_pretrain.txt"
        )
        window_testauc, window_testacc = evaluate(
            model,
            test_window_loader,
            model_name,
            save_test_window_path,
        )
        # print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
        print(f"window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    if model_name in ["unikt"]:
        save_test_path = os.path.join(
            save_dir, model.emb_type + "_test_predictions_unikt.txt"
        )
        testauc, testacc, _ = evaluate(model, test_loader, model_name, save_test_path)
        print(f"testauc: {testauc}, testacc: {testacc}")

        save_test_window_path = os.path.join(
            save_dir, model.emb_type + "_test_window_predictions_unikt.txt"
        )
        window_testauc, window_testacc = evaluate(
            model,
            test_window_loader,
            model_name,
            save_test_window_path,
        )
        # print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
        print(f"window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    question_testauc, question_testacc = -1, -1
    question_window_testauc, question_window_testacc = -1, -1

    dres = {
        "testauc": testauc,
        "testacc": testacc,
        "window_testauc": window_testauc,
        "window_testacc": window_testacc,
    }

    q_testaucs, q_testaccs = -1, -1
    qw_testaucs, qw_testaccs = -1, -1
    # if "test_question_file" in data_config and not test_question_loader is None:
    #     save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt")
    #     q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
    #     for key in q_testaucs:
    #         dres["oriauc"+key] = q_testaucs[key]
    #     for key in q_testaccs:
    #         dres["oriacc"+key] = q_testaccs[key]

    if dataset_name in [
        "nips_task34",
        "assist2009",
        "algebra2005",
        "bridge2algebra2006",
        "peiyou",
        "ednet",
        "ednet5w",
    ]:
        if (
            "test_question_window_file" in data_config
            and not test_question_window_loader is None
        ):
            save_test_question_window_path = os.path.join(
                save_dir, model.emb_type + "_test_question_window_predictions.txt"
            )
            qw_testaucs, qw_testaccs = evaluate_question(
                model,
                test_question_window_loader,
                model_name,
                fusion_type,
                save_test_question_window_path,
                dataset_name,
                fold,
            )
            for key in qw_testaucs:
                dres["windowauc" + key] = qw_testaucs[key]
            for key in qw_testaccs:
                dres["windowacc" + key] = qw_testaccs[key]

    print(
        f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}"
    )
    print(
        f"question_testauc: {question_testauc}, question_testacc: {question_testacc}, question_window_testauc: {question_window_testauc}, question_window_testacc: {question_window_testacc}"
    )

    print(dres)
    raw_config = json.load(open(os.path.join(save_dir, "config.json")))
    dres.update(raw_config["params"])
    dres["dataset_name"] = dataset_name

    if params["use_wandb"] == 1:
        wandb.log(dres)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="late_fusion")
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--pretrain_epoch", type=int, default=0)
    parser.add_argument("--load_finetune", type=str, default="0")
    parser.add_argument("--eval_epoch", type=str, default="0")

    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params, args)
