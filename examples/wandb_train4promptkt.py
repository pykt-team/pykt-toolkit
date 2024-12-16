import os
import argparse
import json

import torch

torch.set_num_threads(4)
from torch.optim import SGD, Adam
import copy

from pykt.models import train_model4promptkt, evaluate, init_model4promptkt, load_model4promptkt
from pykt.utils import set_seed, debug_print
from pykt.datasets import init_dataset4train
import datetime

import os
import torch.distributed as dist


def init_process():
    """初始化进程组"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


init_process()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


local_rank = 0
node_rank = 0


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def save_config(train_config, model_config, data_config, params, save_dir, args=None):
    # print(f"type_args:{type(args)}")
    if args:
        d = {
            "train_config": train_config,
            "model_config": model_config,
            "data_config": data_config,
            "params": params,
            "train_args": vars(args),
        }
    else:
        d = {
            "train_config": train_config,
            "model_config": model_config,
            "data_config": data_config,
            "params": params,
        }
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)


def main(params, args=None):
    global local_rank, node_rank
    local_rank = args.local_rank
    node_rank = int(os.environ["RANK"])
    project_name = args.project_name
    # local_rank = os.environ.get('LOCAL_RANK') # args.local_rank
    # print(f"local_rank====",local_rank)
    torch.cuda.set_device(local_rank)

    if "use_wandb" not in params:
        params["use_wandb"] = 1

    if params["use_wandb"] == 1 and local_rank == 0 and node_rank == 0:
        import wandb

        wandb.init(project=project_name)
        use_wandb = True
    else:
        use_wandb = False

        # wandb.init()

    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir, re_mapping = (
        params["model_name"],
        params["dataset_name"],
        params["fold"],
        params["emb_type"],
        params["save_dir"],
        params["re_mapping"],
    )
    re_mapping = False if re_mapping == "0" else True
    print(f"re_mapping:{re_mapping}")

    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        if model_name in ["promptkt", "unikt"]:
            seqlen = params["seq_len"]
            train_config["seq_len"] = seqlen
            if seqlen == 1024:
                if params["d_model"] <= 1024:
                    train_config["batch_size"] = 16  ## because of OOM
                else:
                    train_config["batch_size"] = 16  ## because of OOM
            elif seqlen == 200:
                if params["d_model"] > 2560:
                    train_config["batch_size"] = 32  ## because of OOM
                elif params["d_model"] > 1536 and params["d_model"] <= 2560:
                    train_config["batch_size"] = 160  ## because of OOM
                elif params["d_model"] > 1024 and params["d_model"] <= 1536:
                    train_config["batch_size"] = 640  ## because of OOM
                    # train_config["batch_size"] = (
                    #     args.global_bs // args.num_gpus // args.num_workers
                    # )
                else:
                    train_config["batch_size"] = 1280  # 512
                    # train_config["batch_size"] = (
                    #     args.global_bs // args.num_gpus // args.num_workers
                    # )
                if args.pretrain_path != "" and args.train_mode == "ft":
                    # if args.dataset_name in ["peiyou", "bridge2algebra2006"]:
                    train_config["batch_size"] = (
                        512 // args.num_gpus // args.num_workers
                    )
                    # else:
                    #     train_config["batch_size"] = 512
            else:  # seqlen = 512
                train_config["batch_size"] = 32  ## because of OOM

        model_config = copy.deepcopy(params)
        if model_name in ["promptkt", "unikt"]:
            for key in [
                "model_name",
                "dataset_name",
                "emb_type",
                "save_dir",
                "fold",
                "seed",
                "train_ratio",
                "not_select_dataset",
            ]:
                del model_config[key]

        if "batch_size" in params:
            train_config["batch_size"] = params["batch_size"]
        if "num_epochs" in params:
            train_config["num_epochs"] = params["num_epochs"]
        # model_config = {"d_model": params["d_model"], "n_blocks": params["n_blocks"], "dropout": params["dropout"], "d_ff": params["d_ff"]}
    batch_size, num_epochs, optimizer = (
        train_config["batch_size"],
        train_config["num_epochs"],
        train_config["optimizer"],
    )

    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    # if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
    #     train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]

    rank0_print("Start init data")
    rank0_print(dataset_name, model_name, data_config, fold, batch_size)

    if model_name not in ["simplekt_sr", "parkt", "promptkt", "simplekt", "unikt"]:
        train_loader, valid_loader, curtrain = init_dataset4train(
            dataset_name, model_name, data_config, fold, batch_size
        )
        # print(f"curtrain:{len(curtrain)}")
    elif model_name in ["promptkt", "unikt"]:
        not_select_dataset = params["not_select_dataset"]
        if not_select_dataset == "all":
            not_select_dataset = None
        train_loader, valid_loader = init_dataset4train(
            dataset_name,
            model_name,
            data_config,
            fold,
            batch_size,
            args=args,
            not_select_dataset=not_select_dataset,
            re_mapping=re_mapping,
        )

    params_str = "_".join(
        [
            str(v)
            for k, v in params.items()
            if not k in ["other_config", "pretrain_path", "pretrain_epoch"]
        ]
    )

    rank0_print(f"params: {params}, params_str: {params_str}")
    if params["add_uuid"] == 1 and params["use_wandb"] == 1:
        import uuid

        # if not model_name in ['saint','saint++']:
        params_str = params_str + f"_{ str(uuid.uuid4())}"
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path) and local_rank == 0 and node_rank == 0:
        os.makedirs(ckpt_path)
    # 不在0号卡不保存
    # if model_name in ["unikt", "promptkt"] and local_rank != 0 and node_rank != 0:
    #     ckpt_path = None

    rank0_print(
        f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}"
    )
    rank0_print(f"model_config: {model_config}")
    rank0_print(f"train_config: {train_config}")

    # if model_name in ["stosakt"]:
    #     save_config(
    #         train_config,
    #         model_config,
    #         data_config[dataset_name],
    #         params,
    #         ckpt_path,
    #         args,
    #     )
    if local_rank == 0 and node_rank == 0:
        save_config(
            train_config, model_config, data_config[dataset_name], params, ckpt_path
        )

    learning_rate = params["learning_rate"]
    for remove_item in [
        "use_wandb",
        "learning_rate",
        "add_uuid",
        "l2",
        "global_bs",
        "num_gpus",
        "num_workers",
        "pretrain_epoch",
        "project_name",
        "local_rank",
        "train_mode",
    ]:
        if remove_item in model_config:
            del model_config[remove_item]

    rank0_print(f"model_name:{model_name}")

    if model_name in ["promptkt", "unikt"]:
        pretrain_path = params["pretrain_path"]
        if pretrain_path == "":
            del model_config["pretrain_path"]
            model = init_model4promptkt(
                model_name, model_config, data_config[dataset_name], emb_type, args
            )
        else:
            with open(os.path.join(pretrain_path, "config.json")) as fin:
                config = json.load(fin)
                model_config = copy.deepcopy(config["model_config"])
                for remove_item in [
                    "use_wandb",
                    "learning_rate",
                    "add_uuid",
                    "l2",
                    "num_gpus",
                    "num_workers",
                    "global_bs",
                    "pretrain_path",
                    "pretrain_epoch",
                    "project_name",
                    "local_rank",
                    "train_mode",
                ]:
                    if remove_item in model_config:
                        del model_config[remove_item]
                trained_params = config["params"]
            model = load_model4promptkt(
                model_name,
                model_config,
                data_config[dataset_name],
                emb_type,
                pretrain_path,
                args,
                mode="train",
                finetune=True,
            )
        rank0_print(
            f"model_parameter:{sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())}"
        )
    else:
        model = init_model4promptkt(
            model_name, model_config, data_config[dataset_name], emb_type, args
        )
        rank0_print(
            f"model_parameter:{sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())}"
        )
    rank0_print(f"model is {model}")

    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True

    rank0_print("Start training")

    if model_name in ["promptkt", "unikt"]:
        if params["pretrain_path"] == "":
            global_bs = 2560
        else:
            global_bs = params["global_bs"]
        num_gpus = params["num_gpus"]
        num_workers = params["num_workers"]
        gradient_accumulation_steps = max(
            global_bs / num_workers / num_gpus / train_config["batch_size"], 1.0
        )
        rank0_print(f"gradient_accumulation_steps:{gradient_accumulation_steps}")
        (
            testauc,
            testacc,
            window_testauc,
            window_testacc,
            validauc,
            validacc,
            best_epoch,
        ) = train_model4promptkt(
            model,
            train_loader,
            valid_loader,
            num_epochs,
            opt,
            ckpt_path,
            None,
            None,
            save_model,
            dataset_name,
            fold,
            gradient_accumulation_steps=gradient_accumulation_steps,
            args=args,
            use_wandb=use_wandb,
        )

    if save_model:
        if model_name in ["promptkt", "unikt"]:
            best_model = init_model4promptkt(
                model_name,
                model_config,
                data_config[dataset_name],
                emb_type,
                args,
                train_start=False,
            )

        if ckpt_path is not None:
            try:
                net = torch.load(
                    os.path.join(
                        ckpt_path,
                        emb_type
                        + "_model.module_{}.ckpt".format(str(args.pretrain_epoch + 1)),
                    )
                )
            except:
                net = torch.load(
                    os.path.join(ckpt_path, emb_type + "_model.module.ckpt")
                )
            best_model.load_state_dict(net)

    rank0_print(
        "fold\tmodelname\tembtype\ttestauc\ttestacc\twindow_testauc\twindow_testacc\tvalidauc\tvalidacc\tbest_epoch"
    )

    rank0_print(
        f"{fold}\t{model_name}\t{emb_type}\t{round(testauc, 4)}\t{round(testacc, 4)}\t{round(window_testauc, 4)}\t{round(window_testacc, 4)}\t{validauc}\t{validacc}\t{best_epoch}"
    )

    if ckpt_path is not None:
        model_save_path = os.path.join(
            ckpt_path, f"{emb_type}_model.module_{args.pretrain_epoch + 1}.ckpt"
        )
    else:
        model_save_path = None

    rank0_print(f"end:{datetime.datetime.now()}")

    if params["use_wandb"] == 1 and local_rank == 0 and node_rank == 0:
        wandb.log(
            {
                "validauc": validauc,
                "validacc": validacc,
                "best_epoch": best_epoch,
                "model_save_path": model_save_path,
            }
        )
