import torch
import numpy as np
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from .promptkt import promptKT
from collections import OrderedDict
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
)
from torch.nn import Module, Dropout, Linear
import functools
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import json

# from typing import Dict

device = "cpu" if not torch.cuda.is_available() else "cuda"


def init_model4promptkt(
    model_name,
    model_config,
    data_config,
    emb_type,
    args=None,
    num_stu=None,
    mode="train",
    train_start=True,
):
    if model_name == "promptkt":
        # 2） 配置每个进程的gpu
        # if mode == "train" and train_start:
        #     print(f"init torch.distributed.init_process_group")
        #     # torch.distributed.init_process_group(backend='nccl')
        #     # torch.cuda.set_device(args.local_rank)
        if emb_type.find("pt") == -1:
            model = promptKT(
                data_config["num_c"],
                data_config["num_q"],
                **model_config,
                emb_type=emb_type,
                emb_path=data_config["emb_path"],
            ).to(device)
        else:
            model = promptKT(
                data_config["num_c"],
                data_config["num_q"],
                **model_config,
                emb_type=emb_type,
                emb_path=data_config["emb_path"],
                num_sgap=data_config["num_sgap"],
            ).to(device)
        print(f"mode:{mode}")
        if mode == "train" and train_start:
            # ref https://pytorch.org/docs/1.13/fsdp.html?highlight=how+fsdp+works
            ignored_modules = [model.emb_q, model.emb_c, model.model.position_emb]

            def my_auto_wrap_policy(
                module: nn.Module, recurse: bool, unwrapped_params: int
            ) -> bool:
                # 对其它层使用默认的自动包装策略
                return size_based_auto_wrap_policy(
                    module, recurse, unwrapped_params, min_num_params=1
                )

            # model = DDP(model)
            ignored_dropouts = {
                module for module in model.modules() if isinstance(module, Dropout)
            }
            if args.emb_type.find("fusion") != -1:
                ignored_modules += [model.autodis]
            ignored_modules += ignored_dropouts
            model = FSDP(
                model,
                auto_wrap_policy=my_auto_wrap_policy,
                ignored_modules=ignored_modules,
            )
    elif model_name == "unikt":
        # 2） 配置每个进程的gpu
        # if mode == "train" and train_start:
        #     print(f"init torch.distributed.init_process_group")
        #     # torch.distributed.init_process_group(backend='nccl')
        #     # torch.cuda.set_device(args.local_rank)
        if emb_type.find("qid_pt") == -1:
            model = UNIKT(
                data_config["num_c"],
                data_config["num_q"],
                **model_config,
                emb_type=emb_type,
                emb_path=data_config["emb_path"],
            ).to(device)
        else:
            model = UNIKT(
                data_config["num_c"],
                data_config["num_q"],
                **model_config,
                emb_type=emb_type,
                emb_path=data_config["emb_path"],
                num_sgap=data_config["num_sgap"],
            ).to(device)
        print(f"mode:{mode}")
        if mode == "train" and train_start:
            # ref https://pytorch.org/docs/1.13/fsdp.html?highlight=how+fsdp+works
            ignored_modules = [
                model.emb_q,
                model.model.position_emb,
                model.emb_c,
            ]
            if args.emb_type.find("prompt_qc") != -1:
                ignored_modules += [model.dataset_prompt, model.autodis]

            def my_auto_wrap_policy(
                module: nn.Module, recurse: bool, unwrapped_params: int
            ) -> bool:
                # 对其它层使用默认的自动包装策略
                return size_based_auto_wrap_policy(
                    module, recurse, unwrapped_params, min_num_params=1
                )

            # model = DDP(model)
            ignored_dropouts = {
                module for module in model.modules() if isinstance(module, Dropout)
            }
            ignored_modules += ignored_dropouts
            # print(ignored_modules)
            model = FSDP(
                model,
                auto_wrap_policy=my_auto_wrap_policy,
                ignored_modules=ignored_modules,
            )
    else:
        print("The wrong model name was used...")
        return None
    return model


def load_model4promptkt(
    model_name,
    model_config,
    data_config,
    emb_type,
    ckpt_path,
    args=None,
    mode="test",
    finetune=False,
):
    model = init_model(model_name, model_config, data_config, emb_type, args, mode=mode)

    # load config to read emb_type
    with open(os.path.join(ckpt_path, "config.json")) as f:
        pretrain_params = json.load(f)

    try:
        net = torch.load(
            os.path.join(
                ckpt_path,
                pretrain_params["params"]["emb_type"] + "_model.module_{}.ckpt",
            ).format(args.pretrain_epoch),
            map_location="cpu",
        )
    except:
        net = torch.load(
            os.path.join(
                ckpt_path,
                pretrain_params["params"]["emb_type"] + "_model.module.ckpt",
            ),
            map_location="cpu",
        )
    print(
        f"load model from:{os.path.join(ckpt_path, pretrain_params['params']['emb_type'] + '_model.module_{}.ckpt').format(args.pretrain_epoch)}"
    )
    # print(f"net:{net}")
    # if args.d_model > 2560: finetune = False
    model.load_state_dict(net)
    # if not finetune:
    #     model.load_state_dict(net)
    # else:
    #     new_state_dict = OrderedDict()
    #     for k, v in net.items():
    #         if "module" not in k:
    #             k = "module." + k
    #             # k = k
    #         else:
    #             k = k.replace("features.module.", "module.features.")
    #         new_state_dict[k] = v
    #     model.load_state_dict(new_state_dict)
    return model
