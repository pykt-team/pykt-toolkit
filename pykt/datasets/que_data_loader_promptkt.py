#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset

# from torch.cuda import FloatTensor, LongTensor
from torch import FloatTensor, LongTensor
import numpy as np

datasets_dic = {
    "ednet": 0,
    "assist2009": 1,
    "algebra2005": 2,
    "bridge2algebra2006": 3,
    "nips_task34": 4,
    "peiyou": 5,
    "ednet5w": 6,
    "ednet_all": 7,
}


class KTQueDataset_promptKT(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).

    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """

    def __init__(
        self,
        file_path,
        input_type,
        folds,
        concept_num,
        max_concepts,
        qtest=False,
        not_select_dataset=None,
        train_ratio=1.0,
        dataset_name=None,
    ):
        super(KTQueDataset_promptKT, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        self.dataset_name = dataset_name
        print(f"dataset_name: {self.dataset_name}")

        if "questions" not in input_type or "concepts" not in input_type:
            raise ("The input types must contain both questions and concepts")

        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])

        if not_select_dataset is not None:
            processed_data = (
                file_path
                + folds_str
                + f"_non_{not_select_dataset}_{train_ratio}_qlevel_gpt4kt.pkl"
            )
        else:
            processed_data = file_path + folds_str + "gpt4kt_qlevel.pkl"

        if not os.path.exists(processed_data):
            print(f"file path {file_path}")
            print(f"Start preprocessing {file_path} fold: {folds_str}...")

            self.dori = self.__load_data__(
                sequence_path,
                folds,
                not_select_dataset=not_select_dataset,
                train_ratio=train_ratio,
                dataset_name=self.dataset_name,
            )
            save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            print(f"dataset_name: {dataset_name}")
            # print(f"dataset_id: {datasets_dic[dataset_name]}")
            self.dori = pd.read_pickle(processed_data)
        print(
            f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}"
        )

    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks", "dataset"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_" + key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            if key == "cseqs":
                seqs = self.dori[key][index][:-1, :]
                shft_seqs = self.dori[key][index][1:, :]
            else:
                try:
                    seqs = self.dori[key][index][:-1] * mseqs
                    shft_seqs = self.dori[key][index][1:] * mseqs
                except:
                    print(self.dori[key][index])
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        dcur["dataset_id"] = self.dori["dataset"][index]
        # print("tseqs", dcur["tseqs"])
        return dcur

    def get_skill_multi_hot(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb

    def __load_data__(
        self,
        sequence_path,
        folds,
        pad_val=-1,
        not_select_dataset=None,
        train_ratio=1.0,
        dataset_name=None,
    ):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]):
            pad_val (int, optional): pad value. Defaults to -1.

        Returns:
            (tuple): tuple containing

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {
            "qseqs": [],
            "cseqs": [],
            "rseqs": [],
            "tseqs": [],
            "utseqs": [],
            "smasks": [],
            "dataset": [],
        }

        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)].copy()  # [0:1000]
        if dataset_name not in ["pretrain", "mix_data"]:
            df["dataset"] = [dataset_name for i in range(df.shape[0])]
        print(f"df:{df.shape}")
        if not_select_dataset is not None:
            # not_select_dataset = [
            #     int(dataset_id) for dataset_id in not_select_dataset.split(",")
            # ]
            not_select_dataset = [
                dataset_id for dataset_id in not_select_dataset.split(",")
            ]
            print(f"before_not_select_dataset:{df.shape}")
            # new_df = df[~(df.dataset.isin(not_select_dataset))]
            new_df = df[df.dataset.isin(not_select_dataset)]
            print(f"before_not_select_dataset1:{new_df.shape}")
            # if train_ratio < 1.0:
            #     sub_df = df[df.dataset.isin(not_select_dataset)].sample(
            #         frac=train_ratio, random_state=1024
            #     )
            #     new_df = pd.concat([new_df, sub_df], ignore_index=True)
            if train_ratio >= 50 and len(folds) != 1:
                new_df = new_df.sample(n=train_ratio, random_state=1024)
            # if train_ratio < 1.0 and len(folds) != 1:
            #     new_df = new_df.sample(frac=train_ratio, random_state=1024)
            df = new_df
            print(f"after_not_select_dataset2:{df.shape}")
        interaction_num = 0
        for i, row in df.iterrows():
            # use kc_id or question_id as input
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills + [-1] * (self.max_concepts - len(skills))
                    row_skills.append(skills)
                dori["cseqs"].append(row_skills)
            if "questions" in self.input_type:
                try:
                    dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
                except:
                    que_seq = row["questions"]
                    print(f"i:{i}, questions:{que_seq}")
            # if "timestamps" in row:
            #     dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            # if "usetimes" in row:
            #     dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])
            dori["dataset"].append(int(datasets_dic[row["dataset"]]))

            interaction_num += dori["smasks"][-1].count(1)
        for key in dori:
            if key not in ["rseqs"]:  # in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:, :-1] != pad_val) * (
            dori["rseqs"][:, 1:] != pad_val
        )
        dori["masks"] = mask_seqs

        dori["smasks"] = dori["smasks"][:, 1:] != pad_val
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])
        return dori
