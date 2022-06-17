#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.cuda import FloatTensor, LongTensor
import numpy as np
from pykt.preprocess.utils import concept_to_question


class KTQueDataset(Dataset):
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
    def __init__(self, file_path, input_type, folds,concept_num,max_concepts, qtest=False):
        super(KTQueDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        if "questions" not in input_type or "concepts" not in input_type:
            raise("The input types must contain both questions and concepts")

        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])

        processed_data = file_path + folds_str + "_qlevel.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")

            self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks = self.__load_data__(sequence_path, folds)
            save_data = [self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks]
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks = pd.read_pickle(processed_data)
        print(f"file path: {file_path}, qlen: {len(self.q_seqs)}, clen: {len(self.c_seqs)}, rlen: {len(self.r_seqs)}")

    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.r_seqs)

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
        """
        q_seqs, qshft_seqs, c_seqs, cshft_seqs = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        if "questions" in self.input_type:
            q_seqs = self.q_seqs[index][:-1] * self.mask_seqs[index]
            qshft_seqs = self.q_seqs[index][1:] * self.mask_seqs[index]
        if "concepts" in self.input_type:
            c_seqs = self.c_seqs[index][:-1] #* self.mask_seqs[index]
            cshft_seqs = self.c_seqs[index][1:] #* self.mask_seqs[index]
        r_seqs = self.r_seqs[index][:-1] * self.mask_seqs[index]
        rshft_seqs = self.r_seqs[index][1:] * self.mask_seqs[index]

        mask_seqs = self.mask_seqs[index]
        select_masks = self.select_masks[index]
        # if not self.qtest:
        return q_seqs, c_seqs, r_seqs, qshft_seqs, cshft_seqs, rshft_seqs, mask_seqs, select_masks
        
    def get_skill_multi_hot(self, this_skills):
        skill_emb = [0] * self.concept_num
        for s in this_skills:
            skill_emb[s] = 1
        return skill_emb

    def __load_data__(self, sequence_path, folds, pad_val=-1):
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

        seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)[:1000]
        df = df[df["fold"].isin(folds)].copy()
        df = concept_to_question(df)
        # print("convert")
        interaction_num = 0

        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                print(f"len(raw_skills) is {len(raw_skills)}")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        # print(concept)
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills +[-1]*(self.max_concepts-len(skills))
                        # skill_multi_hot = self.get_skill_multi_hot(skills)
                    row_skills.append(skills)
                seq_cids.append(row_skills)
            if "questions" in self.input_type:
                seq_qids.append([int(_) for _ in row["questions"].split(",")])

            seq_rights.append([int(_) for _ in row["responses"].split(",")])
            seq_mask.append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += seq_mask[-1].count(1)

        q_seqs, c_seqs, r_seqs = FloatTensor(seq_qids), FloatTensor(seq_cids), FloatTensor(seq_rights)
        seq_mask = LongTensor(seq_mask)

        mask_seqs = (r_seqs[:,:-1] != pad_val) * (r_seqs[:,1:] != pad_val)
        select_masks = (seq_mask[:, 1:] != pad_val)#(seq_mask[:,:-1] != pad_val) * (seq_mask[:,1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        return q_seqs, c_seqs, r_seqs, mask_seqs, select_masks