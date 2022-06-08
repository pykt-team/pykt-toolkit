#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.cuda import FloatTensor, LongTensor
import numpy as np



class KTDataset(Dataset):
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
    def __init__(self, file_path, input_type, folds, qtest=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks, self.dqtest]
            else:
                self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks = self.__load_data__(sequence_path, folds)
                save_data = [self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks]
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks, self.dqtest = pd.read_pickle(processed_data)
            else:
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
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        q_seqs, qshft_seqs, c_seqs, cshft_seqs = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        if "questions" in self.input_type:
            q_seqs = self.q_seqs[index][:-1] * self.mask_seqs[index]
            qshft_seqs = self.q_seqs[index][1:] * self.mask_seqs[index]
        if "concepts" in self.input_type:
            c_seqs = self.c_seqs[index][:-1] * self.mask_seqs[index]
            cshft_seqs = self.c_seqs[index][1:] * self.mask_seqs[index]
        r_seqs = self.r_seqs[index][:-1] * self.mask_seqs[index]
        rshft_seqs = self.r_seqs[index][1:] * self.mask_seqs[index]

        mask_seqs = self.mask_seqs[index]
        select_masks = self.select_masks[index]
        if not self.qtest:
            return q_seqs, c_seqs, r_seqs, qshft_seqs, cshft_seqs, rshft_seqs, mask_seqs, select_masks
        else:
            dcur = dict()
            for key in self.dqtest:
                dcur[key] = self.dqtest[key][index]
            return q_seqs, c_seqs, r_seqs, qshft_seqs, cshft_seqs, rshft_seqs, mask_seqs, select_masks, dcur



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
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                seq_cids.append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                seq_qids.append([int(_) for _ in row["questions"].split(",")])

            seq_rights.append([int(_) for _ in row["responses"].split(",")])
            seq_mask.append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += seq_mask[-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        q_seqs, c_seqs, r_seqs = FloatTensor(seq_qids), FloatTensor(seq_cids), FloatTensor(seq_rights)
        seq_mask = LongTensor(seq_mask)

        mask_seqs = (c_seqs[:,:-1] != pad_val) * (c_seqs[:,1:] != pad_val)
        select_masks = (seq_mask[:, 1:] != pad_val)#(seq_mask[:,:-1] != pad_val) * (seq_mask[:,1:] != pad_val)
        print(f"interaction_num: {interaction_num}")

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return q_seqs, c_seqs, r_seqs, mask_seqs, select_masks, dqtest
        
        return q_seqs, c_seqs, r_seqs, mask_seqs, select_masks