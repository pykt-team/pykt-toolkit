#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor, BoolTensor
else:
    from torch import FloatTensor, LongTensor, BoolTensor
import numpy as np
from data_augmentation import split_sequences

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
    def __init__(self, file_path, input_type, folds, qtest=False, aug=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        self.aug = aug
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        print(f"sequence_path: {sequence_path}")
        if self.aug:
            self.dlabel = split_sequences(sequence_path)
        

        dfdata = pd.read_csv(sequence_path)#[0:1000]
        self.dfdata = dfdata[dfdata["fold"].isin(folds)]
        print(f"dfdata: {len(self.dfdata)}")

        self.pad_val = -1

    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        return self.dfdata.shape[0]

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
        if not self.qtest:
            dori = self.__parserow__(index)
        else:
            dori, dqtest = self.__parserow__(index)
        dcur = dict()
        mseqs = dori["masks"]
        for key in dori:
            if key in ["masks", "smasks"]:
                continue
            if len(dori[key]) == 0:
                dcur[key] = dori[key]
                dcur["shft_"+key] = dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            seqs = dori[key][:-1] * mseqs
            shft_seqs = dori[key][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = dori["smasks"]
        # print("tseqs", dcur["tseqs"])
        if not self.qtest:
            return dcur
        else:
            return dcur, dqtest

    
    def collate_fn(self, batch):
        # if not self.qtest:
        #     dmerge = batch
        # else:
        #     dmerge, dmergeqtest = zip(*batch)
        dfinal = dict()
        dfinalqtest = dict()
        qtest = False
        for item in batch:
            if type(item) == type([]):
                qtest = True
                dcur, dqtest = item[0], item[1]
            else:
                dcur = item
            for key in dcur:
                dfinal.setdefault(key, [])
                dfinal[key].append(dcur[key].tolist())
            if qtest:
                for key in dqtest:
                    dfinalqtest.setdefault(key, [])
                    dfinalqtest[key].append(dqtest[key].tolist())

        for key in dfinal:
            if key in ["smasks"]:
                dfinal[key] = BoolTensor(dfinal[key])
            elif key in ["rseqs"]:
                dfinal[key] = FloatTensor(dfinal[key])
            else:
                dfinal[key] = LongTensor(dfinal[key])
        # print(f"shape: {dfinal['rseqs'].shape}")
        # assert False
        if qtest:  
            for key in dfinalqtest:
                dfinalqtest[key] = LongTensor(dfinalqtest[key])
            return dfinal, dfinalqtest
        return dfinal


    def __parserow__(self, index):
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
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}

        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        # print(f"dfdata: {self.dfdata.shape}")
        # print(f"index: {index}")
        # row = self.dfdata.loc[[index],:].to_dict("records")[0]
        row = self.dfdata[index:index+1].to_dict("records")[0]
        # for i, row in self.dfdata[].iterrows()
            #use kc_id or question_id as input
        if "concepts" in self.input_type:
            dori["cseqs"] = [int(_) for _ in row["concepts"].split(",")]
        if "questions" in self.input_type:
            dori["qseqs"] = [int(_) for _ in row["questions"].split(",")]
        if "timestamps" in row:
            dori["tseqs"] = [int(_) for _ in row["timestamps"].split(",")]
        if "usetimes" in row:
            dori["utseqs"] = [int(_) for _ in row["usetimes"].split(",")]
            
        dori["rseqs"] = [int(_) for _ in row["responses"].split(",")]
        dori["smasks"] = [int(_) for _ in row["selectmasks"].split(",")]

        interaction_num += dori["smasks"].count(1)

        if self.qtest:
            dqtest["qidxs"] = [int(_) for _ in row["qidxs"].split(",")]
            dqtest["rests"] = [int(_) for _ in row["rest"].split(",")]
            dqtest["orirow"] = [int(_) for _ in row["orirow"].split(",")]

        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["cseqs"][:-1] != self.pad_val) * (dori["cseqs"][1:] != self.pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][1:] != self.pad_val)
        # print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[1:]
            return dori, dqtest
        return dori

