#!/usr/bin/env python
# coding=utf-8

from ast import Assert
import os, sys
from re import L
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
import numpy as np

ModelConf = {
    "lpkt": ["timestamps"]
}

class LPKTDataset(Dataset):
    def __init__(self, file_path, at2idx, it2idx, input_type, folds, qtest=False):
        super(LPKTDataset, self).__init__()
        self.sequence_path = file_path
        self.at2idx = at2idx
        self.it2idx = it2idx
        self.input_type = input_type
        self.qtest = qtest
        folds = list(folds)
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_lpkt_qtest.pkl"
        else:
            processed_data = file_path + folds_str + "_lpkt.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = \
                    self.__load_data__(self.sequence_path, folds)
                save_data = [self.dori, self.dqtest]
                # self.q_seqs, self.c_seqs, self.r_seqs, self.at_seqs, self.it_seqs, self.mask_seqs, self.select_masks, self.dqtest = \
                #     self.load_data(self.sequence_path, folds)
                # save_data = [self.q_seqs, self.c_seqs, self.r_seqs, self.at_seqs, self.it_seqs, self.mask_seqs, self.select_masks, self.dqtest]
            else:
                self.dori = \
                self.__load_data__(self.sequence_path, folds)
                save_data = self.dori
                # self.q_seqs, self.c_seqs, self.r_seqs, self.at_seqs, self.it_seqs, self.mask_seqs, self.select_masks = \
                #         self.load_data(self.sequence_path, folds)
                # save_data = [self.q_seqs, self.c_seqs, self.r_seqs, self.at_seqs, self.it_seqs, self.mask_seqs, self.select_masks]
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

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
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest


    # def __getitem__(self, index):
    #     q_seqs, qshft_seqs, c_seqs, cshft_seqs = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    #     # rgaps, sgaps, pcounts = torch.tensor([]), torch.tensor([]), torch.tensor([])
    #     # at_seqs, it_seqs = torch.tensor([]), torch.tensor([])

    #     d, dshft = dict(), dict()
    #     if "questions" in self.input_type:
    #         q_seqs = self.q_seqs[index][:-1] * self.mask_seqs[index]
    #         qshft_seqs = self.q_seqs[index][1:] * self.mask_seqs[index]
    #     if "concepts" in self.input_type:
    #         c_seqs = self.c_seqs[index][:-1] * self.mask_seqs[index]
    #         cshft_seqs = self.c_seqs[index][1:] * self.mask_seqs[index]

    #     d["at_seqs"] = self.at_seqs[index][:-1] * self.mask_seqs[index]
    #     d["it_seqs"] = self.it_seqs[index][:-1] * self.mask_seqs[index]
    #     # d["pcounts"] = self.pcounts[index][:-1] * self.mask_seqs[index]
    #     dshft["at_seqs"] = self.at_seqs[index][1:] * self.mask_seqs[index]
    #     dshft["it_seqs"] = self.it_seqs[index][1:] * self.mask_seqs[index]
    #     # dshft["pcounts"] = self.pcounts[index][1:] * self.mask_seqs[index]

    #     r_seqs = self.r_seqs[index][:-1] * self.mask_seqs[index]
    #     rshft_seqs = self.r_seqs[index][1:] * self.mask_seqs[index]

    #     mask_seqs = self.mask_seqs[index]
    #     select_masks = self.select_masks[index]
    #     if not self.qtest:
    #         return q_seqs, c_seqs, r_seqs, qshft_seqs, cshft_seqs, rshft_seqs, mask_seqs, select_masks, d, dshft
    #     else:
    #         dcur = dict()
    #         for key in self.dqtest:
    #             dcur[key] = self.dqtest[key][index]
    #         return q_seqs, c_seqs, r_seqs, qshft_seqs, cshft_seqs, rshft_seqs, mask_seqs, select_masks, d, dshft, dcur

    # def load_data(self, sequence_path, folds, pad_val=-1):
    #     seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
    #     seq_at, seq_it = [], []

    #     df = pd.read_csv(sequence_path)
    #     df = df[df["fold"].isin(folds)]
    #     dqtest = {"qidxs": [], "rests":[], "orirow":[]}

    #     flag = True
    #     for key in ModelConf["lpkt"]:
    #         if key not in df.columns:
    #             print(f"key: {key} not in data: {self.sequence_path}! can not run dkt_forget model!")
    #             flag = False
    #     assert flag == True

    #     for i, row in df.iterrows():
    #         #use kc_id or question_id as input
    #         if "concepts" in self.input_type:
    #             seq_cids.append([int(_) for _ in row["concepts"].split(",")])
    #         if "questions" in self.input_type:
    #             seq_qids.append([int(_) for _ in row["questions"].split(",")])

    #         seq_rights.append([int(_) for _ in row["responses"].split(",")])
    #         seq_mask.append([int(_) for _ in row["selectmasks"].split(",")])

    #         at = [self.at2idx[str(int(float(t)))] for t in row["usetimes"].split(",")]
    #         seq_at.append(at) 

    #         #cal interval time
    #         timestamps = [int(float(t)) for t in row["timestamps"].split(",")]
    #         shft_timestamps = [0] + timestamps[:-1]
    #         it = np.maximum(np.minimum((np.array(timestamps) - np.array(shft_timestamps)) // 60, 43200),-1)
    #         tmp_it = [self.it2idx[str(t)] for t in it]
    #         seq_it.append(tmp_it)      

    #         if self.qtest:
    #             dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
    #             dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
    #             dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

    #     q_seqs, c_seqs, r_seqs = FloatTensor(seq_qids), FloatTensor(seq_cids), FloatTensor(seq_rights)
    #     # rgaps, sgaps, pcounts = LongTensor(repeated_gap), LongTensor(sequence_gap), LongTensor(past_counts)
    #     at_seqs, it_seqs  = LongTensor(seq_at), LongTensor(seq_it)
    #     seq_mask = LongTensor(seq_mask)

    #     mask_seqs = (c_seqs[:,:-1] != pad_val) * (c_seqs[:,1:] != pad_val)
    #     select_masks = (seq_mask[:, 1:] != pad_val)

    #     if self.qtest:
    #         for key in dqtest:
    #             dqtest[key] = LongTensor(dqtest[key])[:, 1:]
    #         return q_seqs, c_seqs, r_seqs, at_seqs, it_seqs, mask_seqs, select_masks, dqtest

    #     return q_seqs, c_seqs, r_seqs, at_seqs, it_seqs, mask_seqs, select_masks

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
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": [], "itseqs": []}

        # seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)#[0:1000]
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                use_time = [int(float(t)) for t in row["usetimes"].split(",")]
                use_time = [ x // 1000 for x in use_time]
                at = [self.at2idx[str(ut)] for ut in use_time]
                dori["utseqs"].append(at)
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            #cal interval time
            if "timestamps" in row:
                timestamps = dori["tseqs"][-1]
                shft_timestamps = timestamps[:1] + timestamps[:-1]
                it = np.maximum(np.minimum((np.array(timestamps) - np.array(shft_timestamps)) // 1000 // 60, 43200),-1)
            else:
                it = np.ones(len(dori["cseqs"][-1])).astype(int)
            tmp_it = [self.it2idx[str(t)] for t in it]
            dori["itseqs"].append(tmp_it)

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = FloatTensor(dori[key])
                # dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest
        return dori