from ast import Assert
import os, sys
from re import L
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.cuda import FloatTensor, LongTensor
import numpy as np

ModelConf = {
    "dkt_forget": ["timestamps"]
}

class DktForgetDataset(Dataset):
    def __init__(self, file_path, input_type, folds, qtest=False):
        super(DktForgetDataset, self).__init__()
        self.sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        folds = list(folds)
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_dkt_forget_qtest.pkl"
        else:
            processed_data = file_path + folds_str + "_dkt_forget.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.q_seqs, self.c_seqs, self.r_seqs, self.rgaps, self.sgaps, self.pcounts, self.mask_seqs, self.select_masks, self.max_rgap, self.max_sgap, self.max_pcount, self.dqtest = \
                    self.load_data(self.sequence_path, folds)
                save_data = [self.q_seqs, self.c_seqs, self.r_seqs, self.rgaps, self.sgaps, self.pcounts, self.mask_seqs, self.select_masks, self.max_rgap, self.max_sgap, self.max_pcount, self.dqtest]
            else:
                self.q_seqs, self.c_seqs, self.r_seqs, self.rgaps, self.sgaps, self.pcounts, self.mask_seqs, self.select_masks, self.max_rgap, self.max_sgap, self.max_pcount = \
                        self.load_data(self.sequence_path, folds)
                save_data = [self.q_seqs, self.c_seqs, self.r_seqs, self.rgaps, self.sgaps, self.pcounts, self.mask_seqs, self.select_masks, self.max_rgap, self.max_sgap, self.max_pcount]
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.q_seqs, self.c_seqs, self.r_seqs, self.rgaps, self.sgaps, self.pcounts, self.mask_seqs, self.select_masks, self.max_rgap, self.max_sgap, self.max_pcount, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.q_seqs, self.c_seqs, self.r_seqs, self.rgaps, self.sgaps, self.pcounts, self.mask_seqs, self.select_masks, self.max_rgap, self.max_sgap, self.max_pcount = pd.read_pickle(processed_data)
        print(f"file path: {file_path}, qlen: {len(self.q_seqs)}, clen: {len(self.c_seqs)}, rlen: {len(self.r_seqs)}, \
                max_rgap: {self.max_rgap}, max_sgap: {self.max_sgap}, max_pcount: {self.max_pcount}")

    def __len__(self):
        return len(self.r_seqs)

    def __getitem__(self, index):
        q_seqs, qshft_seqs, c_seqs, cshft_seqs = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        rgaps, sgaps, pcounts = torch.tensor([]), torch.tensor([]), torch.tensor([])

        d, dshft = dict(), dict()
        if "questions" in self.input_type:
            q_seqs = self.q_seqs[index][:-1] * self.mask_seqs[index]
            qshft_seqs = self.q_seqs[index][1:] * self.mask_seqs[index]
        if "concepts" in self.input_type:
            c_seqs = self.c_seqs[index][:-1] * self.mask_seqs[index]
            cshft_seqs = self.c_seqs[index][1:] * self.mask_seqs[index]

        d["rgaps"] = self.rgaps[index][:-1] * self.mask_seqs[index]
        d["sgaps"] = self.sgaps[index][:-1] * self.mask_seqs[index]
        d["pcounts"] = self.pcounts[index][:-1] * self.mask_seqs[index]
        dshft["rgaps"] = self.rgaps[index][1:] * self.mask_seqs[index]
        dshft["sgaps"] = self.sgaps[index][1:] * self.mask_seqs[index]
        dshft["pcounts"] = self.pcounts[index][1:] * self.mask_seqs[index]

        r_seqs = self.r_seqs[index][:-1] * self.mask_seqs[index]
        rshft_seqs = self.r_seqs[index][1:] * self.mask_seqs[index]

        mask_seqs = self.mask_seqs[index]
        select_masks = self.select_masks[index]
        if not self.qtest:
            return q_seqs, c_seqs, r_seqs, qshft_seqs, cshft_seqs, rshft_seqs, mask_seqs, select_masks, d, dshft
        else:
            dcur = dict()
            for key in self.dqtest:
                dcur[key] = self.dqtest[key][index]
            return q_seqs, c_seqs, r_seqs, qshft_seqs, cshft_seqs, rshft_seqs, mask_seqs, select_masks, d, dshft, dcur

    def load_data(self, sequence_path, folds, pad_val=-1):
        seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        repeated_gap, sequence_gap, past_counts = [], [], []
        max_rgap, max_sgap, max_pcount = 0, 0, 0

        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}

        flag = True
        for key in ModelConf["dkt_forget"]:
            if key not in df.columns:
                print(f"key: {key} not in data: {self.sequence_path}! can not run dkt_forget model!")
                flag = False
        assert flag == True
        
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                seq_cids.append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                seq_qids.append([int(_) for _ in row["questions"].split(",")])

            seq_rights.append([int(_) for _ in row["responses"].split(",")])
            seq_mask.append([int(_) for _ in row["selectmasks"].split(",")])

            rgap, sgap, pcount = self.calC(row)
            repeated_gap.append(rgap)
            sequence_gap.append(sgap)
            past_counts.append(pcount)
            max_rgap = max(rgap) if max(rgap) > max_rgap else max_rgap
            max_sgap = max(sgap) if max(sgap) > max_sgap else max_sgap
            max_pcount = max(pcount) if max(pcount) > max_pcount else max_pcount

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        q_seqs, c_seqs, r_seqs = FloatTensor(seq_qids), FloatTensor(seq_cids), FloatTensor(seq_rights)
        rgaps, sgaps, pcounts = LongTensor(repeated_gap), LongTensor(sequence_gap), LongTensor(past_counts)
        seq_mask = LongTensor(seq_mask)

        mask_seqs = (c_seqs[:,:-1] != pad_val) * (c_seqs[:,1:] != pad_val)
        select_masks = (seq_mask[:, 1:] != pad_val)

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            return q_seqs, c_seqs, r_seqs, rgaps, sgaps, pcounts, mask_seqs, select_masks, max_rgap, max_sgap, max_pcount, dqtest

        return q_seqs, c_seqs, r_seqs, rgaps, sgaps, pcounts, mask_seqs, select_masks, max_rgap, max_sgap, max_pcount

    def log2(self, t):
        import math
        return round(math.log(t+1, 2))

    def calC(self, row):
        repeated_gap, sequence_gap, past_counts = [], [], []
        uid = row["uid"]
        # default: concepts
        skills = row["concepts"].split(",") if "concepts" in self.input_type else row["questions"].split(",")
        timestamps = row["timestamps"].split(",")
        dlastskill, dcount = dict(), dict()
        pret = None
        for s, t in zip(skills, timestamps):
            s, t = int(s), int(t)
            if s not in dlastskill or s == -1:
                curRepeatedGap = 0
            else:
                curRepeatedGap = self.log2((t - dlastskill[s]) / 1000 / 60) + 1 # minutes
            dlastskill[s] = t

            repeated_gap.append(curRepeatedGap)
            if pret == None or t == -1:
                curLastGap = 0
            else:
                curLastGap = self.log2((t - pret) / 1000 / 60) + 1
            pret = t
            sequence_gap.append(curLastGap)

            dcount.setdefault(s, 0)
            past_counts.append(self.log2(dcount[s]))
            dcount[s] += 1
        return repeated_gap, sequence_gap, past_counts
            
