from ast import Assert
import os, sys
from re import L
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
import numpy as np

ModelConf = {
    "dkt_forget": ["timestamps"]
}

class ParktDataset(Dataset):
    """Dataset for dkt_forget
        can use to init dataset for: dkt_forget
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds, qtest=False):
        super(ParktDataset, self).__init__()
        self.sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        folds = list(folds)
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_parkt_qtest.pkl"
        else:
            processed_data = file_path + folds_str + "_parkt.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount, self.max_it, self.dqtest = \
                        self.__load_data__(self.sequence_path, folds, file_path=file_path)
                save_data = [self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount, self.max_it, self.dqtest]
            else:
                self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount, self.max_it = \
                        self.__load_data__(self.sequence_path, folds, file_path=file_path)
                save_data = [self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount, self.max_it]
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount, self.max_it, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori, self.dgaps, self.max_rgap, self.max_sgap, self.max_pcount, self.max_it = pd.read_pickle(processed_data)
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}, \
                max_rgap: {self.max_rgap}, max_sgap: {self.max_sgap}, max_pcount: {self.max_pcount}, max_it: {self.max_it}")

    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.dori['rseqs'])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:

           - ** q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        # q_seqs, qshft_seqs, c_seqs, cshft_seqs = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
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
        dcurgaps = dict()
        for key in self.dgaps:
            seqs = self.dgaps[key][index][:-1] * mseqs
            shft_seqs = self.dgaps[key][index][1:] * mseqs
            dcurgaps[key] = seqs
            dcurgaps["shft_"+key] = shft_seqs

        if not self.qtest:
            return dcur, dcurgaps
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dcurgaps, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1, file_path=None):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.

        Returns:
            (tuple): tuple containing:

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **max_rgap (int)**: max num of the repeated time gap
            - **max_sgap (int)**: max num of the sequence time gap (log)
            - **max_pcount (int)**: max num of the past exercise counts
            - **max_it (int)**: max num of the sequence time gap (min)
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
        # seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        # repeated_gap, sequence_gap, past_counts = [], [], []
        dgaps = {"rgaps": [], "sgaps": [], "pcounts": [], "its":[], "tlabel":[], "pretlabel":[], "citlabel":[]}
        max_rgap, max_sgap, max_pcount, max_it = 0, 0, 0, 0

        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}

        # flag = True
        # for key in ModelConf["dkt_forget"]:
        #     if key not in df.columns:
        #         print(f"key: {key} not in data: {self.sequence_path}! can not run dkt_forget model!")
        #         flag = False
        # assert flag == True
        
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            if file_path.find("assist2009") == -1 and file_path.find("assist2015") == -1:
                rgap, sgap, pcount, it, t_label, pret_label, cit_label = self.calC(row)
            else:
                rgap, sgap, pcount, it, t_label, pret_label, cit_label = self.calC_INDEX(row)
            # print(f"sgap:{sgap}")
            # print(f"it:{it}")
            dgaps["rgaps"].append(rgap)
            dgaps["sgaps"].append(sgap)
            dgaps["pcounts"].append(pcount)
            dgaps["its"].append(it)
            dgaps["tlabel"].append(t_label)
            dgaps["pretlabel"].append(pret_label)
            dgaps["citlabel"].append(cit_label)

            max_rgap = max(rgap) if max(rgap) > max_rgap else max_rgap
            max_sgap = max(sgap) if max(sgap) > max_sgap else max_sgap
            max_pcount = max(pcount) if max(pcount) > max_pcount else max_pcount
            max_it = max(it) if max(it) > max_it else max_it

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])
        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        # q_seqs, c_seqs, r_seqs = FloatTensor(seq_qids), FloatTensor(seq_cids), FloatTensor(seq_rights)
        for key in dgaps:
            if key not in ["tlabel", "pretlabel", "citlabel"]:
                # print(f"key:{key},  {dgaps[key]}")
                dgaps[key] = LongTensor(dgaps[key])
            else:
                dgaps[key] = FloatTensor(dgaps[key])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            return dori, dgaps, max_rgap, max_sgap, max_pcount, max_it, dqtest

        return dori, dgaps, max_rgap, max_sgap, max_pcount, max_it

    def log2(self, t):
        import math
        return round(math.log(t+1, 2))

    def calC(self, row):
        repeated_gap, sequence_gap, past_counts, sequence_it, t_label, pret_label, cit_label = [], [], [], [], [], [],[]
        uid = row["uid"]
        # default: concepts
        skills = row["concepts"].split(",") if "concepts" in self.input_type else row["questions"].split(",")
        timestamps = row["timestamps"].split(",")
        dpreskill, dlastskill, dcount = dict(), dict(), dict()
        pret, double_pret = None, None
        cnt = 0
        for idx,(s, t) in enumerate(zip(skills, timestamps)):
            s, t = int(s), int(t)
            if s not in dlastskill or s == -1:
                curRepeatedGap = 0
                curCIt = 0
                dlastskill[s] = -1
            else:
                curRepeatedGap = self.log2((t - dlastskill[s]) / 1000 / 60) + 1 # minutes
                if dpreskill[s] == -1:
                    curCIt = 0.5
                else:
                    precit = int((t - dlastskill[s]))/1000
                    double_precit = int((t - dpreskill[s]))/1000
                    curCIt = round(1 - ((precit + 0.01)/(double_precit + 0.01)),2)
            dpreskill[s] = dlastskill[s]
            dlastskill[s] = t

            repeated_gap.append(curRepeatedGap)
            cit_label.append(curCIt)

            if pret == None or t == -1:
                if t == -1:
                    cnt += 1
                curLastGap = 0
                curLastIt = 0
                curPreT = 0
                if idx == 0:
                    curLableT = 0
                    double_pret = t
                else:
                    if cnt == 2:
                        t_label[-1] = 1
                    else:
                        curLableT = 0

            else:
                curLastGap = self.log2((t - pret) / 1000 / 60) + 1
                # curLastIt = min(int((t - pret) / 1000 / 60) + 1,43200)
                curLastIt = int((t - pret)) / 1000
                curPreIt = int((pret - double_pret))/1000
                curPostIt = int((t - double_pret))/1000
                curLableT = round((curPreIt+0.01)/(curPostIt+0.01),2)
                curPreT = round(1 - ((curLastIt + 0.01)/(curPostIt + 0.01)),2)
                double_pret = pret
            pret = t
            sequence_gap.append(curLastGap)
            sequence_it.append(curLastIt)
            t_label.append(curLableT)
            pret_label.append(curPreT)
            
            dcount.setdefault(s, 0)
            past_counts.append(self.log2(dcount[s]))
            dcount[s] += 1
        t_label = [0]+t_label[2:]+[1]
        pret_label = [0, 0.5]+pret_label[2:]
    
        return repeated_gap, sequence_gap, past_counts, sequence_it, t_label, pret_label, cit_label
            

    def calC_INDEX(self, row):
        repeated_gap, sequence_gap, past_counts, sequence_it, t_label, pret_label, cit_label = [], [], [], [], [], [],[]
        uid = row["uid"]
        # default: concepts
        skills = row["concepts"].split(",") if "concepts" in self.input_type else row["questions"].split(",")
        timestamps = [i for i in range(len(skills))]
        dpreskill, dlastskill, dcount = dict(), dict(), dict()
        pret, double_pret = None, None
        cnt = 0

        for idx,(s, t) in enumerate(zip(skills, timestamps)):
            s, t = int(s), int(t)
            if s not in dlastskill or s == -1:
                curRepeatedGap = 0
                curCIt = 0
                dlastskill[s] = -1
            else:
                curRepeatedGap = (t - dlastskill[s])# minutes
                if dpreskill[s] == -1:
                    # print(f"s:{s}")
                    curCIt = 0.5
                else:
                    # print(f"s:{s}")
                    precit = int((t - dlastskill[s]))
                    double_precit = int((t - dpreskill[s]))
                    curCIt = round(1 - ((precit + 0.01)/(double_precit + 0.01)),2)
            dpreskill[s] = dlastskill[s]
            dlastskill[s] = t

            repeated_gap.append(curRepeatedGap)
            cit_label.append(curCIt)

            if pret == None or t == -1:
                if t == -1:
                    cnt += 1
                curLastGap = 0
                curLastIt = 0
                curPreT = 0
                if idx == 0:
                    curLableT = 0
                    double_pret = t
                else:
                    if cnt == 2:
                        t_label[-1] = 1
                    else:
                        curLableT = 0

            else:
                curLastGap = (t - pret) 
                curLastIt = int((t - pret)) 
                curPreIt = int((pret - double_pret))
                curPostIt = int((t - double_pret))
                curLableT = round((curPreIt+0.01)/(curPostIt+0.01),2)
                curPreT = round(1 - ((curLastIt + 0.01)/(curPostIt + 0.01)),2)
                double_pret = pret
            pret = t
            sequence_gap.append(curLastGap)
            sequence_it.append(curLastIt)
            t_label.append(curLableT)
            pret_label.append(curPreT)

            dcount.setdefault(s, 0)
            past_counts.append(dcount[s])
            dcount[s] += 1
        t_label = [0]+t_label[2:]+[1]
        pret_label = [0, 0.5]+pret_label[2:]
    
        return repeated_gap, sequence_gap, past_counts, sequence_it, t_label, pret_label, cit_label



