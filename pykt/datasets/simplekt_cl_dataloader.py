import random
import torch
import os
from .data_augmentation import RecWithContrastiveLearningDataset
from torch.utils.data import Dataset
import copy
import pandas as pd
if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor

class CL4KTDataset(Dataset):
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
    def __init__(self, file_path, input_type, num_c, num_q, folds, qtest=False, args=None):
        super(CL4KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        self.num_c = num_c
        self.num_q = num_q
        # print(f"self.num_c:{num_c}")
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + "_cl" + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + "_cl" + folds_str + ".pkl"
        
        self.cl_data = RecWithContrastiveLearningDataset(args=args)

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
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
            if key not in ["cseqs_cl", "qseqs_cl", "rseqs_cl", "uid"]:
                seqs = self.dori[key][index][:-1] * mseqs
                shft_seqs = self.dori[key][index][1:] * mseqs
                dcur[key] = seqs
                dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        dcur["cseqs_cl"] = self.dori["cseqs_cl"][index]
        dcur["rseqs_cl"] = self.dori["rseqs_cl"][index]
        dcur["uid"] = self.dori["uid"][index]
        if self.num_q != 0:
            dcur["qseqs_cl"] = self.dori["qseqs_cl"][index]
        # print("tseqs", dcur["tseqs"])
        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

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
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": [], "cseqs_cl":[], "qseqs_cl":[], "rseqs_cl":[], "uid":[]}

        # seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)#[0:1000]
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            dori["uid"].append([int(row["uid"])])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])
            seqlen = dori["smasks"][-1].count(1)
            if "concepts" in self.input_type:
                cseqs = [int(_) for _ in row["concepts"].split(",")]
                dori["cseqs"].append(cseqs)
                # cseqs_cl = self.cl_data.processed(cseqs, seqlen, self.num_c)
                # dori["cseqs_cl"].append(cseqs_cl)
            if "questions" in self.input_type:
                qseqs = [int(_) for _ in row["questions"].split(",")]
                dori["qseqs"].append(qseqs)
                # qseqs_cl = self.cl_data.processed(qseqs, seqlen, self.num_c)
                # dori["qseqs_cl"].append(qseqs_cl)
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
            rseqs = [int(_) for _ in row["responses"].split(",")]
            dori["rseqs"].append(rseqs)
            interaction_num += dori["smasks"][-1].count(1)

            # cl data
            if "questions" in self.input_type:
                input_ids = [(x[0], x[1], x[2]) for x in zip(cseqs, qseqs, rseqs)]
                cseqs_cl, qseqs_cl, rseqs_cl = self.cl_data.processed(input_ids, seqlen, self.num_c, self.num_q)
            else:
                qseqs = [0 for x in range(len(rseqs))]
                input_ids = [(x[0], x[1], x[2]) for x in zip(cseqs, qseqs, rseqs)]
                cseqs_cl, qseqs_cl, rseqs_cl = self.cl_data.processed(input_ids, seqlen, self.num_c, self.num_q)
            dori["cseqs_cl"].append(cseqs_cl)
            dori["qseqs_cl"].append(qseqs_cl)
            dori["rseqs_cl"].append(rseqs_cl)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
        for key in dori:
            # print(f"key:{key}")
            if key.find("cl") != -1:
                continue
            elif key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        # print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest
        return dori
