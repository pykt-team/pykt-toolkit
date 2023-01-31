#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.cuda import FloatTensor, LongTensor
import numpy as np
import json

class KTPretrainQueDataset(Dataset):
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
    def __init__(self, file_path, input_type, folds, concept_num, max_concepts, qtest=False):
        super(KTPretrainQueDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts

        self.question_file = os.path.join(os.path.abspath(os.path.dirname(file_path)), "questions.json")
        self.kc_routes_map_file = os.path.join(os.path.abspath(os.path.dirname(file_path)), "kc_routes_map.json")
        self.kc_level = 10

        if "questions" not in input_type or "concepts" not in input_type:
            raise("The input types must contain both questions and concepts")

        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])

        processed_data = file_path + folds_str + "_qlevel.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")

            self.dori = self.__load_data__(sequence_path, folds)
            save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
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
            if key in ['cseqs', 'qcroutes']:
                seqs = self.dori[key][index][:-1,:]
                shft_seqs = self.dori[key][index][1:,:]
                # print(f"key: {key}, seqs: {seqs.shape}, shft_seqs: {shft_seqs.shape}, ori: {self.dori[key][index].shape}")
            else:
                seqs = self.dori[key][index][:-1] * mseqs
                shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        # assert False
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        # print("tseqs", dcur["tseqs"])
        return dcur

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
        kcdict, qtypdict = self.__load_qinfo__()
        dori = {"qseqs": [], "qtypes": [], "qcroutes": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}

        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)].copy()#[0:1000]
        interaction_num = 0
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")

                row_croutes = []

                for concept in raw_skills:
                    croutes = []
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                        croutes = [[-1] * self.kc_level] * self.max_concepts
                    else:
                        skills = [int(_) for _ in concept.split("_")]
                        skills = skills +[-1]*(self.max_concepts-len(skills))

                        dcurcs = []
                        for c in concept.split("_"): # 一个知识点一个知识点处理
                            crs = kcdict[int(c)] if int(c) != -1 else []
                            if len(crs) > 0:
                                croutes = crs + [-2] * (self.kc_level - len(crs))
                            else:
                                croutes = [-1] * self.kc_level
                            dcurcs.append(croutes)
                        # dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
                        # dori["crouteseqs"].append(dcurcs)
                        
                        croutes = dcurcs +[[-1] * self.kc_level] * (self.max_concepts-len(dcurcs))
                        # print(croutes)
                        # assert False

                    row_skills.append(skills)
                    row_croutes.append(croutes)
                # print(f"raw: {len(raw_skills)}, row_skills: {len(row_skills)}, row_croutes: {len(row_croutes)}")
                # assert False
                dori["cseqs"].append(row_skills)
                dori["qcroutes"].append(row_croutes)
                # qcroutes

            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
                dori["qtypes"].append([qtypdict[int(_)] if int(_) != -1 else -1 for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])
                
            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)


        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        mask_seqs = (dori["rseqs"][:,:-1] != pad_val) * (dori["rseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])
        return dori

    def __load_kcroutes_map__(self):
        with open(self.kc_routes_map_file) as fin:
            orimap = json.load(fin)
            kcmap = {orimap[key]: int(key) for key in orimap}
        return kcmap
            
    
    def __load_qinfo__(self):
        kcmap = self.__load_kcroutes_map__()
        with open(self.question_file, "r") as f:
            qdata = json.load(f)
        # kc
        kcdict = dict()
        for qid in qdata:
            # select cur kc, use the larger level
            curkc = []
            for inf in qdata[qid]["kc_routes"]:
                curcs = inf.split("----")
                curcs = [int(kcmap[_]) for _ in curcs]
                curkc = curcs if len(curcs) > len(curkc) else curkc
            qid = int(qid)
            kcdict[qid] = curkc
        # qtype
        typed = {"填空": 0, "单选": 1}
        qtydict = dict()
        for qid in qdata:
            typ = typed[qdata[qid]["type"]]
            qid = int(qid)
            qtydict[qid] = typ
            
        return kcdict, qtydict
