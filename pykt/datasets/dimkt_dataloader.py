import pandas as pd
from torch.utils.data import Dataset
# from torch.cuda import FloatTensor, LongTensor
from torch import FloatTensor, LongTensor
import os
import numpy as np
from tqdm import tqdm
import csv

class DIMKTDataset(Dataset):
    def __init__(self,dpath,file_path,input_type,folds,qtest=False, diff_level=None):
        super(DIMKTDataset,self).__init__()
        self.sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        self.diff_level = diff_level
        skills_difficult_path = dpath+f'/skills_difficult_{diff_level}.csv'
        questions_difficult_path =  dpath+f'/questions_difficult_{diff_level}.csv'
        if not os.path.exists(skills_difficult_path) or not os.path.exists(questions_difficult_path):
            print("start compute difficults")
            train_file_path = dpath+"/train_valid_sequences.csv"
            df = pd.read_csv(train_file_path)
            difficult_compute(df,skills_difficult_path,questions_difficult_path,diff_level=self.diff_level)
            
        folds = sorted(list(folds)) 
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + f"_dimkt_qtest_{diff_level}.pkl"
        else:
            processed_data = file_path + folds_str + f"_dimkt_{diff_level}.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(self.sequence_path,skills_difficult_path,questions_difficult_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(self.sequence_path,skills_difficult_path,questions_difficult_path,folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}, sdlen: {len(self.dori['sdseqs'])}, qdlen:{len(self.dori['qdseqs'])}")


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
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qd_seqs (torch.tensor)**: question difficult sequence of the 0~seqlen-2 interactions
            - **sd_seqs (torch.tensor)**: knowledge concept difficult sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions
            - **qdshft_seqs (torch.tensor)**: question difficult sequence of the 1~seqlen-1 interactions
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **sdshft_seqs (torch.tensor)**: knowledge concept difficult sequence of the 1~seqlen-1 interactions
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
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

    def __load_data__(self,sequence_path,sds_path,qds_path,folds,pad_val=-1):
        dori = {"qseqs":[],"cseqs":[],"rseqs":[],"tseqs":[],"utseqs":[],"smasks": [],"sdseqs":[],"qdseqs":[]}
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]
        sds = {}
        qds = {}
        with open(sds_path,'r',encoding="UTF8") as f:
            reader = csv.reader(f)
            sds_keys = next(reader)
            sds_vals = next(reader)
            for i in range(len(sds_keys)):
                sds[int(sds_keys[i])] = int(sds_vals[i])
        with open(qds_path,'r',encoding="UTF8") as f:
            reader = csv.reader(f)
            qds_keys = next(reader)
            qds_vals = next(reader)
            for i in range(len(qds_keys)):
                qds[int(qds_keys[i])] = int(qds_vals[i])
        interaction_num = 0
        dqtest = {"qidxs":[],"rests":[], "orirow":[]}
        sds_keys = [int(_) for _ in sds_keys]
        qds_keys = [int(_) for _ in qds_keys]
        for i,row in df.iterrows():
            if "concepts" in self.input_type:
                temp = [int(_) for _ in row["concepts"].split(",")]
                temp_1 = []
                dori["cseqs"].append(temp)
                for j in temp:
                    if j == -1:
                        temp_1.append(-1)
                    elif j not in sds_keys:
                        temp_1.append(1)
                    else:
                        temp_1.append(int(sds[j]))
                dori["sdseqs"].append(temp_1)
            if "questions" in self.input_type:
                temp = [int(_) for _ in row["questions"].split(",")]
                temp_1 = []
                dori["qseqs"].append(temp)
                for j in temp:
                    if j == -1:
                        temp_1.append(-1)
                    elif j not in qds_keys:
                        temp_1.append(1)
                    else:
                        temp_1.append(int(qds[j]))
                dori["qdseqs"].append(temp_1)
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

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
        print(f"interaction_num: {interaction_num}")

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            
            return dori, dqtest
        return dori    

def difficult_compute(df,sds_path,qds_path,diff_level):
    concepts = []
    questions = []
    responses = []
    for i,row in tqdm(df.iterrows()):
        concept = [int(_) for _ in row["concepts"].split(",")]
        question = [int(_) for _ in row["questions"].split(",")]
        response = [int(_) for _ in row["responses"].split(",")]
        length = len(response)
        index = -1
        for j in range(length):
            if response[length - j - 1] != -1:
                index = length - j
                break
        concepts = concepts+concept[:index]
        questions = questions+question[:index]
        responses = responses+response[:index]
    df2 = pd.DataFrame({'concepts':concepts,'questions':questions,'responses':responses})
        
    skill_difficult(df2,sds_path,'concepts','responses',diff_level=diff_level)
    question_difficult(df2,qds_path,'questions','responses',diff_level=diff_level)
    
    return
    
def skill_difficult(df,sds_path,concepts,responses,diff_level):
    sd = {}
    df = df.reset_index(drop=True)
    set_skills = set(np.array(df[concepts]))
    for i in tqdm(set_skills):
        count = 0
        idx = df[(df.concepts == i)].index.tolist()
        tmp_data = df.iloc[idx]
        correct_1 = tmp_data[responses]
        if len(idx) < 30:
            sd[i] = 1
            continue
        else:
            for j in np.array(correct_1):
                count += j
            if count == 0:
                sd[i] = 1
                continue
            else:
                avg = int((count/len(correct_1))*diff_level)+1
                sd[i] = avg
    with open(sds_path,'w',newline='',encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(sd.keys())
        writer.writerow(sd.values())
        
    return 

def question_difficult(df,qds_path,questions,responses,diff_level):
    qd = {}
    df = df.reset_index(drop=True)
    set_questions = set(np.array(df[questions]))
    for i in tqdm(set_questions):
        count = 0
        idx = df[(df.questions == i)].index.tolist()
        tmp_data = df.iloc[idx]
        correct_1 = tmp_data[responses]
        if len(idx) < 30:
            qd[i] = 1
            continue
        else:
            for j in np.array(correct_1):
                count += j
            if count == 0:
                qd[i] = 1
                continue
            else:
                avg = int((count/len(correct_1))*diff_level)+1
                qd[i] = avg
    with open(qds_path,'w',newline='',encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(qd.keys())
        writer.writerow(qd.values())

    return 

