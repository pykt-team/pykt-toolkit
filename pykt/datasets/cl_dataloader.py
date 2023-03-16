import torch
from .cl4kt_utils import  (
    augment_kt_seqs,
)
from .data_loader import KTDataset
from torch.utils.data import Dataset
from collections import defaultdict



class CL4KTDataset(KTDataset):
    def __init__(self, file_path, input_type, folds, qtest=False,data_config={}):
        super().__init__(file_path, input_type, folds, qtest)
        self.num_questions = data_config['num_q']
        self.num_skills = data_config['num_c']
        
        # count the number of correct responses for each skill
        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        
        for s_list, r_list in zip(self.dori['cseqs'].cpu().numpy(),self.dori['rseqs'].cpu().numpy()):
            for s, r in zip(s_list, r_list):
                if s==-1:
                    break
                skill_correct[s] += r
                skill_count[s] += 1
        
        # easier and harder skills
        skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s]) for s in skill_correct
        }#correct rate
        
        ordered_skills = [
            item[0] for item in sorted(skill_difficulty.items(), key=lambda x: x[1])
        ]
        self.skill_difficulty = skill_difficulty
        self.ordered_skills = ordered_skills
        self.easier_skills = {}
        self.harder_skills = {}
        for i, s in enumerate(ordered_skills):
            if i == 0:  # the hardest
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = s
            elif i == len(ordered_skills) - 1:  # the easiest
                self.easier_skills[s] = s
                self.harder_skills[s] = ordered_skills[i - 1]
            else:
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = ordered_skills[i - 1]
        
        
    def get_father(self, index):
        return  super().__getitem__(index)
    
    def __getitem__(self, index):
        if not self.qtest:
            dcur = super().__getitem__(index)
        else:
            dcur, dqtest = super().__getitem__(index)
            
        #get full sequence
        q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
        masks = dcur["masks"]
        pad_mask = torch.tensor([True]).to(masks.device)
        attention_mask = torch.cat((pad_mask, masks), dim=-1)
        # print(f"q shape is {q.shape}")
        # print(f"qshft shape is {qshft.shape}")
        cq = torch.cat((q[0:1], qshft), dim=-1)
        cc = torch.cat((c[0:1], cshft), dim=-1)
        cr = torch.cat((r[0:1], rshft), dim=-1)
        
        dcur["skills"] = cc
        dcur["responses"] = cr
        dcur["attention_mask"] = attention_mask
        if self.num_questions!=0:
            dcur["questions"] = cq
        else:
            dcur["questions"] = cc
       
        if not self.qtest:
            return dcur
        else:
            return dcur, dqtest
        

class SimCLRDatasetWrapper(Dataset):
    """Wrapper for SimCLR training
    Args:
        ds (Dataset): The original dataset to be wrapped
        seq_len (int): Length of the sequence to be processed
        mask_prob (float): Probability of masking a token
        crop_prob (float): Probability of cropping a sequence
        permute_prob (float): Probability of permuting a sequence
        replace_prob (float): Probability of replacing a token with another token
        negative_prob (float): Probability of generating a negative response
        eval_mode (bool): Flag to indicate whether in evaluation mode or not
    """
    def __init__(
        self,
        ds: Dataset,
        seq_len: int,
        mask_prob: float,
        crop_prob: float,
        permute_prob: float,
        replace_prob: float,
        negative_prob: float,
        random_action: int,
        eval_mode=False,
    ):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.crop_prob = crop_prob
        self.permute_prob = permute_prob
        self.replace_prob = replace_prob
        self.negative_prob = negative_prob
        self.random_action = random_action
        self.eval_mode = eval_mode

        # Get some information from the original dataset
        self.num_questions = self.ds.num_questions
        self.num_skills = self.ds.num_skills
        self.q_mask_id = self.num_questions + 1
        self.s_mask_id = self.num_skills + 1
        self.easier_skills = self.ds.easier_skills
        self.harder_skills = self.ds.harder_skills

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        """
        Get an item from the wrapped dataset

        Args:
            index (int): Index of the item to get

        Returns:
            A dictionary containing the processed data
        """
        # Get the original data
        original_data = self.ds[index]
        if self.num_questions!=0:
            q_seq = original_data["questions"]
        else:
            q_seq = original_data["skills"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]
        attention_mask = original_data["attention_mask"]
        if self.eval_mode:
            # If in evaluation mode, return the original data
            return {
                "questions": q_seq,
                "skills": s_seq,
                "responses": r_seq,
                "attention_mask": attention_mask,
            }

        else:
            # If not in evaluation mode, augment the data
            if self.num_questions<=0:
                q_seq_list = original_data["skills"].cpu().numpy()#.tolist()
            else:
                q_seq_list = original_data["questions"].cpu().numpy()#.tolist()
            s_seq_list = original_data["skills"].cpu().numpy()#.tolist()
            r_seq_list = original_data["responses"].cpu().numpy()#.tolist()
            attention_mask_aug = original_data["attention_mask"].cpu().numpy()
            #
            q_seq_aug, s_seq_aug, r_seq_aug = q_seq_list[attention_mask_aug].tolist(), s_seq_list[attention_mask_aug].tolist(), r_seq_list[attention_mask_aug].tolist()


            t1 = augment_kt_seqs(
                q_seq_aug,
                s_seq_aug,
                r_seq_aug,
                self.mask_prob,
                self.crop_prob,
                self.permute_prob,
                self.replace_prob,
                self.negative_prob,
                self.easier_skills,
                self.harder_skills,
                self.q_mask_id,
                self.s_mask_id,
                self.seq_len,
                seed=index,
                num_questions = self.num_questions,
                random_action = self.random_action
            )

            t2 = augment_kt_seqs(
                q_seq_aug,
                s_seq_aug,
                r_seq_aug,
                self.mask_prob,
                self.crop_prob,
                self.permute_prob,
                self.replace_prob,
                self.negative_prob,
                self.easier_skills,
                self.harder_skills,
                self.q_mask_id,
                self.s_mask_id,
                self.seq_len,
                seed=index + 1,
                num_questions = self.num_questions,
                random_action = self.random_action
            )
            # Unpack the augmented data
            aug_q_seq_1, aug_s_seq_1, aug_r_seq_1, negative_r_seq, attention_mask_1 = t1
            aug_q_seq_2, aug_s_seq_2, aug_r_seq_2, _, attention_mask_2 = t2

            # Convert the augmented data to tensors
            aug_q_seq_1 = torch.tensor(aug_q_seq_1, dtype=torch.long)
            aug_q_seq_2 = torch.tensor(aug_q_seq_2, dtype=torch.long)
            aug_s_seq_1 = torch.tensor(aug_s_seq_1, dtype=torch.long)
            aug_s_seq_2 = torch.tensor(aug_s_seq_2, dtype=torch.long)
            aug_r_seq_1 = torch.tensor(aug_r_seq_1, dtype=torch.long)
            aug_r_seq_2 = torch.tensor(aug_r_seq_2, dtype=torch.long)
            negative_r_seq = torch.tensor(negative_r_seq, dtype=torch.long)
            attention_mask_1 = torch.tensor(attention_mask_1, dtype=torch.bool)
            attention_mask_2 = torch.tensor(attention_mask_2, dtype=torch.bool)

            # Return the augmented data in a dictionary
            if self.num_questions!=0:
                original_data['questions'] = (aug_q_seq_1, aug_q_seq_2, q_seq)
            else:
                original_data['questions'] = (aug_s_seq_1, aug_s_seq_2, s_seq)
            original_data['skills'] = (aug_s_seq_1, aug_s_seq_2, s_seq)
            original_data['responses'] = (aug_r_seq_1, aug_r_seq_2, r_seq, negative_r_seq)
            original_data['attention_mask'] = (attention_mask_1, attention_mask_2, attention_mask)
            
            return original_data

    def __getitem__(self, index):
        return self.__getitem_internal__(index)

