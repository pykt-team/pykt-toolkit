from .simplekt_cl_utils import Crop, Mask, Reorder, Random
import torch
import math

class RecWithContrastiveLearningDataset(object):
    def __init__(self, args):
        self.args = args
        # currently apply one transform, will extend to multiples
        self.augmentations = {
            "crop": Crop(tao=args.tao),
            "mask": Mask(gamma=args.gamma),
            "reorder": Reorder(beta=args.beta),
            "random": Random(tao=args.tao, gamma=args.gamma, beta=args.beta),
        }
        if self.args.augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.augment_type]
        # number of augmentations for each sequences, current support two
        self.n_views = self.args.n_views

    def nCr(self, n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)

    def _one_pair_data_augmentation(self, input_ids, seqlen, num_c, num_q):
        """
        provides two positive samples given one sequence
        """
        max_len = len(input_ids)
        augmented_cids_seqs, augmented_qids_seqs, augmented_res_seqs = [], [], []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids[:seqlen], num_c, num_q)
            pad_len = max_len - len(augmented_input_ids)
            augmented_input_ids = augmented_input_ids + [(num_c,num_q,0)] * pad_len
            augmented_input_ids = augmented_input_ids[-max_len :]

            assert len(augmented_input_ids) == max_len

            augmented_cids = [x[0] for x in augmented_input_ids]
            augmented_qids = [x[1] for x in augmented_input_ids]
            augmented_res = [x[2] for x in augmented_input_ids]
            cur_cids_tensors = torch.tensor(augmented_cids, dtype=torch.long)
            cur_qids_tensors = torch.tensor(augmented_qids, dtype=torch.long)
            cur_res_tensors = torch.tensor(augmented_res, dtype=torch.long)
            augmented_cids_seqs.append(cur_cids_tensors)
            augmented_qids_seqs.append(cur_qids_tensors)
            augmented_res_seqs.append(cur_res_tensors)
        return augmented_cids_seqs, augmented_qids_seqs, augmented_res_seqs


    def _process_sequence_label_signal(self, seq_label_signal):
        seq_class_label = torch.tensor(seq_label_signal, dtype=torch.long)
        return seq_class_label

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def processed(self, input_id, seqlen, num_c, num_q):
        # print(f"seqlen:{seqlen}")
        augmented_cids_list, augmented_qids_list, augmented_res_list = [], [], []
        # if n_views == 2, then it's downgraded to pair-wise contrastive learning
        total_augmentaion_pairs = self.nCr(self.n_views, 2)
        for i in range(total_augmentaion_pairs):
            augmented_cids_seqs, augmented_qids_seqs, augmented_res_seqs = self._one_pair_data_augmentation(input_id, seqlen, num_c, num_q)
            augmented_cids_list.append(augmented_cids_seqs)
            augmented_qids_list.append(augmented_qids_seqs)
            augmented_res_list.append(augmented_res_seqs)

        # add supervision of sequences
        # seq_class_label = self._process_sequence_label_signal(seq_label_signal)
        # print(f"cf_tensors_list:{cf_tensors_list}")
        return augmented_cids_list, augmented_qids_list, augmented_res_list
