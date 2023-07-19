import random
import copy
import itertools


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self, tao=0.2, gamma=0.7, beta=0.2):
        self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=beta)]
        print("Total augmentation numbers: ", len(self.data_augmentation_methods))

    def __call__(self, sequence, num_c, num_q):
        # randint generate int x in range: a <= x <= b
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        # print(augment_method.__class__.__name__) # debug usage
        return augment_method(sequence, num_c, num_q)


def _ensmeble_sim_models(top_k_one, top_k_two):
    # only support top k = 1 case so far
    #     print("offline: ",top_k_one, "online: ", top_k_two)
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][1]]


class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence, num_c, num_q):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length < 1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index : start_index + sub_seq_length]
            return cropped_seq


class Mask(object):
    """Randomly mask k items given a sequence"""

    def __init__(self, gamma=0.7):
        self.gamma = gamma

    def __call__(self, sequence, num_c, num_q):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.gamma * len(copied_sequence))
        mask = [(num_c, num_q, 0) for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence


class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""

    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence, num_c, num_q):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.beta * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        sub_seq = copied_sequence[start_index : start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + copied_sequence[start_index + sub_seq_length :]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq


if __name__ == "__main__":
    reorder = Reorder(beta=0.2)
    sequence = [
        14052,
        10908,
        2776,
        16243,
        2726,
        2961,
        11962,
        4672,
        2224,
        5727,
        4985,
        9310,
        2181,
        3428,
        4156,
        16536,
        180,
        12044,
        13700,
    ]
    rs = reorder(sequence)
    crop = Crop(tao=0.2)
    rs = crop(sequence)
    # rt = RandomType()
    # rs = rt(sequence)
    n_views = 5
    enum_type = CombinatorialEnumerateType(n_views=n_views)
    for i in range(40):
        if i == 20:
            print("-------")
        es = enum_type(sequence)
