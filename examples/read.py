import pickle

import numpy as np

a = np.load("./save_grad.npz",allow_pickle=True)
print(a["arr_0"].shape)
