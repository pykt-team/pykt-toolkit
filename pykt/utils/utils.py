import os, sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from pykt.datasets.data_loader import KTDataset
from pykt.datasets.dkt_forget_dataloader import DktForgetDataset

def set_seed(seed):
    """Set the global random seed.
    
    Args:
        seed (int): random seed
    """
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)
    # cuda env
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import datetime
def get_now_time():
    """Return the time string, the format is %Y-%m-%d %H:%M:%S

    Returns:
        str: now time
    """
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return dt_string

def debug_print(text,fuc_name=""):
    """Printing text with function name.

    Args:
        text (str): the text will print
        fuc_name (str, optional): _description_. Defaults to "".
    """
    print(f"{get_now_time()} - {fuc_name} - said: {text}")