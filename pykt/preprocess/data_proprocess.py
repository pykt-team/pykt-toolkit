import os, sys
from .split_datasets import main as split

def data_proprocess(dataset_name,min_seq_len,maxlen, kfold,configf,dname2paths):
    if dataset_name == "all":
        dataset_names = dname2paths.keys()
    else:
        dataset_names = [dataset_name]
    for dataset_name in dataset_names:
        readf = dname2paths[dataset_name]
        dname = "/".join(readf.split("/")[0:-1])
        writef = os.path.join(dname, "data.txt")
        print(f"Start preprocessing data: {dataset_name}")
        if dataset_name == "assist2009":
            from .assist2009_preprocess import read_data_from_csv
        elif dataset_name == "assist2015":
            from .assist2015_preprocess import read_data_from_csv
        elif dataset_name == "algebra2005":
            from .algebra2005_preprocess import read_data_from_csv
        elif dataset_name == "bridge2algebra2006":
            from .bridge2algebra2006_preprocess import read_data_from_csv
        elif dataset_name == "statics2011":
            from .statics2011_preprocess import read_data_from_csv
        elif dataset_name == "nips_task34":
            from .nips_task34_preprocess import read_data_from_csv
        elif dataset_name == "poj":
            from .poj_preprocess import read_data_from_csv

        if dataset_name != "nips_task34":
            read_data_from_csv(readf, writef)
        else:
            metap = os.path.join(dname, "metadata")
            read_data_from_csv(readf, metap, "task_3_4", writef)
        print("-"*50)
        # split
        os.system("rm " + dname + "/*.pkl")
        split(dname, writef, dataset_name, configf, min_seq_len,maxlen, kfold)
        print("="*100)
