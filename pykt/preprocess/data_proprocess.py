import os, sys

def process_raw_data(dataset_name,dname2paths):
    readf = dname2paths[dataset_name]
    dname = "/".join(readf.split("/")[0:-1])
    writef = os.path.join(dname, "data.txt")
    print(f"Start preprocessing data: {dataset_name}")
    if dataset_name == "assist2009":
        from .assist2009_preprocess import read_data_from_csv
    elif dataset_name == "assist2012":
        from .assist2012_preprocess import read_data_from_csv
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
    elif dataset_name == "slepemapy":
        from .slepemapy_preprocess import read_data_from_csv
    elif dataset_name == "assist2017":
        from .assist2017_preprocess import read_data_from_csv
    elif dataset_name == "junyi2015":
        from .junyi2015_preprocess import read_data_from_csv, load_q2c
    elif dataset_name in ["ednet","ednet5w"]:
        from .ednet_preprocess import read_data_from_csv
    elif dataset_name == "peiyou":
        from .aaai2022_competition import read_data_from_csv, load_q2c
    
    if dataset_name == "junyi2015":
        dq2c = load_q2c(readf.replace("junyi_ProblemLog_original.csv","junyi_Exercise_table.csv"))
        read_data_from_csv(readf, writef, dq2c)
    elif dataset_name == "peiyou":
        fname = readf.split("/")[-1]
        dq2c = load_q2c(readf.replace(fname,"questions.json"))
        read_data_from_csv(readf, writef, dq2c)
    elif dataset_name == "ednet5w":
        dname, writef = read_data_from_csv(readf, writef, dataset_name=dataset_name)
    elif dataset_name != "nips_task34":#default case
        read_data_from_csv(readf, writef)
    else:
        metap = os.path.join(dname, "metadata")
        read_data_from_csv(readf, metap, "task_3_4", writef)
     
    return dname,writef
