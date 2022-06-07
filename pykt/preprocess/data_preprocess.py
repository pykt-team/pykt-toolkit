import os, sys
import argparse
from split_datasets import main as split

dname2paths = {
    "assist2009": "../data/assist2009/skill_builder_data_corrected_collapsed.csv",
    "assist2015": "../data/assist2015/2015_100_skill_builders_main_problems.csv",
    "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
    "bridge2algebra2006": "../data/bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt",
    "statics2011": "../data/statics2011/AllData_student_step_2011F.csv",
    "nips_task34": "../data/nips_task34/train_task_3_4.csv",
    "poj": "../data/poj/poj_log.csv"
}

def main(args):
    dataset_name = args.dataset_name
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
            from assist2009_preprocess import read_data_from_csv
        elif dataset_name == "assist2015":
            from assist2015_preprocess import read_data_from_csv
        elif dataset_name == "algebra2005":
            from algebra2005_preprocess import read_data_from_csv
        elif dataset_name == "bridge2algebra2006":
            from bridge2algebra2006_preprocess import read_data_from_csv
        elif dataset_name == "statics2011":
            from statics2011_preprocess import read_data_from_csv
        elif dataset_name == "nips_task34":
            from nips_task34_preprocess import read_data_from_csv
        elif dataset_name == "poj":
            from poj_preprocess import read_data_from_csv

        if dataset_name != "nips_task34":
            read_data_from_csv(readf, writef)
        else:
            metap = os.path.join(dname, "metadata")
            read_data_from_csv(readf, metap, "task_3_4", writef)

        print("-"*50)
        
        # split
        os.system("rm ../data/" + dataset_name + "/*.pkl")
        split(dname, writef, dataset_name, args.min_seq_len, args.maxlen, args.kfold)
        print("="*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--kfold", type=int, default=5)

    args = parser.parse_args()
    print(args)
    main(args)
