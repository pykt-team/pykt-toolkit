import os, sys
import argparse
from pykt.preprocess import process_raw_data
from pykt.preprocess.split_datasets import main as split

dname2paths = {
    "assist2009": "../data/assist2009/skill_builder_data_corrected_collapsed.csv",
    "assist2015": "../data/assist2015/2015_100_skill_builders_main_problems.csv",
    "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
    "bridge2algebra2006": "../data/bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt",
    "statics2011": "../data/statics2011/AllData_student_step_2011F.csv",
    "nips_task34": "../data/nips_task34/train_task_3_4.csv",
    "poj": "../data/poj/poj_log.csv",
    "slepemapy": "../data/slepemapy/answer.csv"
}
configf = "../configs/data_config.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--kfold", type=int, default=5)
    args = parser.parse_args()

    print(args)

    # process raw data
    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    print("-"*50)
    # split
    os.system("rm " + dname + "/*.pkl")
    split(dname, writef, args.dataset_name, configf, args.min_seq_len,args.maxlen, args.kfold)
    print("="*100)
