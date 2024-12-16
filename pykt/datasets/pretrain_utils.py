import pandas as pd
import json
import numpy as np
import os


def get_pretrain_data(data_config, train_ratio=1.0):
    # print(f"datapath:{data_config['dpath']}")
    # print(f"train_ratio:{train_ratio}")

    uni_path = "/".join(data_config["dpath"].split("/")[:-1])

    for file_type in ["train_valid_quelevel.csv", "train_valid_sequences_quelevel.csv"]:
        file_type_ = file_type.strip(".csv")
        full_data_path = os.path.join(
            data_config["dpath"], f"{file_type_}_pretrain_nomapping.csv"
        )
        if "pretrain" in data_config["dpath"]:
            datasets = [
                "assist2009",
                "algebra2005",
                "bridge2algebra2006",
                "nips_task34",
                "ednet",
                "peiyou",
                "ednet5w",
            ]
        else:
            datasets = [
                "assist2009",
                "algebra2005",
                "bridge2algebra2006",
                "nips_task34",
                "peiyou",
                "ednet_all",
            ]
            print(f"datasets:{datasets}")
        if not os.path.exists(full_data_path):
            finaldf = merge_data(uni_path, datasets, file_type)
            if not os.path.exists(data_config["dpath"]):
                os.mkdir(data_config["dpath"])
            finaldf.to_csv(full_data_path, index=None)
        else:
            finaldf = pd.read_csv(full_data_path)

        ins, ss, qs, cs, seqnum = calStatistics(
            df=finaldf, stares=[], key="original train+valid question level"
        )
        print(
            f"{file_type}  sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}"
        )
        for data in datasets:
            final_sub_df = finaldf[finaldf["dataset"] == data]
            ins_, ss_, qs_, cs_, seqnum_ = calStatistics(
                df=final_sub_df, stares=[], key="original train+valid question level"
            )
            print(
                f"dataset:{data}, ins_:{ins_}, ss_:{ss_}, qs_:{qs_}, cs_:{cs_}, seqnum_:{seqnum_}"
            )

    return


def merge_data(uni_path, datasets, file_type, train_ratio=None):
    all_datasets = []
    for dataset in datasets:
        df_train = pd.read_csv(f"{uni_path}/{dataset}/{file_type}")
        dataset_name = [dataset for i in range(df_train.shape[0])]
        df_train["dataset"] = dataset_name
        all_datasets.append(df_train)
    all_df = pd.concat(all_datasets)
    return all_df


def calStatistics(df, stares, key):
    allin, allselect = 0, 0
    allqs, allcs = set(), set()
    for i, row in df.iterrows():
        rs = row["responses"].split(",")
        curlen = len(rs) - rs.count("-1")
        allin += curlen
        if "selectmasks" in row:
            ss = row["selectmasks"].split(",")
            slen = ss.count("1")
            allselect += slen
        if "concepts" in row:
            cs = row["concepts"].split(",")
            fc = list()
            for c in cs:
                cc = c.split("_")
                fc.extend(cc)
            curcs = set(fc) - {"-1"}
            allcs |= curcs
        if "questions" in row:
            qs = row["questions"].split(",")
            curqs = set(qs) - {"-1"}
            allqs |= curqs
    stares.append(",".join([str(s) for s in [key, allin, df.shape[0], allselect]]))
    return allin, allselect, len(allqs), len(allcs), df.shape[0]
