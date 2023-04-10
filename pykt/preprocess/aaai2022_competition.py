import pandas as pd
from .utils import sta_infos, write_txt, format_list2str, change2timestamp, replace_text
import json

KEYS = ["stu_id", "concept_id", "que_id"]
def read_data_from_csv(read_file, write_file, dq2c):
    stares = []
    df = pd.read_csv(read_file, low_memory=False)
    # 合并知识点信息
    cs = []
    for i, row in df.iterrows():
        qid = str(row["que_id"])
        cid = dq2c[qid]
        cs.append(cid)
    df["concept_id"] = cs

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = range(df.shape[0])

    df = df.dropna(subset=["stu_id", "timestamp", "que_id", "label"])
    df = df[df['label'].isin([0,1])] #filter responses
    df['label'] = df['label'].astype(int)
    


    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    ui_df = df.groupby(['stu_id'], sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["timestamp", "index"])
        seq_len = len(tmp_inter)
        seq_skills = tmp_inter['concept_id'].astype(str)
        seq_ans = tmp_inter['label'].astype(str)
        seq_problems = tmp_inter['que_id'].astype(str)
        seq_start_time = tmp_inter['timestamp'].astype(str)
        seq_response_cost = ["NA"]

        assert seq_len == len(seq_skills) == len(seq_ans)

        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost])

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

def load_q2c(fname):
    dq2c = dict()
    with open(fname, "r") as fin:
        obj = json.load(fin)
        for qid in obj:
            cur = obj[qid]
            content = cur["content"]
            concept_routes = cur["concept_routes"]
            analysis = cur["analysis"]
            cs = []
            for route in concept_routes:
                tailc = route.split("----")[-1]
                if tailc not in cs:
                    cs.append(tailc)
            dq2c[qid] = "_".join(cs)
    return dq2c

# group_name = "students_b_200"
# tof = os.path.join("data/aaai2022_raw/", group_name+"_data.txt")
# read_data_from_csv(dfs[group_name], tof)


# import os, sys
# import json
# import pandas as pd

# def load_q2c(fname):
#     dq2c = dict()
#     with open(fname, "r") as fin:
#         obj = json.load(fin)
#         for qid in obj:
#             cur = obj[qid]
#             content = cur["content"]
#             concept_routes = cur["concept_routes"]
#             analysis = cur["analysis"]
#             cs = []
#             for route in concept_routes:
#                 tailc = route.split("----")[-1]
#                 if tailc not in cs:
#                     cs.append(tailc)
#             dq2c[qid] = "_".join(cs)
#     return dq2c

# def merge_cs(fname, dq2c):
#     usecols = ["fold", "uid", "questions", "responses", "timestamps"]
#     df = pd.read_csv(fname, dtype="str")
#     if "responses" not in list(df.columns):
#         add_rs(df)
#     df = df[usecols]
#     allcs = []
#     for i, row in df.iterrows():
#         qs = row["questions"].split(",")
#         cs = []
#         for q in qs:
#             cs.append(dq2c[q])
#         cs = ",".join(cs)
#         allcs.append(cs)
#     df["concepts"] = allcs
#     return df

# def merge_testfile(earlyf, latef):
#     df1 = earlyf
#     df2 = latef
#     print(f"df2 columns: {df2.columns}")
#     dres = dict()
#     dres["fold"] = str(df1["fold"])
#     dres["uid"] = str(df1["uid"])
#     for key in ["questions", "concepts", "responses", "timestamps"]:
#         dres[key] = df1[key]+","+df2[key]
#     mergedf = pd.DataFrame(dres)
#     return mergedf

# def add_rs(df2):
#     rss = []
#     for i, row in df2.iterrows():
#         qs = row["questions"]
#         rs = ",".join([str(-1)]*len(qs))
#         rss.append(rs)
#     df2["responses"] = rss
#     return df2
    


# dname = "../../data/peiyou.ori/"
# q2cfile = "../../data/peiyou.ori/map_questions.json"
# dq2c = load_q2c(q2cfile)

# train_valid_fname = "../../data/peiyou.ori/map_train_valid.csv"
# merged_train_valid = merge_cs(train_valid_fname, dq2c)
# merged_train_valid.to_csv("../../data/peiyou.ori/merged_map_train_valid.csv")

# earlyf = "../../data/peiyou.ori/map_test_early_with_label.csv"
# latef = "../../data/peiyou.ori/map_test_late.csv"
# merge_early = merge_cs(earlyf, dq2c)
# merge_late = merge_cs(latef, dq2c)
# testdf = merge_testfile(merge_early, merge_late)

# total_df = pd.concat([merged_train_valid, testdf])
# print(f"merged_train_valid: {merged_train_valid.shape}, testdf: {testdf.shape}, total_df: {total_df.shape}")

# from split_datasets import get_max_concepts, calStatistics, extend_multi_concepts, id_mapping, generate_sequences, generate_window_sequences, generate_question_sequences, write_config, getq2c, save_id2idx, ALL_KEYS, ONE_KEYS, get_inter_qidx

# min_seq_len = 3
# maxlen = 200
# configf = "../../configs/data_config.json"
# dataset_name = "peiyou"
# # main
# stares = []

# effective_keys = ["uid", "questions", "concepts", "responses", "timestamps"]
# effective_keys = set(effective_keys)
# #cal max_concepts
# if 'concepts' in effective_keys:
#     max_concepts = get_max_concepts(total_df)
# else:
#     max_concepts = -1

# oris, _, qs, cs, seqnum = calStatistics(total_df, stares, "original")
# print("="*20)
# print(f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

# total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)
# total_df, dkeyid2idx = id_mapping(total_df)
# dkeyid2idx["max_concepts"] = max_concepts

# dq2c, _ = getq2c(total_df)
# pd.to_pickle(dq2c, os.path.join(dname, "dq2c.pkl"))

# extends, _, qs, cs, seqnum = calStatistics(total_df, stares, "extend multi")
# print("="*20)
# print(f"after extend multi, total interactions: {extends}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

# save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))
# effective_keys.add("fold")
# config = []
# for key in ALL_KEYS:
#     if key in effective_keys:
#         config.append(key)
# # train test split & generate sequences
# ##########
# #splitdf, test_df = merged_train_valid, testdf #train_test_split(total_df, 0.2)
# ###TODO
# trainlen, testlen = merged_train_valid.shape[0], testdf.shape[0]
# splitdf, test_df = total_df[0:trainlen], total_df[trainlen:]
# # splitdf = KFold_split(train_df, kfold)
# # splitdf[config].to_csv(os.path.join(dname, "train_valid.csv"), index=None)
# ins, ss, qs, cs, seqnum = calStatistics(splitdf, stares, "original train+valid")
# print(f"train+valid original interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
# split_seqs = generate_sequences(splitdf, effective_keys, min_seq_len, maxlen)
# ins, ss, qs, cs, seqnum = calStatistics(split_seqs, stares, "train+valid sequences")
# print(f"train+valid sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
# split_seqs.to_csv(os.path.join(dname, "train_valid_sequences.csv"), index=None)
# # print(f"split seqs dtypes: {split_seqs.dtypes}")

# # add default fold -1 to test!
# # test_df["fold"] = [-1] * test_df.shape[0]  
# test_df['cidxs'] = get_inter_qidx(test_df)#add index  
# test_seqs = generate_sequences(test_df, list(effective_keys) + ['cidxs'], min_seq_len, maxlen)
# ins, ss, qs, cs, seqnum = calStatistics(test_df, stares, "test original")
# print(f"original test interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
# ins, ss, qs, cs, seqnum = calStatistics(test_seqs, stares, "test sequences")
# print(f"test sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
# print("="*20)

# test_window_seqs = generate_window_sequences(test_df, list(effective_keys) + ['cidxs'], maxlen)
# flag, test_question_seqs = generate_question_sequences(test_df, effective_keys, False, min_seq_len, maxlen)
# flag, test_question_window_seqs = generate_question_sequences(test_df, effective_keys, True, min_seq_len, maxlen)

# test_df = test_df[config+['cidxs']]

# test_df.to_csv(os.path.join(dname, "test.csv"), index=None)
# test_seqs.to_csv(os.path.join(dname, "test_sequences.csv"), index=None)
# test_window_seqs.to_csv(os.path.join(dname, "test_window_sequences.csv"), index=None)

# ins, ss, qs, cs, seqnum = calStatistics(test_window_seqs, stares, "test window")
# print(f"test window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

# if flag:
#     test_question_seqs.to_csv(os.path.join(dname, "test_question_sequences.csv"), index=None)
#     test_question_window_seqs.to_csv(os.path.join(dname, "test_question_window_sequences.csv"), index=None)
    
#     ins, ss, qs, cs, seqnum = calStatistics(test_question_seqs, stares, "test question")
#     print(f"test question interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
#     ins, ss, qs, cs, seqnum = calStatistics(test_question_window_seqs, stares, "test question window")
#     print(f"test question window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

# kfold = 5
# write_config(dataset_name=dataset_name, dkeyid2idx=dkeyid2idx, effective_keys=effective_keys, 
#             configf=configf, dpath = dname, k=kfold,min_seq_len = min_seq_len, maxlen=maxlen,flag=flag)

# print("="*20)
# print("\n".join(stares))

