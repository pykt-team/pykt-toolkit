import pandas as pd
from .utils import sta_infos, write_txt, format_list2str, change2timestamp, replace_text
import json
import os

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

def load_qinfo(fname):
    dqinfo = dict()
    with open(fname, "r") as fin:
        dqinfo = json.load(fin)

    return dqinfo