import pandas as pd
import numpy as np
import time
from datetime import datetime
from .utils import sta_infos, write_txt, format_list2str, replace_text

def change2timestamp(t):
    datetime_obj = datetime.strptime(t, "%Y/%m/%d %H:%M")
    timeStamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return timeStamp

KEYS = ["Anon Student Id", "KC"]
def read_data_from_csv(read_file, write_file):
    stares = []
    df = pd.read_csv(read_file)
    df['Problem Name'] = df['Problem Name'].apply(replace_text)
    df['Step Name'] = df['Step Name'].apply(replace_text)   
    df["KC"] = df.apply(lambda x:"{}----{}".format(x["Problem Name"],x["Step Name"]),axis=1)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df['tmp_index'] = range(len(df))
    df = df.dropna(subset=['Problem Name','Step Name','First Transaction Time','First Attempt'])
    df = df[df["First Attempt"]!="hint"]
    _df = df.copy()
    _df.loc[df["First Attempt"]=="correct","First Attempt"] = str(1)
    _df.loc[df["First Attempt"]=="incorrect","First Attempt"] = str(0)
    _df.loc[:, "First Transaction Time"] = _df.loc[:, "First Transaction Time"].apply(lambda t: change2timestamp(t))

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(_df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    user_inters = []
    for u, curdf in _df.groupby("Anon Student Id"):
        curdf = curdf.sort_values(by=["First Transaction Time", "tmp_index"])
        curdf["First Transaction Time"] = curdf["First Transaction Time"].astype(str)

        seq_skills = curdf["KC"].values
        seq_ans = curdf["First Attempt"].values
        seq_start_time = curdf["First Transaction Time"].values
        seq_len = len(seq_ans)                
        seq_problems = ["NA"]
        seq_use_time = ["NA"]

        assert seq_len == len(seq_skills) == len(seq_ans) ==  len(seq_start_time) 

        user_inters.append(
            [[u, str(seq_len)], seq_problems, seq_skills, format_list2str(seq_ans), seq_start_time, seq_use_time])

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

