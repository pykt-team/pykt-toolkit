#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
from .utils import sta_infos, write_txt, change2timestamp, replace_text

KEYS = ["studentId", "skill", "problemId"]

def read_data_from_csv(read_file, write_file):
    stares = []

    df = pd.read_csv(read_file, dtype=str, low_memory=False, usecols=['startTime', 'timeTaken', 'studentId', 'skill', 'problemId', 'correct'])

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = df.index
    df = df.dropna(subset=['skill', 'problemId'])
    #filter invalid record
    df = df[df["correct"].isin([str(0),str(1)])]

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares, "~~")
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    user_inters = []
    for user, group in df.groupby("studentId", sort=False):
        group = group.sort_values(by=["startTime", "index"], ascending=True)
        group["startTime"] = group["startTime"].astype(str)

        seq_problems = group["problemId"].tolist()
        seq_ans = group["correct"].tolist()
        seq_start_time = group["startTime"].tolist()
        seq_skills = group["skill"].tolist()
        seq_len = len(seq_ans)
        seq_use_time = group["timeTaken"].tolist()

        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_start_time)
        
        user_inters.append(
            [[user, str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_use_time])

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return