#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
from .utils import sta_infos,skill_difficult,question_difficult, write_txt, format_list2str, change2timestamp, replace_text

KEYS = ["Anon Student Id", "KC(SubSkills)", "Questions"]

def read_data_from_csv(read_file, write_file):
    stares = []

    df = pd.read_table(read_file, low_memory=False)
    #concat problem name & step_name
    df["Problem Name"] = df["Problem Name"].apply(replace_text)
    df["Step Name"] = df["Step Name"].apply(replace_text)
    df["Questions"] = df.apply(lambda x:f"{x['Problem Name']}----{x['Step Name']}",axis=1)
    
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares, '~~')
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = df.index
    df = df.dropna(subset=["Anon Student Id", "KC(SubSkills)", "Questions", "Correct First Attempt", "First Transaction Time"])
    #filter invalid record
    df = df[df["Correct First Attempt"].isin([0,1])]
    df['Correct First Attempt'] = df['Correct First Attempt'].apply(int)
    df.loc[:, "First Transaction Time"] = df.loc[:, "First Transaction Time"].apply(lambda t: change2timestamp(t))
    df["KC(SubSkills)"] = df["KC(SubSkills)"].apply(replace_text)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares, "~~")
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df.rename(columns={'KC(SubSkills)':'skill_id'},inplace=True)
    df.rename(columns={'Questions':'problem_id'},inplace=True)

    df = skill_difficult(df,'skill_id','Correct First Attempt')
    df = question_difficult(df,'problem_id','Correct First Attempt')
    
    df.rename(columns={'skill_id':'KC(SubSkills)'},inplace=True)
    df.rename(columns={'problem_id':'Questions'},inplace=True)
    
    user_inters = []
    for user, group in df.groupby("Anon Student Id", sort=False):
        group = group.sort_values(by=["First Transaction Time", "index"], ascending=True)
        group["First Transaction Time"] = group["First Transaction Time"].astype(str)

        seq_problems = group["Questions"].tolist()
        seq_ans = group["Correct First Attempt"].tolist()
        seq_start_time = group["First Transaction Time"].tolist()
        seq_skills = [x.replace("~~","_") for x in group["KC(SubSkills)"].tolist()]
        seq_len = len(seq_ans)
        seq_use_time = ["NA"]
        seq_skill_difficult = group['skill_difficult'].tolist()
        seq_question_difficult = group['question_difficult'].tolist()
        

        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_start_time)
        
        user_inters.append(
            [[user, str(seq_len)], 
            seq_problems, 
            seq_skills, 
            format_list2str(seq_ans), 
            seq_start_time, 
            seq_use_time,
            format_list2str(seq_skill_difficult),
            format_list2str(seq_question_difficult)])

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

