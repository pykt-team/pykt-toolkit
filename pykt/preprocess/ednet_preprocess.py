import pandas as pd
import random
import os
from .utils import sta_infos, write_txt
from tqdm import tqdm

KEYS = ["user_id", "tags", "question_id"]


def read_data_from_csv(read_file, write_file,dataset_name=None):
    if not dataset_name is None:
        write_file = write_file.replace("/ednet/", f"/{dataset_name}/")
        write_dir = read_file.replace("/ednet/", f"/{dataset_name}")
        print(f"write_dir is {write_dir}")
        print(f"write_file is {write_file}")
    stares = []

    file_list = list()

    random.seed(2)
    samp = [i for i in range(840473)]
    random.shuffle(samp)

    count = 0

    for unum in tqdm(samp):
        str_unum = str(unum)
        df_path = os.path.join(read_file, f"KT1/u{str_unum}.csv")
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            df['user_id'] = unum

            file_list.append(df)
            count = count + 1

        if dataset_name == "ednet" and count == 5000:
            start_i = 0
            break
        elif dataset_name == "ednet5w" and count == 50000+5000:
            start_i = 5000
            break
        
    print(f"total user num: {count}")
    all_sa = pd.concat(file_list[start_i:])
    print(f"after sub all_sa: {len(all_sa)}")
    all_sa["index"] = range(all_sa.shape[0])
    ca = pd.read_csv(os.path.join(read_file, 'contents', 'questions.csv'))
    
    if not dataset_name is None:
        read_file = write_dir 

    all_sa.to_csv(os.path.join(read_file, 'ednet_sample.csv'), index=False)
    ca['tags'] = ca['tags'].apply(lambda x:x.replace(";","_"))
    ca = ca[ca['tags']!='-1']
    co = all_sa.merge(ca, sort=False,how='left')
    co = co.dropna(subset=["user_id", "question_id", "elapsed_time", "timestamp", "tags", "user_answer"])
    co['correct'] = (co['correct_answer']==co['user_answer']).apply(int)


    ins, us, qs, cs, avgins, avgcq, na = sta_infos(co, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")


    ins, us, qs, cs, avgins, avgcq, na = sta_infos(co, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    co.to_csv(os.path.join(read_file, 'ednet_sample_process.csv'), index=False)
    
    ui_df = co.groupby(['user_id'], sort=False)

    user_inters = []
    for ui in tqdm(ui_df):
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["timestamp", "index"])
        seq_len = len(tmp_inter)
        seq_skills = tmp_inter['tags'].astype(str)
        seq_ans = tmp_inter['correct'].astype(str)
        seq_problems = tmp_inter['question_id'].astype(str)
        seq_start_time = tmp_inter['timestamp'].astype(str)
        seq_response_cost = tmp_inter['elapsed_time'].astype(str)

        assert seq_len == len(seq_problems) == len(seq_ans)

        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost])

    write_txt(write_file, user_inters)
    print("\n".join(stares))
    return write_dir, write_file


