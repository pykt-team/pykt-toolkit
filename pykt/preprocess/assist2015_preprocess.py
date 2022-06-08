import pandas as pd
from .utils import sta_infos, write_txt

KEYS = ["user_id", "sequence_id"]

def read_data_from_csv(read_file, write_file):
    stares = []

    df = pd.read_csv(read_file)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = range(df.shape[0])

    df = df.dropna(subset=["user_id", "log_id", "sequence_id", "correct"])
    df = df[df['correct'].isin([0,1])]#filter responses
    df['correct'] = df['correct'].astype(int)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    ui_df = df.groupby(['user_id'], sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["log_id", "index"])
        seq_len = len(tmp_inter)
        seq_skills = tmp_inter['sequence_id'].astype(str)
        seq_ans = tmp_inter['correct'].astype(str)
        seq_problems = ["NA"]
        seq_start_time = ["NA"]
        seq_response_cost = ["NA"]

        assert seq_len == len(seq_skills) == len(seq_ans)

        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost])

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

