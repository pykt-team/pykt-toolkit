import pandas as pd
from .utils import sta_infos, write_txt, replace_text

def load_q2c(qname):
    df = pd.read_csv(qname, encoding = "utf-8",low_memory=False).dropna(subset=["name", "topic"])
    dq2c = dict()
    for name, topic in zip(df["name"], df["topic"]):
        if name not in dq2c:
            dq2c[name] = topic
        else:
            print(f"already has topic in dict: {name}: {topic}, {dq2c[name]}")
    print(f"dq2c: {len(dq2c)}")
    return dq2c

KEYS = ["user_id", "topic", "exercise"]
def read_data_from_csv(read_file, write_file, dq2c):
    stares = []

    df = pd.read_csv(read_file)
    df["topic"] = df["exercise"].apply(lambda q: "NANA" if q not in dq2c else dq2c[q])
    df["exercise"] = df["exercise"].apply(replace_text)
    df["topic"] = df["topic"].apply(replace_text)
    df = df[df["topic"] != "NANA"]

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    df["index"] = range(df.shape[0])
    print(f"original recores shape: {df.shape}")

    usedf = df[["index", "user_id", "exercise", "time_done", "time_taken_attempts", "correct", "count_attempts", "topic"]]
    usedf = usedf.dropna(subset=["user_id", "exercise", "time_done", "correct"])
    usedf = usedf[usedf["correct"].isin([False, True])]
    usedf["time_taken_attempts"] = (usedf["time_taken_attempts"].fillna(-100)).astype(str) # only hint! False correct
    usedf.loc[:, "time_taken_attempts"] = usedf["time_taken_attempts"].astype(str).apply(lambda x: int(x.split("&")[0])*1000).astype(str)
    
    usedf.loc[:, "time_done"] = usedf["time_done"].astype(int)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(usedf, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    data = []
    uids = usedf.user_id.unique()
    problems = usedf.exercise.unique()
    print(f"usedf: {usedf.shape}, uids: {len(uids)}, problems: {len(problems)}")

    ui_df = usedf.groupby('user_id', sort=False)

    for ui in ui_df:
        uid, curdf = ui[0], ui[1]
        curdf = curdf.sort_values(by=["time_done", "index"])

        curdf["time_done"] = curdf["time_done"].apply(lambda x: round(int(x) / 1000)).astype(str) # ms
        questions = curdf["exercise"].tolist()
        concepts = curdf["topic"].tolist()
        rs = curdf["correct"].astype(int).astype(str).tolist()
        ts = curdf["time_done"].tolist()
        uts = curdf["time_taken_attempts"].tolist()
        seq_len = len(rs)
        uc = [str(uid), str(seq_len)]
        data.append([uc, questions, concepts, rs, ts, uts])
        if len(data) % 1000 == 0:
            print(len(data))
    write_txt(write_file, data)

    print("\n".join(stares))

    return