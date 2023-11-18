import pandas as pd
from .utils import sta_infos, write_txt, change2timestamp

KEYS = ["user", "place_asked", "questions"]
def read_data_from_csv(read_file, write_file):
    stares = []
    df = pd.read_csv(read_file, sep=";", low_memory=False)
    df["questions"] = df.apply(lambda x:f"{x['place_asked']}----{x['type']}",axis=1)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = range(df.shape[0])

    df = df.dropna(subset=["user", "place_asked", "inserted"])
    df["place_answered"] = (df["place_answered"].fillna(-100000)).astype(int)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    data = []
    ui_df = df.groupby('user', sort=False)

    for ui in ui_df:
        uid, curdf = ui[0], ui[1]
        curdf.loc[:, "inserted"] = curdf.loc[:, "inserted"].apply(lambda t: change2timestamp(t, False))
        curdf = curdf.sort_values(by=["inserted", "index"])

        curdf["inserted"] = curdf["inserted"].astype(str)
        concepts = curdf["place_asked"].astype(str)
        responses = (curdf["place_asked"] == curdf["place_answered"]).astype(int).astype(str)
        timestamps = curdf["inserted"]
        usetimes = curdf["response_time"].astype(str)
        seq_len = len(responses)
        # print(f"uid: {uid}, seq_len: {seq_len}")
        uids = [str(uid), str(seq_len)]
        questions = curdf["questions"].astype(str)
        data.append([uids, questions, concepts, responses, timestamps, usetimes])
        if len(data) % 1000 == 0:
            print(len(data))
    write_txt(write_file, data)

    print("\n".join(stares))
    
    return

