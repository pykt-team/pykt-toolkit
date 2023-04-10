import pandas as pd
from .utils import sta_infos, write_txt, change2timestamp

KEYS = ["User", "Problem"]
def read_data_from_csv(read_file, write_file):
    stares = []
    df = pd.read_csv(read_file)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    df["index"] = range(df.shape[0])    
    df = df.dropna(subset=["User", "Problem", "Result", "Submit Time"])
    df = df[df["Result"].isin(['Accepted', 'Wrong Answer', 'Compile Error', 'Time Limit Exceeded', 
            'Memory Limit Exceeded', 'Runtime Error', 'Output Limit Exceeded',
            'Presentation Error', 'System Error', 'Validator Error'])]
            # 'Waiting' 'Running & Judging' 'Compiling''])]
    df.loc[:, "Result"] = df.loc[:, "Result"].apply(lambda k: "1" if k == "Accepted" else "0")

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    data = []
    ui_df = df.groupby(['User'], sort=False)

    for ui in ui_df:
        uid, curdf = ui[0], ui[1]
        curdf.loc[:, "Submit Time"] = curdf.loc[:, "Submit Time"].apply(lambda t: change2timestamp(t, False))
        curdf = curdf.sort_values(by=["Submit Time", "index"])

        # problem -> concept
        concepts = curdf["Problem"].astype(str)
        responses = curdf["Result"]
        timestamps = curdf["Submit Time"].astype(str)
        questions = ["NA"]
        usetimes = ["NA"]
        uids = [str(uid), str(len(responses))]
        data.append([uids, questions, concepts, responses, timestamps, usetimes])
        if len(data) % 1000 == 0:
            print(len(data))
    
    write_txt(write_file, data)

    print("\n".join(stares))

    return

