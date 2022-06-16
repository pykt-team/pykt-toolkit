import pandas as pd

def sta_infos(df, keys, stares, split_str="_"):
    # keys: 0: uid , 1: concept, 2: question
    uids = df[keys[0]].unique()
    if len(keys) == 2:
        cids = df[keys[1]].unique()
    elif len(keys) > 2:
        qids = df[keys[2]].unique()
        ctotal = 0
        cq = df.drop_duplicates([keys[2], keys[1]])[[keys[2], keys[1]]]
        cq[keys[1]] = cq[keys[1]].fillna("NANA")
        cids, dq2c = set(), dict()
        for i, row in cq.iterrows():
            q = row[keys[2]]
            ks = row[keys[1]]
            dq2c.setdefault(q, set())
            if ks == "NANA":
                continue
            for k in str(ks).split(split_str):
                dq2c[q].add(k)
                cids.add(k)
        ctotal, na, qtotal = 0, 0, 0
        for q in dq2c:
            if len(dq2c[q]) == 0:
                na += 1 # questions has no concept
                continue
            qtotal += 1
            ctotal += len(dq2c[q])
        
        avgcq = round(ctotal / qtotal, 4)
    avgins = round(df.shape[0] / len(uids), 4)
    ins, us, qs, cs = df.shape[0], len(uids), "NA", len(cids)
    avgcqf, naf = "NA", "NA"
    if len(keys) > 2:
        qs, avgcqf, naf = len(qids), avgcq, na
    curr = [ins, us, qs, cs, avgins, avgcqf, naf]
    stares.append(",".join([str(s) for s in curr]))
    return ins, us, qs, cs, avgins, avgcqf, naf

def write_txt(file, data):
    with open(file, "w") as f:
        for dd in data:
            for d in dd:
                f.write(",".join(d) + "\n")

from datetime import datetime
def change2timestamp(t, hasf=True):
    if hasf:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
    else:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
    return int(timeStamp)

def replace_text(text):
    text = text.replace("_", "####").replace(",", "@@@@")
    return text


def format_list2str(input_list):
    return [str(x) for x in input_list]


def one_row_concept_to_question(row):
    """Convert one row from concept to question

    Args:
        row (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_question = []
    new_concept = []
    new_response = []

    tmp_concept = []
    begin = True
    for q, c, r, mask, is_repeat in zip(row['questions'].split(","),
                                        row['concepts'].split(","),
                                        row['responses'].split(","),
                                        row['selectmasks'].split(","),
                                        row['is_repeat'].split(","),
                                        ):
        if begin:
            is_repeat = "0"
            begin = False
        if mask == '-1':
            break
        if is_repeat == "0":
            if len(tmp_concept) != 0:
                new_concept.append("_".join(tmp_concept))
                tmp_concept = []
            new_question.append(q)
            new_response.append(r)
            tmp_concept = [c]
        else:#如果是 1 就累计知识点
            tmp_concept.append(c)
    if len(tmp_concept) != 0:
        new_concept.append("_".join(tmp_concept))

    if len(new_question) < 200:
        pads = ['-1'] * (200 - len(new_question))
        new_question += pads
        new_concept += pads
        new_response += pads

    new_selectmask = ['1']*len(new_question)
    new_is_repeat = ['0']*len(new_question)

    new_row = {"fold": row['fold'],
               "uid": row['uid'],
               "questions": ','.join(new_question),
               "concepts": ','.join(new_concept),
               "responses": ','.join(new_response),
               "selectmasks": ','.join(new_selectmask),
               "is_repeat": ','.join(new_is_repeat),
               }
    return new_row

def concept_to_question(df):
    """Convert df from concept to question
    Args:
        df (_type_): df contains concept

    Returns:
        _type_: df contains question
    """
    new_row_list = list(df.apply(one_row_concept_to_question,axis=1).values)
    df_new = pd.DataFrame(new_row_list)
    return df_new

def get_df_from_row(row):
    value_dict = {}
    for col in ['questions', 'concepts', 'responses', 'is_repeat']:
        value_dict[col] = row[col].split(",")
    df_value = pd.DataFrame(value_dict)
    df_value = df_value[df_value['questions']!='-1']
    return df_value