import sys
# sys.path.append("..")
from pykt.preprocess.split_datasets import read_data

cols = ['concepts', 'timestamps', 'usetimes', 'questions', 'responses', 'uid']

def check_result(path1,path2,sort=True):
    #sort
    df_1 = read_data(path1,min_seq_len=-1)[0]
    check_cols = [x for x in cols if x in df_1.columns]
    if sort:
        df_1 = df_1.sort_values('uid')

    df_2 = read_data(path2,min_seq_len=-1)[0]
    if sort:
        df_2 = df_2.sort_values('uid')
        
    for col in check_cols:
        print(col)
        print((df_1[col].values==df_2[col].values).mean())
        

if __name__ =="__main__":
    dataset_str = "assist2009 algebra2005 nips_task34 statics2011 assist2015 poj bridge2algebra2006"
    for dataset in dataset_str.split():
        dataset = dataset.strip()
        print('+',"-"*40,dataset,"-"*40,'+')
        path1 = f'/share/tabchen/tal_project/pykt-toolkit/data/{dataset}/data.txt'
        path2 = f'/share/tabchen/tal_project/pykt-toolkit/data_old/{dataset}/data.txt'
        check_result(path1,path2,sort=False)