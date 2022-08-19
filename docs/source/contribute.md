# How to contribute to Pykt?
Everyone is welcome to contribute, and we value everybody's contribution.


## You can contribute in so many ways!
There are some ways you can contribute to PyKT:
1. Find bugs and create an issue.
2. Add new datasets.
3. Implementing new models.

## Install for Development
1、Clone pykt repositoriy

```shell
git clone https://github.com/pykt-team/pykt-toolkit
```

2、Change to dev branch 

```shell
cd pykt-toolkit
git checkout dev
```

**Do not** work on the main branch.

3、Editable install

You can use the following command to install the pykt library. 

```shell
pip install -e .
```
In this mode, every modification in `pykt` directory will take effect immediately. You do not need to reinstall the package again. 

4、Push to remote(dev)

After development models or fix bugs, you can push your codes to dev branch. 


The main branch is **not allowed** to push codes (the push will be failed). You can use a Pull Request to merge your code from **dev** branch to the main branch. We will reject the Pull Request from another branch to main branch, you can merge to dev branch first.



## Add Your Datasets

In this section, we will use the `ASSISTments2015` dataset to show the add dataset procedure. Here we use `assist2015` as the dataset name, you can change `assist2015` to your dataset name.

### Add Data Files
1、Add a new dataset folder in the `data` directory with the name of the dataset. 

2、Then, you can store the raw files in this directory. Here is the `assist2015` file structure:

```shell
$tree data/assist2015/
├── 2015_100_skill_builders_main_problems.csv
```

3、Then add the data path to `dname2paths` of `examples/data_preprocess.py`.

![](../pics/dataset-add_data_path.jpg)

### Write Python Script

1、Create the processing script `assist2015_preprocess.py` in `pykt/preprocess` directory. Before write the preprocess python scipt you are suggestd to  read the [Data Preprocess Standards](#Data Preprocess Standards), which contains some guidlines to process dataset. Here is the scipt for `assist2015` we show the main steps, full codes can see in `pykt/preprocess/algebra2005_preprocess.py`.

<!-- 
```python
import pandas as pd
from pykt.utils import write_txt, change2timestamp, replace_text

def read_data_from_csv(read_file, write_file):
    # load the original data
    df = pd.read_table(read_file, encoding = "utf-8", dtype=str, low_memory=False)
    df["Problem Name"] = df["Problem Name"].apply(replace_text)
    df["Step Name"] = df["Step Name"].apply(replace_text)
    df["Questions"] = df.apply(lambda x:f"{x['Problem Name']}----{x['Step Name']}",axis=1)
    

    df["index"] = range(df.shape[0])
    df = df.dropna(subset=["Anon Student Id", "Questions", "KC(Default)", "First Transaction Time", "Correct First Attempt"])
    df = df[df["Correct First Attempt"].isin([str(0),str(1)])]#keep the interaction which response in [0,1]
    df = df[["index", "Anon Student Id", "Questions", "KC(Default)", "First Transaction Time", "Correct First Attempt"]]
    df["KC(Default)"] = df["KC(Default)"].apply(replace_text)

    data = []
    ui_df = df.groupby(['Anon Student Id'], sort=False)

    for ui in ui_df:
        u, curdf = ui[0], ui[1]
        curdf.loc[:, "First Transaction Time"] = curdf.loc[:, "First Transaction Time"].apply(lambda t: change2timestamp(t))
        curdf = curdf.sort_values(by=["First Transaction Time", "index"])
        curdf["First Transaction Time"] = curdf["First Transaction Time"].astype(str)

        seq_skills = [x.replace("~~", "_") for x in curdf["KC(Default)"].values]
        seq_ans = curdf["Correct First Attempt"].values
        seq_start_time = curdf["First Transaction Time"].values
        seq_problems = curdf["Questions"].values
        seq_len = len(seq_ans)
        seq_use_time = ["NA"]
        
        data.append(
            [[u, str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_use_time])

    write_txt(write_file, data)
``` -->

2、Import the preprocess file in `pykt/preprocess/data_proprocess.py`.


![](../pics/dataset-import.jpg)



### Data Preprocess Standards
#### Field Extraction

For any data set, we mainly extract 6 fields: user ID, question ID (name), skill ID (name), answering status, answer submission time, and answering time (if the field does not exist in the dataset, it is represented by NA) .

#### Data Filtering

For each answer record, if any of the five fields of user ID, question ID (name), skill ID (name), answer status, and answer submission time are empty, the answer record will be deleted.

#### Data Sorting

Each student's answer sequence is sorted according to the answer order of the students. If different answer records of the same student appear in the same order, the original order is maintained, that is, the order of the answer records in the original data set is kept consistent.

#### Character Process

- **Field concatenation:** Use `----` as the connecting symbol. For example, Algebra2005 needs to concatenate `Problem Name` and `Step Name` as the final problem name.
- **Character replacement:** If there is an underline `_` in the question and skill of original data, replace it with `####`. If there is a comma `,` in the question and skill of original data, replace it with `@@@@`.
- **Multi-skill separator:** If there are multiple skills in a question, we separate the skills with an underline `_`.
- **Time format:** The answer submission time is a millisecond (ms) timestamp, and the answer time is in milliseconds (ms).

#### Output data format

After completing the above data preprocessing, each dataset will generate a data.txt file in the folder named after it (data directory). Each student sequence contains 6 rows of data as follows:

```
User ID, sequence length
Question ID (name)
skill ID (name)
Answer status
Answer submission time
time to answer
```

Example:

```
50121, 4 
106101, 106102, 106103, 106104 
7014, 7012, 7014, 7013 
0, 1, 1, 1 
1647409594000, 1647409601000, 1647409666000, 1647409694000 
123, 234, 456, 789 
```


<!-- ## Add Your Models(todo) -->