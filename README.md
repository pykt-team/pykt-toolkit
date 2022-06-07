# PyKY

PyKT is a python library build upon PyTorch to train deep learning based knowledge tracing models. The library consists of a standardized set of integrated data preprocessing procedures on 7 popular datasets across different domains, 5 detailed prediction scenarios, 10 frequently compared DLKT approaches for transparent and extensive experiments.


## Installation
Use the following command to install PyKY:

```
pip install -U pykt-toolkit
```

<!-- 
# How to use?

CUDA_VISIBLE_DEVICES=3 python wandb_akt_train.py

# description
## preprocess: 
The preprocess code for each dataseet.

* assist2015_preprocess.py

The preprocess code for assist2015 dataset.

If you want to add a new dataseet, please write your own dataset preprocess code, to change the data to this format:
```
    uid,seq_len
    questions ids / names
    concept ids / names
    timestamps
    usetimes
```
a example like this:
```
    50121,4
    106101,106102,106103,106104
    7014,7012,7014,7013
    0,1,1,1
    1647409594,1647409601,1647409666,1647409694
    123,234,456,789
```
* split_datasets.py

Split the data into 5-fold for trainning and testing. 

## data
The data saved dir for each dataset.

## datasets
Including a data_loader.py to prepare data for trainning models.

## models
Including models: dkt, dkt+, dkvmn, sakt, saint, akt, kqn, atkt.

## others
train.py: trainning code. -->
