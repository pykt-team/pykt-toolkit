# PyKY

PyKT is a python library build upon PyTorch to train deep learning based knowledge tracing models. The library consists of a standardized set of integrated data preprocessing procedures on 7 popular datasets across different domains, 5 detailed prediction scenarios, 10 frequently compared DLKT approaches for transparent and extensive experiments.


## Installation
Use the following command to install PyKY:

```
pip install -U pykt-toolkit -i  https://pypi.python.org/simple 
```

## Reference
### Projects

1. https://github.com/hcnoh/knowledge-tracing-collection-pytorch 
2. https://github.com/arshadshk/SAKT-pytorch 
3. https://github.com/shalini1194/SAKT 
4. https://github.com/arshadshk/SAINT-pytorch 
5. https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing- 
6. https://github.com/arghosh/AKT 
7. https://github.com/JSLBen/Knowledge-Query-Network-for-Knowledge-Tracing 
8. https://github.com/xiaopengguo/ATKT 
9. https://github.com/jhljx/GKT 

### Papers

1. DKT: Deep knowledge tracing 
2. DKT+: Addressing two problems in deep knowledge tracing via prediction-consistent regularization 
3. DKT-Forget: Augmenting knowledge tracing by considering forgetting behavior 
4. KQN: Knowledge query network for knowledge tracing: How knowledge interacts with skills 
5. DKVMN: Dynamic key-value memory networks for knowledge tracing 
6. ATKT: Enhancing Knowledge Tracing via Adversarial Training 
7. GKT: Graph-based knowledge tracing: modeling student proficiency using graph neural network 
8. SAKT: A self-attentive model for knowledge tracing 
9. SAINT: Towards an appropriate query, key, and value computation for knowledge tracing 
10. AKT: Context-aware attentive knowledge tracing 



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
