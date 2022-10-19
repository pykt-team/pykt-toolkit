# Sample Codes of Global Knowledge Tracing Challenge @AAAI2023

This repository contains sample codes and guidelines for the [AAAI2023 Global Knowledge Tracing Challenge](http://ai4ed.cc/competitions/aaai2023competition). 

## Setup
### Download datasets

You can find the dataset download link [here](http://ai4ed.cc/competitions/aaai2023competition). 

### Install pyKT

[pyKT](https://pykt.org/) is a python library for knowledge tracing which contains more than 10 popular deep learning based knowledge tracing models. In the challenge, we provide detailed introductions to using pyKT. You can run more than 10 models using almost the same codes. 


First, you need to install pyKT.

1、Clone pyKT project
```shell
git clone https://github.com/pykt-team/pykt-toolkit/tree/main
```

2、Checkout out to the main branch
```shell
cd pykt-toolkit
git checkout main
```

3、Create python environment

```shell
conda create --name=pykt python=3.7.5
source activate pykt
```

4、Install python

For easier to modify codes, you can install pyKT in the editable mode:

```shell
pip install -e .
```



## Baselines
We provide three baselines, Majority model, DKT, and AKT. We suggest reading the Majority model first. 

### Majority model
The [sample_majority.ipynb](sample_majority.ipynb) contains a simple baseline. You can learn how to load datasets and generate the submission file. 

### Run pyKT
Before you run pyKT, you need to copy the downloaded files to pyKT workspace, `data/peiyou`.

Now, let us start running models.

### DKT
1、Train one model.


```shell
#in `examples` directory.
python wandb_dkt_train.py --use_wandb=0 --dataset_name=peiyou
```

Here is the training log

```shell
2022-10-18 15:34:52 - main - said: train model
ts.shape: (1023185,), ps.shape: (1023185,)
Epoch: 1, validauc: 0.8063, validacc: 0.8309, best epoch: 1, best auc: 0.8063, train loss: 0.4542354702468046, emb_type: qid, model: dkt, save_dir: saved_model/peiyou_dkt_qid_saved_model_42_0_0.2_200_0.001_0_1
            testauc: -1, testacc: -1, window_testauc: -1, window_testacc: -1
```

The trained model will save to the directory `save_dir`, which is printed in the training log.

2、Predict on the test dataset

```shell
#in `examples` directory.
python wandb_eval.py --use_wandb=0 --train_ratio=0.5 --test_filename=pykt_test.csv --save_dir="{save_dir}"
```
- use_wandb: 0 will send the result to wandb, 1 not send
- use_pred: 0 for `Non-Accumulative`, 1 for `Accumulative`
- train_ratio: 0.5 **don't change this value!**



For example:

```shell
python wandb_eval.py --use_wandb=0 --use_pred 0 --train_ratio=0.5 --test_filename=pykt_test.csv --save_dir="saved_model/peiyou_dkt_qid_saved_model_42_0_0.2_200_0.001_0_1"
```

After predicting, the output file is `{save_dir}/qid_test_ratio0.5_False_predictions.txt`(Non-Accumulative) or `{save_dir}/qid_test_ratio0.5_False_predictions.txt`(Accumulative)



3、Generate submission file

```
#in `examples/competitions/aaai2023_competition`
cd competitions/aaai2023_competition
python concert_pykt_to_submit.py --input_path {save_dir}/qid_test_ratio0.5_False_predictions.txt
```

**Note**: `input_path` is the absolute path.

For examples:

```
cd competitions/aaai2023_competition
python concert_pykt_to_submit.py --input_path /share/tabchen/tal_project/pykt-toolkit/examples/saved_model/peiyou_dkt_qid_saved_model_42_0_0.2_200_0.001_0_1/qid_test_ratio0.5_False_predictions.txt
```


### AKT

Similar to **DKT**, you only need to change `wandb_dkt_train.py` to `wandb_akt_train.py`.

### How to improve

- You can try other models, e.g. `dkvmn`、`sakt`、`gkt`. The supported models are listed [here](https://pykt-toolkit.readthedocs.io/en/latest/models.html).
- Try different hyperparameters. See `wandb_{model_name}_train.py` to see how to pass hyperparameters. And [here](https://github.com/pykt-team/pykt-toolkit/tree/main/examples/seedwandb) are hyperparameters used in the paper: pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models.
- Modify(add) models in pyKT, you can read  [add your models](https://pykt-toolkit.readthedocs.io/en/latest/contribute.html#add-your-models) to learn how to add new models to pyKT.
- ……



More details about [pyKT](https://pykt.org/) can see our [docs](https://pykt-toolkit.readthedocs.io/en/latest/quick_start.html).



## Submission format

The  format of submission is following:

```
responses
"0.8782808902532617,0.8782808902532617,0.8852459016393442,0.7898353843695062,0.8852459016393442,0.6502890173410405,0.5933641975308642,0.6192307692307693"
"0.8782808902532617,0.8782808902532617,0.8852459016393442,0.7898353843695062,0.8852459016393442,0.6502890173410405,0.5933641975308642,0.6192307692307693"
"0.8782808902532617,0.8782808902532617,0.8852459016393442,0.7898353843695062,0.8852459016393442,0.6502890173410405,0.5933641975308642,0.6192307692307693"
```

**Note**:

- You should name the file `prediction.csv` and zip this file to `{any name you like}.zip`. 
- Please use `double quotation marks` to enclose one line.