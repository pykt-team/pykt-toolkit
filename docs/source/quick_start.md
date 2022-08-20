# Welcome to pyKT

## Installation
You can specify to install it through `pip`.

```shell
pip install -U pykt-toolkit
```

We recommend creating a new Conda environment using the following command.

```shell
conda create --name=pykt python=3.7.5
source activate pykt
pip install -U pykt-toolkit
```

## Train Your First Model
### Prepare a Dataset
**1、Download a Dataset**

You can find the download link for a dataset from [here](datasets.md). Download the dataset to the `data/{dataset_name}` folder.

**2、Preprocess the Dataset**

`python data_preprocess.py [parameter]`

```shell
Args:
    --dataset_name: dataset name, default=“assist2015”
    --min_seq_len: minimum sequence length, default=3
    --maxlen: maximum sequence length, default=200
    --kfold: divided folds, default=5
```

Example:

```shell
cd examples
python data_preprocess.py --dataset_name=ednet
```

### Training a Model
After processed the dataset, you can use the `python wandb_modelname_train.py [parameter]` to train a model:

```shell
CUDA_VISIBLE_DEVICES=2 nohup python wandb_sakt_train.py --dataset_name=assist2015 --use_wandb=0 --add_uuid=0 --num_attn_heads=2 > sakt_train.txt &
```

Run the `get_wandb_new` file. If the model has selected more than 300 sets of parameters, and the most recent 100 sets of parameters have not achieved optimal results on the test set (that is, when end! is output), stop.

## Evaluating Your Model

After train you model, you can use `wandb_predict.py` to evaluate the trained model's performance in the datasets.

`python wandb_predict.py`

```shell
Args:
    --bz: batch_size, default is 256
    --save_dir: the dictory of the trained model, default is "saved_model"
    --fusion_type: the fusion mode,default is "late_fusion"
    --use_wandb: use wandb or not, default is 1
```

 
## Hyperparameter Tuning

### Create a Wandb Account

Weights & Biases (Wandb) is the machine learning platform for developers to build better models faster. Fisrly, you should register an account in [Wandb](https://wandb.ai/) webpage, hhen you can get the API key from [here](https://wandb.ai/settings):

![](../pics/api_key.png)


Final, add your `uid` and `api_key` in `configs/wandb.json`.

### Write a Sweep Config

`python generate_wandb.py [parameter]`

```shell
Args:
       --src_dir: The parameter configuration file path of the model
       --project_name: Project name on wandb, default: kt_toolkits
       --dataset_names: Dataset names, you can fill in multiple, separated by commas ",", default: "assist2015"
       --model_names: Model names, you can fill in multiple, separated by commas ",", default: dkt
       --folds: Default: "0,1,2,3,4"
       --save_dir_suffix: Add extra characters to the model storage path name, default: ""
       --all_dir: Generate the configuration file of the model for this dataset, default: "all_wandbs"
       --launch_file: Generated sweep startup script, default: "all_start.sh"
       --generate_all: The input is "True" or "False", indicating whether to generate the wandb startup files of all datasets and models in the all_dir directory (True means: generate the startup files of all data models in the all_dir directory, False means: only the current execution is generated data model startup file), default: "False"
```

### Start the Sweep

**Step1**: `sh [launch_file] [parameter]`

```shell
sh [launch_file] > [Directed log] 2>&1
   
    - [launch_file]: required, the user submits the script of sweep to wandbs, and directs the execution output to [directed log])
    - [Directed log]: Required, execute the sweep in the log
```

Example:

```shell
python generate_wandb.py --dataset_names="assist2009,assist2015" --model_names="dkt,dkt+"
sh all_start.sh > log.all 2>&1
(The log file needs to be defined by yourself. )
```

**Step 2:** `sh run_all.sh [parameter]`

```shell
sh run_all.sh [Directed log] [start_sweep] [end_sweep] [dataset_name] [model_name] [gpu_ids] [project_name]

    - [Directed log]: Required, execute the sweep in the log
    - [start_sweep]: Required, the start id to start a sweep
    - [end_sweep]: Required, start sweep end id
    - [dataset_name]: Required, dataset name
    - [model_name]: Required, model name
    - [gpu_ids]: Required, GPU ID
    - [project_name]: optional, default: kt_toolkits
```

<!-- Execute run_all.sh, start sweep, read [directed log] here -->

Example:

```shell
sh run_all.sh log.all 0 5 assist2015 dkt 0,1,2,3,4
```

### Start Agents

```shell
sh start_sweep_0_5.sh
(0_5 represents the start sweep and end sweep)
```


### Start Evaluate


Run the `get_wandb_new` file to generate the `{modal name}_{emb type}_pred.yaml` file, modify the program keyword in the YAML file, and change its path to `./wandb_predict.py` or `wandb_predict.py` .

Then, execute the following command:

```shell
WANDB_API_KEY=xxx wandb sweep all_wandbs/dkt_qid_pred.yaml -p pykt_wandb 
#(xxx is your api_key, pykt_wandb is your project name)

#i.e.
CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=xxx nohup wandb agent swwwish/pykt_wandb/qn91y02m &
# qn91y02m is the agent name generated after the first command line is executed
```

![](../pics/predict.png)


In this stage, only 5 sweeps will be run, and no parameter tuning will be involved. After the end, export the results externally or call the wandb API for statistical results, and calculate the mean and standard deviation of each indicator in the five sweeps. The final comprehensive result is: ***mean ± standard deviation***

If you want to add you models or datasets, you can read [Contribute](contribute.md).