#-----------------
#The script can lanuch multi models for one dataset.
#You can run `sh multi_run_all.sh {dataset} {models} {task_name} {log_name}> result.txt`, all the commnds for start agents will write to result.txt. project_name = $task_name-$dataset
#For example:
#sh multi_run_all.sh nips_task34 "gkt,kqn,atktfix" nips2022_tmp2 rerun_tabchen> result.txt
#all model list, "akt,dkt,dkvmn,dkt_forget,dkt+,sakt,gkt,kqn,atktfix,atkt,saint"
#-----------------

dataset=$1
models=$2
task_name=$3
log_name=$4
echo "Input params is: "
echo dataset=$1
echo models=$2
echo task_name=$3
echo log_name=$4

echo "Start generate yamls"
#generate sweep config
python generate_wandb.py --model_names $models --dataset_names $dataset  --project_name $task_name --src_dir seedwandb --all_dir $task_name

echo "Start lanuch sweeps"
#lanuch sweep
sh all_start.sh >$log_name.log 2>&1

echo "Start find agents' command"
#get agent ids
IFS=','
for i in $models; do
  echo "# $i"
  sh run_all.sh $log_name.log 0 5 $dataset $i 0,1,2,3,4 $task_name-$dataset
  echo ""
done
