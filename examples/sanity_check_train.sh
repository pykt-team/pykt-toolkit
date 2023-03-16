dataset="assist2009,algebra2005,nips_task34,statics2011,assist2015,poj,bridge2algebra2006"

IFS=','
for dataset in $dataset; do
  echo "# $i"
  python -u wandb_cl4kt_train.py --dataset $dataset > sanity_check_train_$dataset.log 2>&1
  echo ""
done


# nohup sh extract_raw.sh >extract.log&