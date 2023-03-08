
dataset="nips_task34,statics2011,algebra2005,bridge2algebra2006"
models="bakt_qid_disentangled_sparseattn"

#,nips_task34,statics2011,algebra2005,bridge2algebra2006
IFS=','
for one_dataset in $dataset; do
  echo "Dataset is $one_dataset"
  for model in $models; do
  echo "Model is $model"
  python -u extract_raw_result.py --dataset $one_dataset --model_name $model
  echo ""
  done
done
# nohup sh extract_raw.sh >extract.log&