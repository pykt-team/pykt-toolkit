
dataset="assist2009,nips_task34,statics2011,poj,assist2015"
models="bakt_qid_sparseattn,bakt_qid_disentangled_sparse_attn"

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