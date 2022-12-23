
dataset="assist2009"
models="akt_qid,akt_qid_selfattn_fixed_add_block"

IFS=','
for i in $models; do
  echo "# $i"
  python -u extract_raw_result.py --dataset $dataset --model_name $i
  echo ""
done


# nohup sh extract_raw.sh >extract.log&