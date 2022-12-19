
dataset="ednet"
models="dkt,akt,iekt,qdkt,qikt,saint"

IFS=','
for i in $models; do
  echo "# $i"
  python -u extract_raw_result.py --dataset $dataset --model_name $i
  echo ""
done


# nohup sh extract_raw.sh >extract.log&