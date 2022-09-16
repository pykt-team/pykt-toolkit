models="dkt+,sakt,gkt,kqn,atktfix,atkt,saint"
IFS=','
for i in $models; do
  echo "# $i"
  sh run_all.sh rerun.log 0 5 assist2009 $i 0,1,2,3,4 nips2022-assist2009
  echo ""
done
