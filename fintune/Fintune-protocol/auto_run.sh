#for i in $(seq 1 6)
for i in $(seq 1 )
do
  #echo $i
  sh train_cls_objNN.sh $i
  #sleep 6800
done

#for i in $(seq 1 2)
##for i in 1
#do
#  sh train_cls_objNN_hard.sh $i
#  sleep 11000
#done