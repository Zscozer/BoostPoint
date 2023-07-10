#!/usr/bin/env sh
mkdir -p log/objNN
now=$(date +"%Y%m%d_%H%M%S")
seed=$((109+$1))
#echo seed=$seed
log_name="Cls_LOG_""$1""_"""$now""
export CUDA_VISIBLE_DEVICES=0
python -u train_cls_objNN.py \
--config cfgs/config_pointnet2_cls_objNN.yaml \
--seed $seed \
2>&1|tee log/objNN/$log_name.log &
