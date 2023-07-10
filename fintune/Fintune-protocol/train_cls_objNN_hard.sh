#!/usr/bin/env sh
mkdir -p log/objNN/hard
now=$(date +"%Y%m%d_%H%M%S")
#seed=$((122+$1))
log_name="Cls_LOG_""$1""_"""$now""
export CUDA_VISIBLE_DEVICES=0
python -u train_cls_objNN.py \
--config cfgs/config_pointnet2_cls_objNN_hard.yaml \
2>&1|tee log/objNN/hard/$log_name.log &


