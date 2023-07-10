#!/usr/bin/env sh
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
log_name="Cls_LOG_"$now""
export CUDA_VISIBLE_DEVICES=0
# export CUDA_HOME=/usr/local/cuda
python -u train_cls.py \
--config cfgs/config_pointnet2_cls.yaml \
2>&1|tee log/$log_name.log &
