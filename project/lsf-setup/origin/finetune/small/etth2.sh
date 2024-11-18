#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

model=moirai_1.0_R_small
cp=conf/lsf-setup/origin/finetune
exp_name=lsf
cl=3000
ps=64
ft_pattern=full

####### ETTh2 ######
for pl in 96 192 336 720; do
  python -m cli.train \
  -cp $cp \
  exp_name=$exp_name \
  run_name=cl${cl}_pl${pl} \
  model=$model \
  model.patch_size=$ps \
  model.context_length=$cl \
  model.prediction_length=$pl \
  model.finetune_pattern=$ft_pattern \
  data=etth2 \
  val_data=etth2 \
  val_data._args_.patch_sizes=[$ps] \
  val_data._args_.context_lengths=[$cl] \
  val_data._args_.prediction_lengths=[$pl]
done