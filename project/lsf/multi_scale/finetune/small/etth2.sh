#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=2;

model=moirai_1.0_R_small
cp=conf/lsf/multi_scale/finetune
exp_name=Etth2_cl3000_w010_lr5e-7_wlr1e-2
data=etth2
cl=3000
ps=64
mode=S
ft_pattern=freeze_ffn


for pl in 96 192 336 720; do
  python -m cli.train \
  -cp $cp \
  exp_name=$exp_name \
  run_name=cl${cl}_pl${pl} \
  model=$model \
  model.patch_size=${ps} \
  model.context_length=$cl \
  model.prediction_length=$pl \
  model.finetune_pattern=$ft_pattern \
  data=${data} \
  data.patch_size=${ps} \
  data.context_length=$cl \
  data.prediction_length=$pl \
  data.mode=${mode} \
  val_data=${data} \
  val_data.patch_size=${ps} \
  val_data.context_length=$cl \
  val_data.prediction_length=$pl \
  val_data.mode=${mode} \
  model.lr=5e-7 \
  model.scale_weight_lr=1e-2 \
  model.prior_scale=0
done