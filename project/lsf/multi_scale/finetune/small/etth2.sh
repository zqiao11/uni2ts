#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=2;

model=moirai_1.0_R_small
cp=conf/lsf/multi_scale/finetune
exp_name=Final_lr5e-6_xscale1e-5_lr5e-6
data=etth2
cl=3000
ps=64
mode=S
ft_pattern=freeze_ffn


for pl in 96 192 336 720; do  # 96  192  720
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
  model.lr=5e-6 \
  model.xscale_lr=1e-5 \
  model.scale_weight_lr=1e-5 \
  model.prior_scale=0 \
  model.num_new_scales=3 \
  model.ds_factor=2 \
  model.shared_by_dim=False
done

#  model.head_lr=5e-6 \