#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=2;

model=moirai_1.0_R_small
cp=conf/lsf/multi_scale/finetune
exp_name=Final_lr1e-7_xscale_lr5e-6_head_lr5e-6
data=etth1
cl=5000
ps=64
mode=S
ft_pattern=freeze_ffn


for pl in 96 192 336 720 ; do  #  96 192 336 720
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
  model.lr=1e-7 \
  model.xscale_lr=5e-6 \
  model.head_lr=5e-6 \
  model.scale_weight_lr=1e-2 \
  model.prior_scale=0 \
  model.num_new_scales=3 \
  model.ds_factor=2 \
  model.shared_by_dim=False
done


#  model.num_new_scales=1 \
#model.prior_scale=0 \
# model.prior_scale=0 \
#  trainer.callbacks."1".monitor=val/PackedMSELoss \
#  trainer.callbacks."2".monitor=val/PackedMSELoss