#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=3;

model=moirai_1.0_R_small
cp=conf/lsf-setup/multi_scale/finetune_two_stage
exp_name=direct_1full_2head
data=ettm2
cl=3000
ps=64
mode=S
ft_pattern=full


for pl in 96 192 336 720; do
  python -m cli.train_two_stage \
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
  val_data.mode=${mode}
#  trainer_warmup.callbacks."1".monitor=val/PackedMSELoss \
#  trainer_warmup.callbacks."2".monitor=val/PackedMSELoss
done