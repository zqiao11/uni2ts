#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=1;

model=moirai_1.0_R_small
cp=conf/lsf-setup/multi_scale/finetune
exp_name=learned_time_id_valMSE
data=weather
cl=2000
ps=128
mode=S  # M
ft_pattern=full


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
  trainer.callbacks."1".monitor=val/PackedMSELoss \
  trainer.callbacks."2".monitor=val/PackedMSELoss
done