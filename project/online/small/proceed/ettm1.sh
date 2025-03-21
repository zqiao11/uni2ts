#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

model=moirai_1.0_R_small_proceed
cp=conf/online
exp_name=default
data=ettm1 # ettm1_val
cl=512
ps=128
mode=S
ft_pattern=full


for pl in 96; do  # 24 48
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
  model.lr=1e-5 \
  train_dataloader.batch_size=7 \
  trainer.max_epochs=1 \
  model.zero_shot=False
done