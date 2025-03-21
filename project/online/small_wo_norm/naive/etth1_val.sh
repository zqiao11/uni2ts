#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=3

model=moirai_1.0_R_small
cp=conf/online_wo_norm
exp_name=default
data=etth1_val
cl=512
mode=S
ft_pattern=full
batch_size=7
max_epochs=1

for pl in 24 48 96; do
  for ps in 32 64; do
    for lr in 1e-4 1e-5; do
      python -m cli.train \
      -cp $cp \
      exp_name=$exp_name \
      run_name=cl${cl}_pl${pl}_ps${ps}_lr${lr} \
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
      model.lr=${lr} \
      train_dataloader.batch_size=${batch_size} \
      trainer.max_epochs=${max_epochs} \
      model.zero_shot=False \
      model.data=ETTh1
    done
  done
done


# Prediction Length: 24, Patch Size: 64, Lr: 1e-05
# Prediction Length: 48, Patch Size: 64, Lr: 1e-05
# Prediction Length: 96, Patch Size: 64, Lr: 1e-05
