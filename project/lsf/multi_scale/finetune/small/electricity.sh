#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

model=moirai_1.0_R_small
cp=conf/lsf/multi_scale/finetune
exp_name=data_weight_lr1e-2_valScaled
data=electricity
cl=5000
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
  model.lr=1e-5 \
  model.scale_weight_lr=1e-2 \
  trainer.max_epochs=3 \
  trainer.callbacks.1.save_last=true \
  train_dataloader.batch_size=256 \
  trainer.callbacks.'2'.patience=1


#  model.lr=1e-5 \
#  model.scale_weight_lr=1e-5 \
#  model.temperature=10 \
#  trainer.max_epochs=3 \
#  trainer.callbacks.1.save_last=true \
#  train_dataloader.batch_size=256 \
#  trainer.callbacks.'2'.patience=1
done