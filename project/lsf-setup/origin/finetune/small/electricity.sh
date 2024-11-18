#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

model=moirai_1.0_R_small
cp=conf/lsf-setup/origin/finetune
exp_name=lsf
cl=5000
ps=64
ft_pattern=full

####### Electricity ######
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
  data=electricity \
  val_data=electricity \
  val_data._args_.patch_sizes=[$ps] \
  val_data._args_.context_lengths=[$cl] \
  val_data._args_.prediction_lengths=[$pl] \
  train_dataloader.num_batches_per_epoch=300 \
  trainer.callbacks.2.patience=10
done