#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

model=moirai_1.1_R_small
cp=conf/multi_scale_finetune
exp_name=lsf
cl=3000
ft_pattern=in_proj+param_proj

# param_proj
# in_proj
# norm
# mask
# ffn
# param_proj + in_proj
# param_proj + in_proj + norm

##### Weather ######
for pl in 96 192 336 720; do
  python -m cli.train \
  -cp $cp \
  exp_name=$exp_name \
  run_name=cl${cl}_pl${pl} \
  model=$model \
  model.patch_size=128 \
  model.context_length=$cl \
  model.prediction_length=$pl \
  model.finetune_pattern=$ft_pattern \
  data=weather \
  val_data=weather \
  val_data._args_.patch_sizes=[128] \
  val_data._args_.context_lengths=[$cl] \
  val_data._args_.prediction_lengths=[$pl] \
  train_dataloader.num_batches_per_epoch=300 \
  trainer.callbacks.2.patience=10 \
#  model.lr=1e-4
done