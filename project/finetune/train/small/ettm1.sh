#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

model=moirai_1.1_R_small
cp=conf/finetune
exp_name=lsf
cl=3000
ft_pattern=full

# param_proj
# in_proj
# norm
# mask
# ffn
# param_proj + in_proj
# param_proj + in_proj + norm

####### ETTm1 ######
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
  data=ettm1 \
  val_data=ettm1 \
  val_data._args_.patch_sizes=[128] \
  val_data._args_.context_lengths=[$cl] \
  val_data._args_.prediction_lengths=[$pl] \
#  train_dataloader.num_batches_per_epoch=100
done


#python -m cli.train \
#-cp $cp \
#exp_name=$exp_name \
#run_name=cl${cl}_pl${pl} \
#model=$model \
#model.patch_size=64 \
#model.context_length=$cl \
#model.prediction_length=$pl \
#data=etth2 \
#val_data=etth2 \
#val_data._args_.patch_sizes=[64] \
#val_data._args_.context_lengths=[$cl] \
#val_data._args_.prediction_lengths=[$pl] \
#model.lr=1e-7 \
#model.lora_for_moirai=True \
#model.linear_probe=False \
#train_dataloader.num_batches_per_epoch=10