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


#### ETTh1 ######
for pl in 96 192 336 720; do
  python -m cli.train \
  -cp $cp \
  exp_name=$exp_name \
  run_name=cl${cl}_pl${pl} \
  model=$model \
  model.patch_size=64 \
  model.context_length=$cl \
  model.prediction_length=$pl \
  model.finetune_pattern=$ft_pattern \
  data=etth1 \
  val_data=etth1 \
  val_data._args_.patch_sizes=[64] \
  val_data._args_.context_lengths=[$cl] \
  val_data._args_.prediction_lengths=[$pl]
done


#pl=96
#python -m cli.train \
#-cp $cp \
#exp_name=$exp_name \
#run_name=cl${cl}_pl${pl} \
#model=$model \
#model.patch_size=64 \
#model.context_length=$cl \
#model.prediction_length=$pl \
#model.finetune_pattern=$ft_pattern \
#data=etth1 \
#val_data=etth1 \
#val_data._args_.patch_sizes=[64] \
#val_data._args_.context_lengths=[$cl] \
#val_data._args_.prediction_lengths=[$pl]