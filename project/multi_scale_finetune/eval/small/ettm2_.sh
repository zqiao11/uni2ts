#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mode=S
cp=conf/multi_scale_eval
exp_name=lsf_finetune
cl=3000
model=moirai_lightning_ckpt


cpp='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/ettm2/cl3000_pl96/checkpoints/epoch_0-step_10.ckpt'
pl=96

pretrained_model=$(echo $cpp | cut -d'/' -f4)
ft_pattern=$(echo $cpp | cut -d'/' -f6)

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name/$pretrained_model/$ft_pattern  \
  model=$model \
  model.patch_size=64 \
  model.context_length=$cl \
  model.checkpoint_path=$cpp \
  model.pretrained_checkpoint_path=ckpt/$pretrained_model.ckpt \
  data=lsf_test \
  data.dataset_name=ETTm2 \
  data.mode=$mode \
  data.prediction_length=$pl

