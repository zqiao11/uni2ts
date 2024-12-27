#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2

cp=conf/pf/single_scale/eval
cl=1000
pl=24
model=moirai_lightning_ckpt


cpp=''


pretrained_model=$(echo $cpp | cut -d'/' -f6)
exp_name=$(echo $cpp | cut -d'/' -f7)
ft_pattern=$(echo $cpp | cut -d'/' -f8)

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name/$pretrained_model/$ft_pattern  \
  model=$model \
  model.patch_size=64 \
  model.context_length=$cl \
  model.checkpoint_path=$cpp \
  model.pretrained_checkpoint_path=ckpt/$pretrained_model.ckpt \
  data=gluonts_test \
  data.dataset_name=turkey_power \
  data.prediction_length=$pl
