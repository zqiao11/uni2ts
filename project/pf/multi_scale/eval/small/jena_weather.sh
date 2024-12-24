#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

cp=conf/pf/multi_scale/eval
cl=4000
pl=144
model=moirai_lightning_ckpt


cpp='./outputs/pf/pf/finetune/moirai_1.0_R_small/pf/full/jena_weather/cl4000_pl144/checkpoints/epoch_4-step_9725.ckpt'


pretrained_model=$(echo $cpp | cut -d'/' -f6)
exp_name=$(echo $cpp | cut -d'/' -f7)
ft_pattern=$(echo $cpp | cut -d'/' -f8)

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name/$pretrained_model/$ft_pattern  \
  model=$model \
  model.patch_size=32 \
  model.context_length=$cl \
  model.checkpoint_path=$cpp \
  model.pretrained_checkpoint_path=ckpt/$pretrained_model.ckpt \
  data=gluonts_test \
  data.dataset_name=jena_weather \
  data.prediction_length=$pl
