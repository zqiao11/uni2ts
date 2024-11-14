#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

cp=conf/pf-setup/pf/eval
exp_name=pf
cl=3000
pl=24
model=moirai_lightning_ckpt


cpp='./outputs/pf-setup/pf/finetune/moirai_1.1_R_small/pf/full/solar-energy/cl3000_pl24/checkpoints/epoch_0-step_63.ckpt'


pretrained_model=$(echo $cpp | cut -d'/' -f6)
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
  data.dataset_name=solar-energy \
  data.prediction_length=$pl
