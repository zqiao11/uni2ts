#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mode=S
cp=conf/lsf-setup/origin/eval
exp_name=lsf
cl=2000
model=moirai_lightning_ckpt


cpp1='./outputs/lsf/moirai_1.1_R_small/lsf/full/weather/cl3000_pl96/checkpoints/epoch_2-step_300.ckpt'
cpp2=''
cpp3=''
cpp4=''

index=1
for pl in 96 192 336 720; do
  case $index in
    1) cpp=$cpp1 ;;
    2) cpp=$cpp2 ;;
    3) cpp=$cpp3 ;;
    4) cpp=$cpp4 ;;
  esac

  pretrained_model=$(echo $cpp | cut -d'/' -f6)
  ft_pattern=$(echo $cpp | cut -d'/' -f8)

  python -m cli.eval \
    -cp $cp \
    exp_name=$exp_name/$pretrained_model/$ft_pattern  \
    model=$model \
    model.patch_size=128 \
    model.context_length=$cl \
    model.checkpoint_path=$cpp \
    model.pretrained_checkpoint_path=ckpt/$pretrained_model.ckpt \
    data=lsf_test \
    data.dataset_name=weather \
    data.mode=$mode \
    data.prediction_length=$pl

  index=$((index+1))
done
