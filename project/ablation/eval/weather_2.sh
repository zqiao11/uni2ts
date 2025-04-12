#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

cp=conf/lsf/multi_scale/eval
model=moirai_lightning_ckpt
data=weather
cl=2000
ps=128
mode=S


cpp2='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/Idea0/freeze_ffn/weather/S/cl2000_pl192/checkpoints/epoch_6-step_9968.ckpt'

index=2
for pl in 192 ; do
  case $index in
    1) cpp=$cpp1 ;;
    2) cpp=$cpp2 ;;
    3) cpp=$cpp3 ;;
    4) cpp=$cpp4 ;;
  esac

  pretrained_model=$(echo $cpp | cut -d'/' -f6)
  exp_name=$(echo $cpp | cut -d'/' -f7)
  ft_pattern=$(echo $cpp | cut -d'/' -f8)

  python -m cli.eval \
    -cp $cp \
    exp_name=$exp_name/$pretrained_model/$ft_pattern  \
    model=$model \
    model.patch_size=$ps \
    model.context_length=$cl \
    model.checkpoint_path=$cpp \
    model.pretrained_checkpoint_path=ckpt/$pretrained_model.ckpt \
    data=lsf_test \
    data.dataset_name=$data \
    data.mode=$mode \
    data.prediction_length=$pl

  index=$((index+1))
done
