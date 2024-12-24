#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2

mode=S
cp=conf/lsf/multi_scale/eval
cl=5000
model=moirai_lightning_ckpt

cpp1='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/ms_qkv_1.0/full/weather/S/cl2000_pl96/checkpoints/epoch_10-step_15708.ckpt'
cpp2='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/ms_qkv_1.0/full/weather/S/cl2000_pl192/checkpoints/epoch_7-step_11392.ckpt'
cpp3='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/ms_qkv_1.0/full/weather/S/cl2000_pl336/checkpoints/epoch_4-step_7090.ckpt'
cpp4='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/ms_qkv_1.0/full/weather/S/cl2000_pl720/checkpoints/epoch_2-step_4206.ckpt'

index=1
for pl in 96 192 336 720; do
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