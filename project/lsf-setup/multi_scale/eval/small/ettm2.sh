#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

mode=S
cp=conf/lsf-setup/multi_scale/eval
exp_name=lsf
cl=3000
model=moirai_lightning_ckpt

cpp1='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/scale_bias_10000/freeze_ffn/ettm2/S/cl3000_pl96/checkpoints/epoch_14-step_6465.ckpt'
cpp2='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/scale_bias_10000/freeze_ffn/ettm2/S/cl3000_pl192/checkpoints/epoch_7-step_3432.ckpt'
cpp3='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/scale_bias_10000/freeze_ffn/ettm2/S/cl3000_pl336/checkpoints/epoch_3-step_1708.ckpt'
cpp4='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/scale_bias_10000/freeze_ffn/ettm2/S/cl3000_pl720/checkpoints/epoch_0-step_422.ckpt'

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
    model.patch_size=64 \
    model.context_length=$cl \
    model.checkpoint_path=$cpp \
    model.pretrained_checkpoint_path=ckpt/$pretrained_model.ckpt \
    data=lsf_test \
    data.dataset_name=ETTm2 \
    data.mode=$mode \
    data.prediction_length=$pl

  index=$((index+1))
done
