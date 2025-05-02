#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2

mode=S
cp=conf/lsf/multi_scale/eval
cl=3000
model=moirai_lightning_ckpt

cpp1='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/Final_lr1e-7_xscale_lr1e-5_head_lr5e-6/freeze_ffn/etth1/S/cl5000_pl96/checkpoints/epoch_13-step_686.ckpt'
cpp2='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/Final_lr1e-7_xscale_lr1e-5_head_lr5e-6/freeze_ffn/etth1/S/cl5000_pl192/checkpoints/epoch_1-step_96.ckpt'
cpp3='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/Final_lr5e-6_xscale1e-5_lr5e-6/freeze_ffn/etth2/S/cl3000_pl336/checkpoints/epoch_16-step_1241.ckpt'
cpp4='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/Final_lr5e-6_xscale1e-5_lr5e-6/freeze_ffn/etth2/S/cl3000_pl720/checkpoints/epoch_0-step_68.ckpt'

index=3
for pl in 336 720 ; do  # 96 192 336 720
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
    model.patch_size=64 \
    model.context_length=$cl \
    model.checkpoint_path=$cpp \
    model.pretrained_checkpoint_path=ckpt/$pretrained_model.ckpt \
    data=lsf_test \
    data.dataset_name=ETTh2 \
    data.mode=$mode \
    data.prediction_length=$pl

  index=$((index+1))
done
