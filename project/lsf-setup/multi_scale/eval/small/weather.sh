#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=3

mode=S
cp=conf/lsf-setup/multi_scale/eval
exp_name=lsf
cl=2000
model=moirai_lightning_ckpt

#cpp1='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/learned_time_id/full/weather/S/cl2000_pl96/checkpoints/epoch_8-step_12852.ckpt'
#cpp2='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/learned_time_id/full/weather/S/cl2000_pl192/checkpoints/epoch_6-step_9968.ckpt'
cpp3='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/learned_time_id/full/weather/S/cl2000_pl336/checkpoints/epoch_5-step_8508.ckpt'
cpp4='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/learned_time_id/full/weather/S/cl2000_pl720/checkpoints/epoch_3-step_5608.ckpt'

index=1
for pl in 336 720; do  # 96 192
  case $index in
    1) cpp=$cpp3 ;;
    2) cpp=$cpp4 ;;

#    1) cpp=$cpp1 ;;
#    2) cpp=$cpp2 ;;
#    3) cpp=$cpp3 ;;
#    4) cpp=$cpp4 ;;
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