#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

cp=conf/lsf-setup/lsf/eval
exp_name=lsf
model=moirai_lightning_ckpt
data=ETTm1
cl=5000
ps=64
mode=S


cpp1='./outputs/lsf-setup/lsf/finetune/moirai_1.0_R_small/lsf/full/ettm1/cl4000_pl96/checkpoints/epoch_5-step_1254.ckpt'
#cpp2='./outputs/lsf-setup/lsf/finetune/moirai_1.0_R_small/lsf/full/ettm1/cl4000_pl192/checkpoints/epoch_2-step_624.ckpt'
#cpp3='./outputs/lsf-setup/lsf/finetune/moirai_1.0_R_small/lsf/full/ettm1/cl4000_pl336/checkpoints/epoch_1-step_414.ckpt'
#cpp4='./outputs/lsf-setup/lsf/finetune/moirai_1.0_R_small/lsf/full/ettm1/cl4000_pl720/checkpoints/epoch_0-step_204.ckpt'

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
