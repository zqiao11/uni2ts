#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mode=S
cp=conf/lsf/eval
exp_name=lsf
cl=3000
model=moirai_lightning_ckpt


cpp1='./outputs/lsf/finetune/moirai_1.1_R_small/lsf/full/ettm2/cl3000_pl96/checkpoints/epoch_9-step_2160.ckpt'
cpp2='./outputs/lsf/finetune/moirai_1.1_R_small/lsf/full/ettm2/cl3000_pl192/checkpoints/epoch_4-step_1075.ckpt'
cpp3='./outputs/lsf/finetune/moirai_1.1_R_small/lsf/full/ettm2/cl3000_pl336/checkpoints/epoch_1-step_428.ckpt'
cpp4='./outputs/lsf/finetune/moirai_1.1_R_small/lsf/full/ettm2/cl3000_pl720/checkpoints/epoch_1-step_422.ckpt'


index=1
for pl in 96 192 336 720; do
  case $index in
    1) cpp=$cpp1 ;;
    2) cpp=$cpp2 ;;
    3) cpp=$cpp3 ;;
    4) cpp=$cpp4 ;;
  esac

  pretrained_model=$(echo $cpp | cut -d'/' -f5)
  ft_pattern=$(echo $cpp | cut -d'/' -f7)

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
