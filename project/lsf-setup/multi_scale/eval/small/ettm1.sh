#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

mode=S
cp=conf/lsf-setup/multi_scale/eval
exp_name=lsf
cl=4000
model=moirai_lightning_ckpt

cpp1='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/attn_lora/freeze_ffn/ettm1/S/cl4000_pl96/checkpoints/epoch_33-step_14178.ckpt'
cpp2='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/attn_lora/freeze_ffn/ettm1/S/cl4000_pl192/checkpoints/epoch_13-step_5824.ckpt'
cpp3='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/attn_lora/freeze_ffn/ettm1/S/cl4000_pl336/checkpoints/epoch_6-step_2898.ckpt'
cpp4='./outputs/lsf-setup/multi_scale/finetune/moirai_1.0_R_small/attn_lora/freeze_ffn/ettm1/S/cl4000_pl720/checkpoints/epoch_3-step_1632.ckpt'

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
    data.dataset_name=ETTm1 \
    data.mode=$mode \
    data.prediction_length=$pl

  index=$((index+1))
done
