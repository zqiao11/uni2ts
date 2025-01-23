#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mode=S
cp=conf/lsf/multi_scale/eval
cl=5000
model=moirai_lightning_ckpt


cpp1=''
cpp2=''
cpp3='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/data_weight_lr1e-2_valScaled_DFscaled/freeze_ffn/electricity/S/cl5000_pl336/checkpoints/epoch_2-step_49194.ckpt'
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
    data.dataset_name=electricity \
    data.mode=$mode \
    data.prediction_length=$pl

  index=$((index+1))
done

