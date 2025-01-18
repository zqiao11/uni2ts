#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mode=S
cp=conf/lsf/multi_scale/eval
cl=500
model=moirai_lightning_ckpt

cpp1=''
cpp2='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/weighted_loss_mfc_tid_lr5e-6_t1/freeze_ffn/etth2/S/cl500_pl192/checkpoints/epoch_27-step_3052.ckpt'
cpp3='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/weighted_loss_mfc_tid_lr5e-6_t1/freeze_ffn/etth2/S/cl500_pl336/checkpoints/epoch_20-step_2247.ckpt'
cpp4='./outputs/lsf/multi_scale/finetune/moirai_1.0_R_small/weighted_loss_mfc_tid_lr5e-6_t1/freeze_ffn/etth2/S/cl500_pl720/checkpoints/epoch_28-step_2958.ckpt'

index=2
for pl in 192 336 720; do  # 96
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
    model.patch_size=32 \
    model.context_length=$cl \
    model.checkpoint_path=$cpp \
    model.pretrained_checkpoint_path=ckpt/$pretrained_model.ckpt \
    data=lsf_test \
    data.dataset_name=ETTh2 \
    data.mode=$mode \
    data.prediction_length=$pl

  index=$((index+1))
done
