#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mode=S
cp=conf/seasonal_naive_eval
exp_name=lsf_finetune
cl=3000
model=moirai_lightning_ckpt


#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/full/ettm2/cl3000_pl96/checkpoints/epoch_0-step_10.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/full/ettm2/cl3000_pl192/checkpoints/epoch_6-step_70.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/full/ettm2/cl3000_pl336/checkpoints/epoch_2-step_30.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/full/ettm2/cl3000_pl720/checkpoints/epoch_1-step_20.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/ettm2/cl3000_pl96/checkpoints/epoch_0-step_10.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/ettm2/cl3000_pl192/checkpoints/epoch_278-step_2790.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/ettm2/cl3000_pl336/checkpoints/epoch_395-step_3960.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/ettm2/cl3000_pl720/checkpoints/epoch_127-step_1280.ckpt'


#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/ettm2/cl3000_pl96/checkpoints/epoch_0-step_10.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/ettm2/cl3000_pl192/checkpoints/epoch_166-step_1670.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/ettm2/cl3000_pl336/checkpoints/epoch_113-step_1140.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/ettm2/cl3000_pl720/checkpoints/epoch_36-step_370.ckpt'


#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/ettm2/cl3000_pl96/checkpoints/epoch_0-step_10.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/ettm2/cl3000_pl192/checkpoints/epoch_168-step_1690.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/ettm2/cl3000_pl336/checkpoints/epoch_114-step_1150.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/ettm2/cl3000_pl720/checkpoints/epoch_61-step_620.ckpt'


cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/norm/ettm2/cl3000_pl96/checkpoints/epoch_0-step_100.ckpt'
cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/norm/ettm2/cl3000_pl192/checkpoints/epoch_61-step_6200.ckpt'
cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/norm/ettm2/cl3000_pl336/checkpoints/epoch_10-step_1100.ckpt'
cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/norm/ettm2/cl3000_pl720/checkpoints/epoch_11-step_1200.ckpt'


index=1
for pl in 96 192 336 720; do
  case $index in
    1) cpp=$cpp1 ;;
    2) cpp=$cpp2 ;;
    3) cpp=$cpp3 ;;
    4) cpp=$cpp4 ;;
  esac

  pretrained_model=$(echo $cpp | cut -d'/' -f4)
  ft_pattern=$(echo $cpp | cut -d'/' -f6)

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
