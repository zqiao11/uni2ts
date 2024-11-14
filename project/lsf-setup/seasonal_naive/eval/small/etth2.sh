#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mode=S
cp=conf/lsf-setup/seasonal_naive/eval
exp_name=lsf
cl=3000
model=moirai_lightning_ckpt


#cpp1='./outputs/lsf/moirai_1.1_R_small/lsf/full/etth2/cl3000_pl96/checkpoints/epoch_0-step_10.ckpt'
#cpp2='./outputs/lsf/moirai_1.1_R_small/lsf/full/etth2/cl3000_pl192/checkpoints/epoch_0-step_10.ckpt'
#cpp3='./outputs/lsf/moirai_1.1_R_small/lsf/full/etth2/cl3000_pl336/checkpoints/epoch_0-step_10.ckpt'
#cpp4='./outputs/lsf/moirai_1.1_R_small/lsf/full/etth2/cl3000_pl720/checkpoints/epoch_0-step_10.ckpt'

#cpp1='./outputs/lsf/moirai_1.1_R_small/lsf/param_proj/etth2/cl3000_pl96/checkpoints/epoch_78-step_790.ckpt'
#cpp2='./outputs/lsf/moirai_1.1_R_small/lsf/param_proj/etth2/cl3000_pl192/checkpoints/epoch_113-step_1140.ckpt'
#cpp3='./outputs/lsf/moirai_1.1_R_small/lsf/param_proj/etth2/cl3000_pl336/checkpoints/epoch_153-step_1540.ckpt'
#cpp4='./outputs/lsf/moirai_1.1_R_small/lsf/param_proj/etth2/cl3000_pl720/checkpoints/epoch_20-step_210.ckpt'

#cpp1='./outputs/lsf/moirai_1.1_R_small/lsf/in_proj/etth2/cl3000_pl96/checkpoints/epoch_5-step_60.ckpt'
#cpp2='./outputs/lsf/moirai_1.1_R_small/lsf/in_proj/etth2/cl3000_pl192/checkpoints/epoch_4-step_50.ckpt'
#cpp3='./outputs/lsf/moirai_1.1_R_small/lsf/in_proj/etth2/cl3000_pl336/checkpoints/epoch_6-step_70.ckpt'
#cpp4='./outputs/lsf/moirai_1.1_R_small/lsf/in_proj/etth2/cl3000_pl720/checkpoints/epoch_14-step_150.ckpt'

#cpp1='./outputs/lsf/moirai_1.1_R_small/lsf/in_proj+param_proj/etth2/cl3000_pl96/checkpoints/epoch_4-step_50.ckpt'
#cpp2='./outputs/lsf/moirai_1.1_R_small/lsf/in_proj+param_proj/etth2/cl3000_pl192/checkpoints/epoch_4-step_50.ckpt'
#cpp3='./outputs/lsf/moirai_1.1_R_small/lsf/in_proj+param_proj/etth2/cl3000_pl336/checkpoints/epoch_7-step_80.ckpt'
#cpp4='./outputs/lsf/moirai_1.1_R_small/lsf/in_proj+param_proj/etth2/cl3000_pl720/checkpoints/epoch_14-step_150.ckpt'


cpp1='./outputs/lsf/moirai_1.1_R_small/lsf/norm/etth2/cl3000_pl96/checkpoints/epoch_3-step_400.ckpt'
cpp2='./outputs/lsf/moirai_1.1_R_small/lsf/norm/etth2/cl3000_pl192/checkpoints/epoch_4-step_500.ckpt'
cpp3='./outputs/lsf/moirai_1.1_R_small/lsf/norm/etth2/cl3000_pl336/checkpoints/epoch_4-step_500.ckpt'
cpp4='./outputs/lsf/moirai_1.1_R_small/lsf/norm/etth2/cl3000_pl720/checkpoints/epoch_6-step_700.ckpt'

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
    data.dataset_name=ETTh2 \
    data.mode=$mode \
    data.prediction_length=$pl

  index=$((index+1))
done
