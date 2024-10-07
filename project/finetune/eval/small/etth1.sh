#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mode=S
cp=conf/eval
exp_name=lsf_finetune
cl=3000
model=moirai_lightning_ckpt

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/full/etth1/cl3000_pl96/checkpoints/epoch_3-step_40.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/full/etth1/cl3000_pl192/checkpoints/epoch_3-step_40.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/full/etth1/cl3000_pl336/checkpoints/epoch_2-step_30.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/full/etth1/cl3000_pl720/checkpoints/epoch_1-step_20.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/etth1/cl3000_pl96/checkpoints/epoch_114-step_1150.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/etth1/cl3000_pl192/checkpoints/epoch_123-step_1240.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/etth1/cl3000_pl336/checkpoints/epoch_19-step_200.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/etth1/cl3000_pl720/checkpoints/epoch_26-step_270.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/etth1/cl3000_pl96/checkpoints/epoch_233-step_2340.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/etth1/cl3000_pl192/checkpoints/epoch_101-step_1020.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/etth1/cl3000_pl336/checkpoints/epoch_23-step_240.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/etth1/cl3000_pl720/checkpoints/epoch_37-step_380.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/etth1/cl3000_pl96/checkpoints/epoch_121-step_1220.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/etth1/cl3000_pl192/checkpoints/epoch_119-step_1200.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/etth1/cl3000_pl336/checkpoints/epoch_45-step_460.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/etth1/cl3000_pl720/checkpoints/epoch_37-step_380.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/norm/etth1/cl3000_pl96/checkpoints/epoch_796-step_7970.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/norm/etth1/cl3000_pl192/checkpoints/epoch_914-step_9150.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/norm/etth1/cl3000_pl336/checkpoints/epoch_34-step_10500.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/norm/etth1/cl3000_pl720/checkpoints/epoch_8-step_2700.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/etth1/cl3000_pl96/checkpoints/epoch_121-step_1220.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/etth1/cl3000_pl192/checkpoints/epoch_101-step_1020.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/etth1/cl3000_pl336/checkpoints/epoch_88-step_890.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/etth1/cl3000_pl720/checkpoints/epoch_37-step_380.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/etth1/cl3000_pl96/checkpoints/epoch_35-step_360.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/etth1/cl3000_pl192/checkpoints/epoch_25-step_260.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/etth1/cl3000_pl336/checkpoints/epoch_20-step_210.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/etth1/cl3000_pl720/checkpoints/epoch_2-step_30.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/k_proj/etth1/cl3000_pl96/checkpoints/epoch_35-step_360.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/k_proj/etth1/cl3000_pl192/checkpoints/epoch_26-step_270.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/k_proj/etth1/cl3000_pl336/checkpoints/epoch_20-step_210.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/k_proj/etth1/cl3000_pl720/checkpoints/epoch_15-step_160.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/etth1/cl3000_pl96/checkpoints/epoch_26-step_270.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/etth1/cl3000_pl192/checkpoints/epoch_22-step_230.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/etth1/cl3000_pl336/checkpoints/epoch_14-step_150.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/etth1/cl3000_pl720/checkpoints/epoch_2-step_30.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/etth1/cl3000_pl96/checkpoints/epoch_233-step_2340.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/etth1/cl3000_pl192/checkpoints/epoch_230-step_2310.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/etth1/cl3000_pl336/checkpoints/epoch_130-step_1310.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/etth1/cl3000_pl720/checkpoints/epoch_24-step_250.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/etth1/cl3000_pl96/checkpoints/epoch_302-step_3030.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/etth1/cl3000_pl192/checkpoints/epoch_298-step_2990.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/etth1/cl3000_pl336/checkpoints/epoch_330-step_3310.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/etth1/cl3000_pl720/checkpoints/epoch_263-step_2640.ckpt'

cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/etth1/cl3000_pl96/checkpoints/epoch_21-step_220.ckpt'
cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/etth1/cl3000_pl192/checkpoints/epoch_21-step_220.ckpt'
cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/etth1/cl3000_pl336/checkpoints/epoch_31-step_320.ckpt'
cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/etth1/cl3000_pl720/checkpoints/epoch_29-step_300.ckpt'


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
    data.dataset_name=ETTh1 \
    data.mode=$mode \
    data.prediction_length=$pl

  index=$((index+1))
done