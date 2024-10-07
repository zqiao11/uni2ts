#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mode=S
cp=conf/eval
exp_name=lsf_finetune
cl=3000
model=moirai_lightning_ckpt


#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/full/ettm1/cl3000_pl96/checkpoints/epoch_4-step_50.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/full/ettm1/cl3000_pl192/checkpoints/epoch_2-step_30.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/full/ettm1/cl3000_pl336/checkpoints/epoch_0-step_10.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/full/ettm1/cl3000_pl720/checkpoints/epoch_0-step_10.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/ettm1/cl3000_pl96/checkpoints/epoch_199-step_2000.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/ettm1/cl3000_pl192/checkpoints/epoch_98-step_990.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/ettm1/cl3000_pl336/checkpoints/epoch_25-step_260.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj/ettm1/cl3000_pl720/checkpoints/epoch_27-step_280.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/ettm1/cl3000_pl96/checkpoints/epoch_42-step_430.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/ettm1/cl3000_pl192/checkpoints/epoch_28-step_290.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/ettm1/cl3000_pl336/checkpoints/epoch_24-step_250.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj/ettm1/cl3000_pl720/checkpoints/epoch_14-step_150.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/ettm1/cl3000_pl96/checkpoints/epoch_39-step_400.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/ettm1/cl3000_pl192/checkpoints/epoch_24-step_250.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/ettm1/cl3000_pl336/checkpoints/epoch_23-step_240.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj/ettm1/cl3000_pl720/checkpoints/epoch_12-step_130.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/norm/ettm1/cl3000_pl96/checkpoints/epoch_29-step_3000.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/norm/ettm1/cl3000_pl192/checkpoints/epoch_17-step_1800.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/norm/ettm1/cl3000_pl336/checkpoints/epoch_15-step_1600.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/norm/ettm1/cl3000_pl720/checkpoints/epoch_11-step_1200.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/ettm1/cl3000_pl96/checkpoints/epoch_41-step_420.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/ettm1/cl3000_pl192/checkpoints/epoch_24-step_250.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/ettm1/cl3000_pl336/checkpoints/epoch_23-step_240.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/ettm1/cl3000_pl720/checkpoints/epoch_16-step_170.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/ettm1/cl3000_pl96/checkpoints/epoch_50-step_510.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/ettm1/cl3000_pl192/checkpoints/epoch_30-step_310.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/ettm1/cl3000_pl336/checkpoints/epoch_25-step_260.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/ettm1/cl3000_pl720/checkpoints/epoch_16-step_170.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/ettm1/cl3000_pl96/checkpoints/epoch_28-step_290.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/ettm1/cl3000_pl192/checkpoints/epoch_19-step_200.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/ettm1/cl3000_pl336/checkpoints/epoch_9-step_100.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/ettm1/cl3000_pl720/checkpoints/epoch_8-step_90.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/ettm1/cl3000_pl96/checkpoints/epoch_230-step_2310.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/ettm1/cl3000_pl192/checkpoints/epoch_168-step_1690.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/ettm1/cl3000_pl336/checkpoints/epoch_65-step_660.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/ettm1/cl3000_pl720/checkpoints/epoch_75-step_760.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/ettm1/cl3000_pl96/checkpoints/epoch_485-step_4860.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/ettm1/cl3000_pl192/checkpoints/epoch_390-step_3910.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/ettm1/cl3000_pl336/checkpoints/epoch_200-step_2010.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/ettm1/cl3000_pl720/checkpoints/epoch_176-step_1770.ckpt'

cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/ettm1/cl3000_pl96/checkpoints/epoch_34-step_350.ckpt'
cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/ettm1/cl3000_pl192/checkpoints/epoch_64-step_650.ckpt'
cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/ettm1/cl3000_pl336/checkpoints/epoch_113-step_1140.ckpt'
cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/ettm1/cl3000_pl720/checkpoints/epoch_134-step_1350.ckpt'

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
