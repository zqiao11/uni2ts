#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mode=S
cp=conf/eval
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


#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/ettm2/cl3000_pl96/checkpoints/epoch_0-step_10.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/ettm2/cl3000_pl192/checkpoints/epoch_231-step_2320.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/ettm2/cl3000_pl336/checkpoints/epoch_113-step_1140.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/in_proj+param_proj+norm/ettm2/cl3000_pl720/checkpoints/epoch_61-step_620.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/ettm2/cl3000_pl96/checkpoints/epoch_0-step_10.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/ettm2/cl3000_pl192/checkpoints/epoch_36-step_370.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/ettm2/cl3000_pl336/checkpoints/epoch_19-step_200.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/q_proj/ettm2/cl3000_pl720/checkpoints/epoch_14-step_150.ckpt'


#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/k_proj/ettm2/cl3000_pl96/checkpoints/epoch_0-step_10.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/k_proj/ettm2/cl3000_pl192/checkpoints/epoch_36-step_370.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/k_proj/ettm2/cl3000_pl336/checkpoints/epoch_22-step_230.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/k_proj/ettm2/cl3000_pl720/checkpoints/epoch_13-step_140.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/ettm2/cl3000_pl96/checkpoints/epoch_0-step_10.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/ettm2/cl3000_pl192/checkpoints/epoch_53-step_540.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/ettm2/cl3000_pl336/checkpoints/epoch_21-step_220.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/v_proj/ettm2/cl3000_pl720/checkpoints/epoch_13-step_140.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/ettm2/cl3000_pl96/checkpoints/epoch_0-step_10.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/ettm2/cl3000_pl192/checkpoints/epoch_384-step_3850.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/ettm2/cl3000_pl336/checkpoints/epoch_428-step_4290.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/studentT/ettm2/cl3000_pl720/checkpoints/epoch_171-step_1720.ckpt'

#cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/ettm2/cl3000_pl96/checkpoints/epoch_0-step_10-v1.ckpt'
#cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/ettm2/cl3000_pl192/checkpoints/epoch_346-step_3470.ckpt'
#cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/ettm2/cl3000_pl336/checkpoints/epoch_391-step_3920.ckpt'
#cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/param_proj_replace/ettm2/cl3000_pl720/checkpoints/epoch_499-step_5000.ckpt'

cpp1='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/ettm2/cl3000_pl96/checkpoints/epoch_137-step_1380.ckpt'
cpp2='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/ettm2/cl3000_pl192/checkpoints/epoch_267-step_2680.ckpt'
cpp3='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/ettm2/cl3000_pl336/checkpoints/epoch_181-step_1820.ckpt'
cpp4='./outputs/finetune/moirai_1.1_R_small/lsf/full_sn/ettm2/cl3000_pl720/checkpoints/epoch_221-step_2220.ckpt'

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