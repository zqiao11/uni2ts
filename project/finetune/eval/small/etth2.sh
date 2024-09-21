#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

mode=S
cp=conf/eval
exp_name=lsf_finetune
cl=3000
model=moirai_lightning_ckpt


cpp1='/home/zhongzheng/uni2ts-mm/outputs/finetune/moirai_1.0_R_small/lsf/etth2/cl3000_pl96/checkpoints/epoch_37-step_380.ckpt'
cpp2='/home/zhongzheng/uni2ts-mm/outputs/finetune/moirai_1.0_R_small/lsf/etth2/cl3000_pl192/checkpoints/epoch_51-step_520.ckpt'
cpp3='/home/zhongzheng/uni2ts-mm/outputs/finetune/moirai_1.0_R_small/lsf/etth2/cl3000_pl336/checkpoints/epoch_27-step_280.ckpt'
cpp4='/home/zhongzheng/uni2ts-mm/outputs/finetune/moirai_1.0_R_small/lsf/etth2/cl3000_pl720/checkpoints/epoch_0-step_100.ckpt'

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
    data.dataset_name=ETTh2 \
    data.mode=$mode \
    data.prediction_length=$pl

  index=$((index+1))
done
