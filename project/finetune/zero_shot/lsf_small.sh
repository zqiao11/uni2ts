#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

mode=S
cp=conf/eval
exp_name=lsf_zero_shot
context_length=3000
model=moirai_1.1_R_small


for pl in 96 192 336 720; do
  python -m cli.eval \
    -cp $cp \
    exp_name=$exp_name \
    model=$model \
    model.patch_size=64 \
    model.context_length=$context_length \
    data=lsf_test \
    data.dataset_name=ETTh1 \
    data.mode=$mode \
    data.prediction_length=$pl
done


for pl in 96 192 336 720; do
  python -m cli.eval \
    -cp $cp \
    exp_name=$exp_name \
    model=$model \
    model.patch_size=64 \
    model.context_length=$context_length \
    data=lsf_test \
    data.dataset_name=ETTh2 \
    data.mode=$mode \
    data.prediction_length=$pl
done


for pl in 96 192 336 720; do
  python -m cli.eval \
    -cp $cp \
    exp_name=$exp_name \
    model=$model \
    model.patch_size=128 \
    model.context_length=$context_length \
    data=lsf_test \
    data.dataset_name=ETTm1 \
    data.mode=$mode \
    data.prediction_length=$pl
done


for pl in 96 192 336 720; do
  python -m cli.eval \
    -cp $cp \
    exp_name=$exp_name \
    model=$model \
    model.patch_size=64 \
    model.context_length=$context_length \
    data=lsf_test \
    data.dataset_name=ETTm2 \
    data.mode=$mode \
    data.prediction_length=$pl
done


for pl in 96 192 336 720; do
  python -m cli.eval \
    -cp $cp \
    exp_name=$exp_name \
    model=$model \
    model.patch_size=64 \
    model.context_length=$context_length \
    data=lsf_test \
    data.dataset_name=electricity \
    data.mode=$mode \
    data.prediction_length=$pl
done


for pl in 96 192 336 720; do
  python -m cli.eval \
    -cp $cp \
    exp_name=$exp_name \
    model=$model \
    model.patch_size=128 \
    model.context_length=$context_length \
    data=lsf_test \
    data.dataset_name=weather \
    data.mode=$mode \
    data.prediction_length=$pl
done
