#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

model=moirai_1.0_R_small
cp=conf/online_wo_norm
exp_name=default
data=etth1
cl=512
mode=S
ft_pattern=full
batch_size=7
max_epochs=1
zero_shot=False

# 定义参数组合
declare -a experiments=(
    "pl=24 ps=64 lr=1e-5"
    "pl=48 ps=64 lr=1e-5"
    "pl=96 ps=64 lr=1e-5"
)

# 统一 for loop 执行
for exp in "${experiments[@]}"; do
    eval "$exp"
    python -m cli.train \
        -cp "$cp" \
        exp_name="$exp_name" \
        run_name="cl${cl}_pl${pl}" \
        model="$model" \
        model.patch_size="$ps" \
        model.context_length="$cl" \
        model.prediction_length="$pl" \
        model.finetune_pattern="$ft_pattern" \
        data="$data" \
        data.patch_size="$ps" \
        data.context_length="$cl" \
        data.prediction_length="$pl" \
        data.mode="$mode" \
        model.lr="$lr" \
        train_dataloader.batch_size="$batch_size" \
        trainer.max_epochs="$max_epochs" \
        model.zero_shot="$zero_shot" \
        model.data=ETTh1
done