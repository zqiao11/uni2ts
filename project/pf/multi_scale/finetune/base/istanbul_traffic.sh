#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

model=moirai_1.0_R_base
cp=conf/pf/multi_scale/finetune
exp_name=Istanbul_lr5e-6_swlr1e-3_bs512
cl=1000
pl=24
ft_pattern=freeze_ffn

data=istanbul_traffic
ps=32

python -m cli.train \
-cp $cp \
exp_name=$exp_name \
run_name=cl${cl}_pl${pl} \
model=$model \
model.patch_size=${ps} \
model.context_length=$cl \
model.prediction_length=$pl \
model.finetune_pattern=$ft_pattern \
data=${data} \
data.patch_size=${ps} \
data.context_length=$cl \
data.prediction_length=$pl \
val_data=${data} \
val_data.patch_size=${ps} \
val_data.context_length=$cl \
val_data.prediction_length=$pl \
model.lr=5e-6 \
model.scale_weight_lr=1e-3