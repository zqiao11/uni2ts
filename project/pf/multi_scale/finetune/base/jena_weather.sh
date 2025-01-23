#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=3;

model=moirai_1.0_R_base
cp=conf/pf/multi_scale/finetune
exp_name=default
cl=5000
pl=144
ft_pattern=freeze_ffn

data=jena_weather
ps=128

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
train_dataloader.batch_size=256 \
model.lr=5e-5 \
model.scale_weight_lr=1e-2 \
trainer.callbacks.'2'.patience=1
