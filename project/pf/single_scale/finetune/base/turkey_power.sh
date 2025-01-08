#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=3;

model=moirai_1.0_R_base
cp=conf/pf/single_scale/finetune
exp_name=lr5e-6_patience30
cl=1000
pl=24
ft_pattern=full

data=turkey_power
ps=64

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
trainer.callbacks.'2'.patience=30 \
trainer.callbacks.1.save_last=true \
model.lr=5e-6
