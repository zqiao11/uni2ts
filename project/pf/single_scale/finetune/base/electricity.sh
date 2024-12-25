#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=2;

model=moirai_1.0_R_base
cp=conf/pf/single_scale/finetune
exp_name=default
cl=1000
pl=24
ft_pattern=full

data=electricity
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
trainer.callbacks.1.save_last=true \
trainer.callbacks.2.patience=5 \
trainer.max_epochs=5