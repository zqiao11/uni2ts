#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

model=moirai_1.1_R_small
cp=conf/pf-setup/pf/finetune
exp_name=pf
cl=3000
pl=24
ft_pattern=full

data=solar-energy
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
val_data.prediction_length=$pl
