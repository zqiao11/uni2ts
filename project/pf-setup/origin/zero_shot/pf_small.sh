#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

cp=conf/pf-setup/origin/eval
exp_name=pf_zero_shot
context_length=3000
model=moirai_1.1_R_small


python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name \
  model=$model \
  model.patch_size=32 \
  model.context_length=$context_length \
  data=gluonts_test \
  data.dataset_name=electricity

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name \
  model=$model \
  model.patch_size=32 \
  model.context_length=$context_length \
  data=gluonts_test \
  data.dataset_name=solar-energy

#python -m cli.eval \
#  -cp $cp \
#  exp_name=$exp_name \
#  model=$model \
#  model=moirai_1.0_R_small \
#  model.patch_size=32 \
#  model.context_length=1000 \
#  data=gluonts_test \
#  data.dataset_name=walmart

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name \
  model=$model \
  model.patch_size=32 \
  model.context_length=$context_length \
  data=gluonts_test \
  data.dataset_name=jena_weather

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name \
  model=$model \
  model.patch_size=32 \
  model.context_length=$context_length \
  data=gluonts_test \
  data.dataset_name=istanbul_traffic

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name \
  model=$model \
  model.patch_size=64 \
  model.context_length=$context_length \
  data=gluonts_test \
  data.dataset_name=turkey_power