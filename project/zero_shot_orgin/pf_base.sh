#!/bin/bash

export HYDRA_FULL_ERROR=1; export CUDA_VISIBLE_DEVICES=0;

cp=conf/origin/eval
exp_name=pf_zero_shot
model=moirai_1.0_R_base


python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name \
  model=$model \
  model.patch_size=32 \
  model.context_length=1000 \
  data=gluonts_test \
  data.dataset_name=electricity

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name \
  model=$model \
  model.patch_size=64 \
  model.context_length=4000 \
  data=gluonts_test \
  data.dataset_name=solar-energy

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name \
  model=$model \
  model.patch_size=32 \
  model.context_length=4000 \
  data=gluonts_test \
  data.dataset_name=walmart

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name \
  model=$model \
  model.patch_size=128 \
  model.context_length=5000 \
  data=gluonts_test \
  data.dataset_name=jena_weather

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name \
  model=$model \
  model.patch_size=32 \
  model.context_length=1000 \
  data=gluonts_test \
  data.dataset_name=istanbul_traffic

python -m cli.eval \
  -cp $cp \
  exp_name=$exp_name \
  model=$model \
  model.patch_size=64 \
  model.context_length=1000 \
  data=gluonts_test \
  data.dataset_name=turkey_power