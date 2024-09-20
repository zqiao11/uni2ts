#!/bin/bash

set -a
source .env
set +a

ds_type="wide"  # "wide_multivariate"
path_prefix=$LSF_PATH

for data in ETTh1 ETTh2; do
  python -m uni2ts.data.builder.simple \
    $data \
    "${path_prefix}/ETT-small/${data}.csv" \
    --dataset_type $ds_type\
    --offset 8640 \
    --normalize
done


for data in ETTm1 ETTm2; do
  python -m uni2ts.data.builder.simple \
    $data \
    "${path_prefix}/ETT-small/${data}.csv" \
    --dataset_type $ds_type\
    --offset 34560 \
    --normalize
done


python -m uni2ts.data.builder.simple \
  weather \
  "${path_prefix}/weather/weather.csv" \
  --dataset_type $ds_type\
  --offset 36887 \
  --freq 10T \
  --normalize


python -m uni2ts.data.builder.simple \
  electricity \
  "${path_prefix}/electricity/electricity.csv" \
  --dataset_type $ds_type\
  --offset 18412 \
  --normalize
