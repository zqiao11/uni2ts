#!/bin/bash
set -a
source .env
set +a

ds_type="wide"  # "wide_multivariate"
path_prefix=$LSF_PATH

for data in ETTh1 ETTh2; do
  python -m uni2ts.data.builder.online \
    $data \
    "${path_prefix}/ETT-small/${data}.csv" \
    --dataset_type $ds_type\
    --offset 14400
done

for data in ETTm1 ETTm2; do
  python -m uni2ts.data.builder.online \
    $data \
    "${path_prefix}/ETT-small/${data}.csv" \
    --dataset_type $ds_type\
    --offset 57600
done

python -m uni2ts.data.builder.online \
  weather \
  "${path_prefix}/weather/weather.csv" \
  --dataset_type $ds_type\
  --freq 10T

python -m uni2ts.data.builder.online \
  electricity \
  "${path_prefix}/electricity/electricity.csv" \
  --dataset_type $ds_type
