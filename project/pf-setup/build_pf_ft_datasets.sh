#!/bin/bash
set -a
source .env
set +a

ds_type="wide"
path_prefix=$LSF_PATH

python -m uni2ts.data.builder.pf_simple \
  jena_weather \
  "${path_prefix}/weather/weather.csv" \
  --dataset_type $ds_type\
  --offset -1152 \
  --freq 10T \


python -m uni2ts.data.builder.pf_simple \
  istanbul_traffic \
  "${path_prefix}/istanbul-traffic-index/istanbul_traffic.csv" \
  --dataset_type $ds_type\
  --offset -192 \


python -m uni2ts.data.builder.pf_simple \
  turkey_power \
  "${path_prefix}/electrical-power-demand-in-turkey/power Generation and consumption.csv" \
  --dataset_type $ds_type \
  --offset -192 \


python -m uni2ts.data.builder.pf_simple \
  bizitobs_l2c \
  "${path_prefix}/BizITObs/L2C.csv" \
  --dataset_type $ds_type \
  --offset -1008 \


python -m uni2ts.data.builder.pf_simple \
  electricity \
  "gluonts" \
  --dataset_type $ds_type \
  --offset -192 \


python -m uni2ts.data.builder.pf_simple \
  solar-energy \
  "gluonts" \
  --dataset_type $ds_type \
  --offset -192 \
