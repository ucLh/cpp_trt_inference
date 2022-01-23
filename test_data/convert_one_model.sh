#!/bin/bash

red=`tput setaf 1`
green=`tput setaf 2`
yellow=`tput setaf 3`
reset_color=`tput sgr0`

cd "$(dirname "$0")"

if [ ! -z "$1" ]; then
  model_in="$1"
  # Strip .onnx suffix and add .bin suffix
  model_out=${model_in:: -5}
  model_out+=".bin"
else
  echo "${red}No model provided. Aborting.${reset_color}"
  exit 1
fi

if [ ! -f ${model_out} ]; then
  echo "${green}Converting the ${model_in} network to TRT...${reset_color}"
  /usr/src/tensorrt/bin/trtexec --onnx="${model_in}" \
  --saveEngine=${model_out} \
  --explicitBatch \
  --workspace=2048 \
  --fp16
else
  echo "${yellow}The ${model_out} network has already been converted to TRT${reset_color}"
fi
