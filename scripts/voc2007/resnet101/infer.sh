#!/usr/bin/env bash
CHECKPOINT=$1
INPUT_IMAGE=$2
OUTPUT_IMAGE=$3
python infer.py -c=${CHECKPOINT} -s=voc2007 -b=resnet101 --pooling_mode=align ${INPUT_IMAGE} ${OUTPUT_IMAGE}