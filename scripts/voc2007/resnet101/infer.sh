#!/usr/bin/env bash
CHECKPOINT=$1
INPUT_IMAGE=$2
OUTPUT_IMAGE=$3
if ! ([[ -n "${CHECKPOINT}" ]] && [[ -n "${INPUT_IMAGE}" ]] && [[ -n "${OUTPUT_IMAGE}" ]]); then
    echo "Argument CHECKPOINT or INPUT_IMAGE or OUTPUT_IMAGE is missing"
    exit
fi

python infer.py -c=${CHECKPOINT} -s=voc2007 -b=resnet101 ${INPUT_IMAGE} ${OUTPUT_IMAGE}