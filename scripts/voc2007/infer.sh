#!/usr/bin/env bash
BACKBONE=$1
CHECKPOINT=$2
INPUT_IMAGE=$3
OUTPUT_IMAGE=$4
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${CHECKPOINT}" ]] && [[ -n "${INPUT_IMAGE}" ]] && [[ -n "${OUTPUT_IMAGE}" ]]); then
    echo "Argument BACKBONE or CHECKPOINT or INPUT_IMAGE or OUTPUT_IMAGE is missing"
    exit
fi

python infer.py -s=voc2007 -b=${BACKBONE} -c=${CHECKPOINT} ${INPUT_IMAGE} ${OUTPUT_IMAGE}