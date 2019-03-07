#!/usr/bin/env bash
BACKBONE=$1
CHECKPOINT=$2
INPUT_IMAGE=$3
OUTPUT_IMAGE=$4
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${CHECKPOINT}" ]] && [[ -n "${INPUT_IMAGE}" ]] && [[ -n "${OUTPUT_IMAGE}" ]]); then
    echo "Argument BACKBONE or CHECKPOINT or INPUT_IMAGE or OUTPUT_IMAGE is missing"
    exit
fi

python infer.py -s=coco2017 -b=${BACKBONE} -c=${CHECKPOINT} --image_min_side=800 --image_max_side=1333 --anchor_sizes="[64, 128, 256, 512]" --rpn_post_nms_top_n=1000 ${INPUT_IMAGE} ${OUTPUT_IMAGE}