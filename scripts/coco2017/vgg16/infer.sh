#!/usr/bin/env bash
CHECKPOINT=$1
INPUT_IMAGE=$2
OUTPUT_IMAGE=$3
python infer.py -c=${CHECKPOINT} -s=coco2017 -b=vgg16 \
               --image_min_side=800 --image_max_side=1333 --anchor_sizes="[64, 128, 258, 512]" \
               --pooling_mode=align --rpn_post_nms_top_n=1000 \
               ${INPUT_IMAGE} ${OUTPUT_IMAGE}