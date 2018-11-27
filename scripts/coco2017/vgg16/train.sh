#!/usr/bin/env bash
python train.py -s=coco2017 -b=vgg16 \
                --image_min_side=800 --image_max_side=1333 --anchor_sizes="[64, 128, 258, 512]" \
                --pooling_mode=align --weight_decay=0.0001 --step_lr_size=900000 \
                --num_steps_to_snapshot=100000 --num_steps_to_finish=1200000