#!/usr/bin/env bash
BACKBONE=$1
OUTPUTS_DIR=$2
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${OUTPUTS_DIR}" ]]); then
    echo "Argument BACKBONE or OUTPUTS_DIR is missing"
    exit
fi

python train.py -s=coco2017 -b=${BACKBONE} -o=${OUTPUTS_DIR} --image_min_side=800 --image_max_side=1333 --anchor_sizes="[64, 128, 256, 512]" --anchor_smooth_l1_loss_beta=0.1111 --batch_size=16 --learning_rate=0.02 --weight_decay=0.0001 --step_lr_sizes="[60000, 80000]" --num_steps_to_snapshot=20000 --num_steps_to_finish=90000