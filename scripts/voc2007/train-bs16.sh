#!/usr/bin/env bash
BACKBONE=$1
OUTPUTS_DIR=$2
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${OUTPUTS_DIR}" ]]); then
    echo "Argument BACKBONE or OUTPUTS_DIR is missing"
    exit
fi

python train.py -s=voc2007 -b=${BACKBONE} -o=${OUTPUTS_DIR} --batch_size=16 --learning_rate=0.016 --step_lr_sizes="[3125, 4375]" --num_steps_to_snapshot=625 --num_steps_to_finish=5625