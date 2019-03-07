#!/usr/bin/env bash
BACKBONE=$1
OUTPUTS_DIR=$2
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${OUTPUTS_DIR}" ]]); then
    echo "Argument BACKBONE or OUTPUTS_DIR is missing"
    exit
fi

python train.py -s=voc2007 -b=${BACKBONE} -o=${OUTPUTS_DIR} --batch_size=4 --learning_rate=0.004 --step_lr_sizes="[12500, 17500]" --num_steps_to_snapshot=2500 --num_steps_to_finish=22500