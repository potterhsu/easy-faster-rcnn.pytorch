#!/usr/bin/env bash
BACKBONE=$1
OUTPUTS_DIR=$2
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${OUTPUTS_DIR}" ]]); then
    echo "Argument BACKBONE or OUTPUTS_DIR is missing"
    exit
fi

python train.py -s=voc2007 -b=${BACKBONE} -o=${OUTPUTS_DIR} --batch_size=1 --learning_rate=0.001 --step_lr_sizes="[50000, 70000]" --num_steps_to_snapshot=10000 --num_steps_to_finish=90000