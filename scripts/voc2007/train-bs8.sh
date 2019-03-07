#!/usr/bin/env bash
BACKBONE=$1
OUTPUTS_DIR=$2
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${OUTPUTS_DIR}" ]]); then
    echo "Argument BACKBONE or OUTPUTS_DIR is missing"
    exit
fi

python train.py -s=voc2007 -b=${BACKBONE} -o=${OUTPUTS_DIR} --batch_size=8 --learning_rate=0.008 --step_lr_sizes="[6250, 8750]" --num_steps_to_snapshot=1250 --num_steps_to_finish=11250