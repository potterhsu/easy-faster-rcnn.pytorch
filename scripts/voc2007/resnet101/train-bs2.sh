#!/usr/bin/env bash
OUTPUTS_DIR=$1
if ! [[ -n "${OUTPUTS_DIR}" ]]; then
    echo "Argument OUTPUTS_DIR is missing"
    exit
fi

python train.py -s=voc2007 -b=resnet101 -o=${OUTPUTS_DIR} --batch_size=2 --learning_rate=0.002 --step_lr_sizes="[25000]" --num_steps_to_snapshot=5000 --num_steps_to_finish=35000