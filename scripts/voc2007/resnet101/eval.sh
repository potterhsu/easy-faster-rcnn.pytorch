#!/usr/bin/env bash
CHECKPOINT=$1
if ! [[ -n "${CHECKPOINT}" ]]; then
    echo "Argument CHECKPOINT is missing"
    exit
fi

python eval.py -s=voc2007 -b=resnet101 ${CHECKPOINT}