#!/usr/bin/env bash
BACKBONE=$1
CHECKPOINT=$2
if ! [[ -n "${CHECKPOINT}" ]]; then
    echo "Argument CHECKPOINT is missing"
    exit
fi

python eval.py -s=voc2007 -b=${BACKBONE} ${CHECKPOINT}