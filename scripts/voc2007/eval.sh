#!/usr/bin/env bash
BACKBONE=$1
CHECKPOINT=$2
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${CHECKPOINT}" ]]); then
    echo "Argument BACKBONE or CHECKPOINT is missing"
    exit
fi

python eval.py -s=voc2007 -b=${BACKBONE} ${CHECKPOINT}