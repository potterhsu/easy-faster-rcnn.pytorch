#!/usr/bin/env bash
BACKBONE=$1
CHECKPOINT=$2
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${CHECKPOINT}" ]]); then
    echo "Argument BACKBONE or CHECKPOINT is missing"
    exit
fi

python eval.py -s=khnp -b=${BACKBONE} ${CHECKPOINT}
python eval.py -s=khnp -b=resnet101 ./outputs/checkpoints-20200527010205-khnp-resnet101-4d3d8602/model-90000.pth
python eval.py -s=khnp -b=resnet101 ./outputs/checkpoints-20200527185123-khnp-resnet101-93fb3faa/model-90000.pth