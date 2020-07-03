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
python eval.py -s=cocokhnp -b=resnet101 ./outputs/checkpoints-20200610024751-cocokhnp-resnet101-4e3164a5/model-90000.pth -d "E:\\"
#Outlier Before
python eval.py -s=khnp -b=resnet101 ./paper/checkpoints-20200623164013-khnp-resnet101-702bfb9e/model-90000.pth

python eval.py -s=khnp -b=resnet101 ./paper/checkpoints-20200624034750-khnp-resnet101-4579f547/model-900000.pth

#Outlier After
python eval.py -s=khnp -b=resnet101 ./paper/checkpoints-20200623012407-khnp-resnet101-fd68f2da/model-90000.pth



python eval.py -s=pothole -b=resnet101 ./outputs/checkpoints-20200621222413-pothole-resnet101-41b4bdc0/model-90000.pth -d "E:\\pothole"
