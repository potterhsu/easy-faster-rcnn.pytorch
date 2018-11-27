#!/usr/bin/env bash
CHECKPOINT=$1
python eval.py -s=voc2007 -b=vgg16 --pooling_mode=align ${CHECKPOINT}