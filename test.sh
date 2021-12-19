#!/usr/bin/env bash
CONFIG=$1
DIR=$(echo $CONFIG | cut -d'.' -f 1)
DIR=$(echo $DIR | rev | cut -d'/' -f 1 | rev)
CHECKPOINT=${2:-"work_dirs/$DIR/latest.pth"}
GPU="${3:-2}"
OUT=${4:-"work_dirs/$DIR/progressions.pkl"}

PYTHONPATH=$PWD:$PYTHONPATH mim test mmaction $CONFIG --checkpoint $CHECKPOINT --out $OUT --eval mAP --gpus $GPU --launcher pytorch ${@:5}