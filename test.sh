#!/usr/bin/env bash
CONFIG=$1
DIR=$(echo $CONFIG | cut -d'.' -f 1)
DIR=$(echo $DIR | rev | cut -d'/' -f 1 | rev)
CHECKPOINT=${2:-"latest.pth"}
CHECKPOINT="work_dirs/$DIR/$CHECKPOINT"
GPU="${3:-2}"
#OUT=${4:-"work_dirs/$DIR/progressions.pkl"}

PYTHONPATH=$PWD:$PYTHONPATH mim test mmaction $CONFIG --checkpoint $CHECKPOINT --gpus $GPU --eval mAP --launcher pytorch "${@:4}"