#!/usr/bin/env bash
CONFIG=$1
GPU=${2:-2}

PYTHONPATH=$PWD:$PYTHONPATH mim train mmaction $CONFIG --gpus $GPU --launcher pytorch ${@:3} --validate