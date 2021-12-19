#!/usr/bin/env bash
RGB_CONFIG=$1
FLOW_CONFIG=$3
GPU="${5:-2}"

RGB_DIR=$(echo $RGB_CONFIG | cut -d'.' -f 1)
RGB_DIR=$(echo $RGB_DIR | cut -d'/' -f 2)
RGB_CHECKPOINT=${2:-"work_dirs/$RGB_DIR/latest.pth" }
RGB_RESULT=${2:-"work_dirs/$RGB_DIR/results.pkl" }

FLOW_DIR=$(echo $FLOW_CONFIG | cut -d'.' -f 1)
FLOW_DIR=$(echo $FLOW_DIR | cut -d'/' -f 2)
FLOW_CHECKPOINT=${4:-"work_dirs/$FLOW_DIR/latest.pth" }
FLOW_RESULT=${2:-"work_dirs/$FLOW_DIR/results.pkl" }

echo "Testing on RGB checkpoint: ..."
PYTHONPATH=$PWD:$PYTHONPATH mim test mmaction $RGB_CONFIG --gpus $GPU --launcher pytorch --checkpoint $RGB_CHECKPOINT --out "results.pkl" --eval mAP ${@:6}
echo "RGB inference finished"
echo "Testing on FLOW checkpoint: ..."
PYTHONPATH=$PWD:$PYTHONPATH mim test mmaction $FLOW_CONFIG --gpus $GPU --launcher pytorch --checkpoint $FLOW_CHECKPOINT --out "results.pkl" --eval mAP ${@:6}
echo "FLOW inference finished"
python utils/evaluate_two_stream.py --rgb_result $RGB_RESULT --flow-result $FLOW_RESULT