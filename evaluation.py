import argparse
import numpy as np
import json
import time
import pickle
from mmcv import Config
from mmaction.datasets import build_dataset
import warnings
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation on predicted action progressions")
    parser.add_argument('--config', default='', help="path to the test config .py file")
    parser.add_argument('--result',
                        default='',
                        help="path to the predicted progressions .pkl file. If not specified, "
                             "then will try to find 'progressions.pkl' in work_dir defined in the config")
    args = parser.parse_args()
    return args


def evaluate_results(cfg_file="configs/apn_r3dsony_bg0.5_32x4_10e_thumos14_flow.py",
                     results_file=''):
    # Init
    run_name = cfg_file.split('/')[-1].rsplit('.', 1)[0]
    if results_file == '':
        results_file = f"work_dirs/{run_name}/results.pkl"
    if '.pkl' in results_file:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    else:
        with open(results_file, 'r') as f:
            results = json.load(f)
    before = time.time()
    eval_results = {}
    cfg = Config.fromfile(cfg_file)
    ds = build_dataset(cfg.data.test)

    # Compute mAP
    # metric_options = cfg.eval_config.metric_options
    aps = ds.evaluate(results, metrics='mAP')
    eval_results.update(aps)
    print(f"AP50:                     {eval_results['AP50']}")
    execution_time = time.time() - before
    print(f"Execution Time:              {execution_time:.2f} seconds")


if __name__ == '__main__':
    args = parse_args()
    evaluate_results(args.config, args.result)
