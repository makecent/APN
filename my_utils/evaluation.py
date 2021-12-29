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
    parser.add_argument('--get-mae', action='store_true', help="whether to compute MAE for the progressions")
    args = parser.parse_args()
    return args


def evaluate_results(cfg_file="configs/localization/apn/apn_coral+random_r3dsony_16x1_10e_thumos14_flow.py",
                     results_file='',
                     compute_MAE=False,
                     metric_options_override=dict(mAP=dict(
                         dump_detections=None,
                         dump_evaluation=None))):
    # Init
    run_name = cfg_file.split('/')[-1].split('.')[0]
    if results_file == '':
        results_file = f"work_dirs/{run_name}/progressions.pkl"
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

    if compute_MAE:
        # Compute MAE
        warnings.warn(
            "The below MAE is computed based on the derived results, which may come from a down-sampled datasets. \
            For an accurate MAE, pls refer to the 'validation MAE' occurs in the training process")
        MAE, pv_trimmed = ds.get_MAE_on_untrimmed_results(results, return_pv=True)
        pv = np.var(results, axis=-1).mean()
        test_sampling = ds.test_sampling
        print(f"MAE({test_sampling}):        {MAE:.2f}")
        print(f"PV of All Frames:            {pv:.1f}")
        print(f"PV of Action Frames:         {pv_trimmed:.1f}")
        eval_results[f'MAE({test_sampling})'] = MAE
        eval_results['pv_of_all_frames'] = pv
        eval_results['pv_of_action_frames'] = pv_trimmed

    # Compute mAP
    metric_options = cfg.eval_config.metric_options
    if metric_options_override:
        metric_options.update(metric_options_override)
    mAP = ds.evaluate(results,
                      metrics='mAP',
                      metric_options=metric_options)
    eval_results.update(mAP)
    execution_time = time.time() - before
    print(f"Execution Time:               {execution_time} seconds")


if __name__ == '__main__':
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    args = parse_args()
    evaluate_results(args.config, args.result, args.get_mae)
