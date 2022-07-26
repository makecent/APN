import argparse
from mmcv import Config, load, DictAction
from mmaction.datasets import build_dataset
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation on predicted action progressions")
    parser.add_argument('config', default='', help="path to the test config .py file")
    parser.add_argument('result', default='', help="path to the predicted progressions .pkl file.")
    parser.add_argument('--metric-options', nargs='+', action=DictAction, default={})
    args = parser.parse_args()
    return args


def evaluate_results(cfg_file="configs/apn_mvit_16x8_10e_thumos14_rgb.py",
                     results_file='results.pkl',
                     cmd_metric_options={}):
    # Init
    run_name = cfg_file.split('/')[-1].rsplit('.', 1)[0]
    if results_file == '':
        results_file = f"work_dirs/{run_name}/progressions.pkl"
    results = load(results_file)
    cfg = Config.fromfile(cfg_file)
    ds = build_dataset(cfg.data.test)

    # Compute mAP
    # metric_options = cfg.eval_config.metric_options
    metric_config = Config(dict(
                     mAP=dict(
                         iou_thr=0.5,
                         search=dict(
                             min_e=60,
                             max_s=40,
                             min_L=60,
                             method='mse'),
                         nms=dict(
                             score_thr=0,
                             max_per_video=-1,
                             nms=dict(iou_thr=0.4)))))
    cfg_metric_options = cfg.get('eval_config', {}).get('metric_options', {})
    if cfg_metric_options:
        metric_config.merge_from_dict(cfg_metric_options)
    if cmd_metric_options:
        metric_config.merge_from_dict(cmd_metric_options)
    aps = ds.evaluate(results, metrics=['mAP'], metric_options=metric_config.to_dict())


if __name__ == '__main__':
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(1)

    args = parse_args()
    evaluate_results(args.config, args.result, args.metric_options)
