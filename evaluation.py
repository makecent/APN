import argparse
from mmcv import Config, load
from mmaction.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation on predicted action progressions")
    parser.add_argument('--config', default='', help="path to the test config .py file")
    parser.add_argument('--result', default='', help="path to the predicted progressions .pkl file.")
    args = parser.parse_args()
    return args


def evaluate_results(cfg_file="configs/apn_mvit_16x8_10e_thumos14_rgb.py",
                     results_file='results.pkl'):
    # Init
    run_name = cfg_file.split('/')[-1].rsplit('.', 1)[0]
    if results_file == '':
        results_file = f"work_dirs/{run_name}/progressions.pkl"
    results = load(results_file)
    cfg = Config.fromfile(cfg_file)
    ds = build_dataset(cfg.data.test)

    # Compute mAP
    # metric_options = cfg.eval_config.metric_options
    aps = ds.evaluate(results, metrics=['mAP'])


if __name__ == '__main__':
    args = parse_args()
    evaluate_results(args.config, args.result)
