from mmaction.datasets import build_dataset
from mmcv import Config, load
import torch
from matplotlib import pyplot as plt
import numpy as np
cfg = Config.fromfile("configs/apn_mvit2_32x4_10e_thumos14_rgb.py")
untrimmed_ds = build_dataset(cfg.data.test)
cfg.data.test.untrimmed = False
trimmed_ds = build_dataset(cfg.data.test)

trimmed_rs = load("/home/louis/PycharmProjects/APN/trimmed_result.pkl")
untrimmed_rs = load("/home/louis/PycharmProjects/APN/untrimmed_1000_result.pkl")

# Trimmed
trimmed_pr = np.array([i[1] for i in trimmed_rs])
trimmed_action_frames = [e['frame_dir'].rsplit('/', 1)[-1] + f"/{e['frame_index']:05}" for e in trimmed_ds.frame_infos]
trimmed_gt_prog = [e['progression_label']*100 for e in trimmed_ds.frame_infos]
trimmed_MAE = np.abs(trimmed_pr - np.array(trimmed_gt_prog))
print(trimmed_MAE.mean())
trimmed_summary = {f: {'pr': p, 'gt': g, 'mae': m} for f, p, g, m in zip(trimmed_action_frames, trimmed_pr, trimmed_gt_prog, trimmed_MAE)}

# Untrimmed
sampled_idx_pre, _, untrimmed_gt_prog = untrimmed_ds.get_sample_points_on_untrimmed(return_gt_progs=True)

untrimmed_pr = np.array([i[1] for i in untrimmed_rs])
untrimmed_pr = untrimmed_pr[sampled_idx_pre]
untrimmed_action_frames = np.array([e['frame_dir'].rsplit('/', 1)[-1] + f"/{e['frame_index']:05}" for e in untrimmed_ds.frame_infos])
untrimmed_action_frames = untrimmed_action_frames[sampled_idx_pre]

untrimmed_MAE = np.abs(untrimmed_pr - np.array(untrimmed_gt_prog))
print(untrimmed_MAE.mean())
untrimmed_summary = {f: {'pr': p, 'gt': g, 'mae': m} for f, p, g, m in zip(untrimmed_action_frames, untrimmed_pr, untrimmed_gt_prog, untrimmed_MAE)}

sampled_trimmed_mae = []
for action_frame, untrimmed_info in untrimmed_summary.items():
    trimmed_info = trimmed_summary[action_frame]
    if np.abs(trimmed_info['mae'] - untrimmed_info['mae']) > 0.0001:
        print(np.abs(trimmed_info['mae'] - untrimmed_info['mae']))
    if np.abs(trimmed_info['pr'] - untrimmed_info['pr']) > 0.0001:
        print(np.abs(trimmed_info['pr'] - untrimmed_info['pr']))
    if np.abs(trimmed_info['gt'] - untrimmed_info['gt']) > 0.0001:
        print(np.abs(trimmed_info['gt'] - untrimmed_info['gt']))
    sampled_trimmed_mae.append(trimmed_info['mae'])

sampled_trimmed_mae = np.array(sampled_trimmed_mae)
print(sampled_trimmed_mae.mean())
print('The End')
