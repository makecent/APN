from mmaction.datasets import build_dataset
from mmcv import Config
import torch
from matplotlib import pyplot as plt
import numpy as np
cfg = Config.fromfile("configs/apn_mvit2_K400_32x4_10e_thumos14_in_rgb.py")
cfg.data.train.pipeline = [
    dict(type='SampleActionFrames', clip_len=cfg.clip_len, frame_interval=cfg.frame_interval),
    dict(type='LabelToOrdinal', num_stages=100),
    dict(type='Collect', keys=['progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['progression_label', 'class_label']),
]
ds = build_dataset(cfg.data.train)
prog = [torch.count_nonzero(e['progression_label']).item() for e in ds]
plt.figure()
plt.boxplot(prog)
plt.show()
plt.figure()
plt.hist(prog, bins=np.arange(102))
plt.show()
print('end')