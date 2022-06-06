from mmcv import Config
from mmaction.datasets.builder import build_dataset, build_dataloader

cfg = Config.fromfile("configs/apn_r3dsony_bg0.5_32x4_10e_thumos14_flow.py")
dataset = build_dataset(cfg.data.train, dict(test_mode=False))
print(len(dataset))
dataloader_setting = dict(
    videos_per_gpu=16*8,
    workers_per_gpu=cfg.data.get('workers_per_gpu', 6),
    shuffle=False)
dataloader_setting = dict(dataloader_setting,
                          **cfg.data.get('val_dataloader', {}))
val_dataloader = build_dataloader(dataset, **dataloader_setting)
print(len(val_dataloader))