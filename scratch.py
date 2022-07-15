# %% fvcore
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
from mmcv import Config
from mmaction.models.builder import build_model, build_backbone
# torch.backends.cudnn.benchmark=True
inputs = torch.randn(1, 1, 3, 16, 224, 224).cuda()
# model = build_backbone(Config.fromfile(
#     "configs/apn_r3dsony_8x32_10e_aty13_video.py").model.backbone).cuda()

model = build_model(Config.fromfile(
    "configs/apn_slowonly_unregl_16x8_10e_thumos14_flow.py").model).cuda()

# flops = FlopCountAnalysis(model.backbone, inputs)
# # print(flop_count_table(flops, max_depth=10))
# params = parameter_count(model.backbone)
#
# print(f"GFLOPS:\t{flops.total()/1e9:.2f} G")
# print(f"Params:\t{params['']/1e6:.2f} M")

from mmcv import track_iter_progress
for i in track_iter_progress(list(range(10000))):
    # loss = model(inputs, class_label=torch.randint(20, (1, 1)).cuda(), return_loss=True)
    # loss['loss_cls'].backward()

    with torch.no_grad():
        out = model(inputs, return_loss=False)