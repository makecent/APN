import torch
from torch import nn
from torchvision.transforms import CenterCrop
import numpy as np

from mmaction.models.builder import LOCALIZERS, build_backbone
from mmcv.runner import auto_fp16, load_checkpoint
from mmflow.models.builder import build_flow_estimator
from mmcv.video import quantize_flow


@LOCALIZERS.register_module()
class Flow_Wrapper(nn.Module):

    def __init__(self,
                 flow_est,
                 backbone,
                 normalize=dict(mean=127.5, std=127.5),
                 center_crop=False):
        super().__init__()

        self.flow_est = build_flow_estimator(flow_est)
        self.flow_est.eval()
        load_checkpoint(self.flow_est, 'https://download.openmmlab.com/mmflow/pwcnet/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.pth', map_location='cpu')
        for p in self.flow_est.parameters():
            p.requires_grad = False
        self.center_crop = CenterCrop(center_crop) if center_crop else nn.Identity()
        self.normalize = normalize

        self.backbone = build_backbone(backbone)
        self.init_weights()
        self.fp16_enabled = False

    def init_weights(self):
        """Weight initialization for model."""
        self.backbone.init_weights()

    @auto_fp16()
    def forward(self, imgs):
        imgs = self.estimate_flow(imgs)
        feat = self.backbone(imgs)
        return feat

    def estimate_flow(self, imgs):
        B, C, T, H, W = imgs.shape
        dtype, device = imgs.dtype, imgs.device
        with torch.no_grad():
            img1 = imgs[:, :, :-1, :, :]
            img2 = imgs[:, :, 1:, :, :]
            imgs = torch.cat([img1, img2], dim=1).transpose(1, 2).flatten(end_dim=1)
            flow = self.flow_est(imgs, test_mode=True)
            imgs = self.flow_pipe(flow, dtype, device)
            imgs = imgs.unflatten(dim=0, sizes=(B, T - 1)).transpose(1, 2)
        return imgs

    def flow_pipe(self, x, dtype, device):
        imgs = np.array([quantize_flow(d['flow']) for d in x])
        imgs = torch.from_numpy(imgs).to(dtype).to(device)
        imgs = self.center_crop(imgs)
        imgs = (imgs - self.normalize['mean']) / self.normalize['std']
        return imgs