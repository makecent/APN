import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import CenterCrop
import numpy as np

from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.recognizers import BaseRecognizer
from mmaction.core import top_k_accuracy
from ..apn_utils import binary_accuracy, decode_progression, progression_mae
from mmcv.runner import auto_fp16, load_checkpoint
from mmflow.models.builder import build_flow_estimator
from mmcv.video import quantize_flow

@LOCALIZERS.register_module()
class APN_fly(nn.Module):
    """APN model framework."""

    def __init__(self,
                 flow_est,
                 backbone,
                 cls_head,
                 blending=None):
        super().__init__()

        self.flow_est = build_flow_estimator(flow_est)
        self.flow_est.eval()
        load_checkpoint(self.flow_est, 'checkpoints/mmflow/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.pth', map_location='cpu')
        for p in self.flow_est.parameters():
            p.requires_grad = False
        self.center_crop = CenterCrop(224)

        self.backbone = build_backbone(backbone)
        self.cls_head = build_head(cls_head)
        self.init_weights()
        self.fp16_enabled = False
        self.blending = blending
        if blending:
            from mmcv.utils import build_from_cfg
            from mmaction.datasets.builder import BLENDINGS
            self.blending = build_from_cfg(blending, BLENDINGS)

    def init_weights(self):
        """Weight initialization for model."""
        self.backbone.init_weights()
        self.cls_head.init_weights()

    def _forward(self, imgs):
        x = self.backbone(imgs)
        output = self.cls_head(x)

        return output

    @auto_fp16()
    def forward_train(self, imgs, progression_label=None, class_label=None):
        hard_class_label = class_label
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        imgs = self.estimate_flow(imgs)

        if self.blending is not None:
            imgs, class_label, progression_label = self.blending(imgs, class_label, progression_label)

        cls_score, reg_score = self._forward(imgs)

        class_label = class_label.squeeze(1)
        hard_class_label = hard_class_label.squeeze(1)
        losses = {'loss_cls': self.cls_head.loss_cls(cls_score, class_label)}
        if not self.blending:
            cls_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                     hard_class_label.detach().cpu().numpy(),
                                     topk=(1,))
            losses[f'cls_acc'] = torch.tensor(cls_acc, device=cls_score.device)

        losses['loss_reg'] = self.cls_head.loss_reg(reg_score, progression_label)
        reg_score = reg_score.sigmoid()
        reg_acc = binary_accuracy(reg_score.detach().cpu().numpy(), progression_label.detach().cpu().numpy())
        reg_mae = progression_mae(reg_score.detach().cpu().numpy(), progression_label.detach().cpu().numpy())
        losses[f'reg_acc'] = torch.tensor(reg_acc, device=reg_score.device)
        losses[f'reg_mae'] = torch.tensor(reg_mae, device=reg_score.device)

        return losses

    def forward_test(self, imgs, raw_progression=None):
        """Defines the computation performed at every call when evaluation and testing."""
        batch_size, num_segs = imgs.shape[:2]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        imgs = self.estimate_flow(imgs)
        cls_score, reg_score = self._forward(imgs)

        cls_score = cls_score.softmax(-1)
        reg_score = reg_score.sigmoid()
        if num_segs > 1:
            cls_score = cls_score.view(batch_size, num_segs, -1).mean(dim=1)
            reg_score = reg_score.view(batch_size, num_segs, -1).mean(dim=1)
        progression = decode_progression(reg_score)
        if raw_progression is not None:
            prog_mae = torch.abs(progression - raw_progression.squeeze(1))
            return list(zip(cls_score.detach().cpu().numpy(), prog_mae.detach().cpu().numpy()))
        else:
            return list(zip(cls_score.detach().cpu().numpy(), progression.detach().cpu().numpy()))

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
        imgs = (imgs - 127.5) / 127.5
        return imgs

    def forward(self, *args, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(*args, **kwargs)

        return self.forward_test(*args, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        losses = self.forward(**data_batch)
        loss, log_vars = BaseRecognizer._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        results = self.forward(return_loss=False, **data_batch)
        outputs = dict(results=results)

        return outputs
