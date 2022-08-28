import torch
from torch import nn
from torch.nn import functional as F

from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.recognizers import BaseRecognizer
from mmaction.core import top_k_accuracy
from ..apn_utils import binary_accuracy, decode_progression, progression_mae
from mmcv.runner import auto_fp16


@LOCALIZERS.register_module()
class APN(nn.Module):
    """APN model framework."""

    def __init__(self,
                 backbone,
                 cls_head,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.cls_head = build_head(cls_head)
        self.init_weights()
        self.fp16_enabled = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if train_cfg is not None and 'blending' in train_cfg:
            from mmcv.utils import build_from_cfg
            from mmaction.datasets.builder import BLENDINGS
            self.blending = build_from_cfg(train_cfg['blending'], BLENDINGS)
        else:
            self.blending = None
        if test_cfg is not None and 'feature_extraction' in test_cfg:
            self.feature_extraction = True
        else:
            self.feature_extraction = False

    def init_weights(self):
        """Weight initialization for model."""
        self.backbone.init_weights()
        self.cls_head.init_weights()

    def _forward(self, imgs):
        # [N, 1, C, T, H, W] -> [N, C, T, H, W]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        x = self.backbone(imgs)
        output = self.cls_head(x)

        return output

    @auto_fp16()
    def forward_train(self, imgs, progression_label=None, class_label=None):
        print(imgs.dtype)
        hard_class_label = class_label

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

    def forward(self, *args, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(*args, **kwargs)
        elif self.feature_extraction:
            return self.feature_extract(*args, **kwargs)
        return self.forward_test(*args, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        losses = self.forward(**data_batch)
        loss, log_vars = BaseRecognizer._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
    #
    # def val_step(self, data_batch, optimizer, **kwargs):
    #     results = self.forward(return_loss=False, **data_batch)
    #     outputs = dict(results=results)
    #
    #     return outputs

    def feature_extract(self, imgs, img_metas=None):
        batch_size, num_segs = imgs.shape[:2]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        feat = self.backbone(imgs)
        feat = self.cls_head.avg_pool(feat)
        if num_segs > 1:
            feat = feat.reshape((batch_size, num_segs, -1))
            feat = feat.mean(axis=1)
        return feat.detach().cpu().numpy()

