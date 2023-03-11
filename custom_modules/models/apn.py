import torch
from torch import nn
from torch.nn import functional as F

from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.recognizers import BaseRecognizer
from mmaction.core import top_k_accuracy
from ..apn_utils import binary_accuracy, progression_mae
from .apn_head import *
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
        self.cls_head = build_head(cls_head)  # just for compatibility, both cls and loc heads are in it.
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

        progression_label = progression_label.squeeze(1)
        losses['loss_reg'] = self.cls_head.loss_reg(reg_score, progression_label)
        self.training_metric(losses, reg_score, progression_label)

        return losses

    def training_metric(self, losses, reg_score, progression_label):
        if isinstance(self.cls_head, APNHead):
            reg_acc = torch.count_nonzero((reg_score > 0) == progression_label) / progression_label.numel()
            losses[f'reg_acc'] = reg_acc.detach().float()
            raw_prog_label = torch.count_nonzero(progression_label > 0.5, dim=-1) / progression_label.size(-1) * 100
        elif isinstance(self.cls_head, APNClsHead):
            reg_acc = torch.count_nonzero(reg_score.argmax(dim=-1) == progression_label.argmax(dim=-1)) / \
                      reg_score.shape[0]
            losses[f'reg_acc'] = reg_acc.detach().float()
            raw_prog_label = torch.argmax(progression_label, dim=-1).float() / (progression_label.size(-1) - 1) * 100
        elif isinstance(self.cls_head, APNRegHead):
            raw_prog_label = progression_label * 100
        elif isinstance(self.cls_head, APNDecHead):
            reg_acc = torch.count_nonzero(reg_score.argmax(dim=1) == progression_label) / progression_label.numel()
            losses[f'reg_acc'] = reg_acc.detach().float()
            raw_prog_label = torch.count_nonzero(progression_label > 0.5, dim=-1) / progression_label.size(-1) * 100
        else:
            raise TypeError

        prog_mae = torch.abs(raw_prog_label - self.decode_progression(reg_score))
        losses[f'reg_mae'] = prog_mae.detach()

    def forward_test(self, imgs, raw_progression=None):
        """Defines the computation performed at every call when evaluation and testing."""
        batch_size, num_segs = imgs.shape[:2]
        cls_score, reg_score = self._forward(imgs)
        if num_segs > 1:
            cls_score = cls_score.unflatten(0, (batch_size, num_segs)).mean(dim=1)
            reg_score = reg_score.unflatten(0, (batch_size, num_segs)).mean(dim=1)

        cls_score = cls_score.softmax(-1)
        progression = self.decode_progression(reg_score)

        if raw_progression is not None:
            prog_mae = torch.abs(progression - raw_progression.squeeze(1))
            return list(zip(cls_score.detach().cpu().numpy(), prog_mae.detach().cpu().numpy()))
        else:
            return list(zip(cls_score.detach().cpu().numpy(), progression.detach().cpu().numpy()))

    def decode_progression(self, reg_score):
        if isinstance(self.cls_head, APNHead):
            reg_score = reg_score.sigmoid()
            progression = torch.count_nonzero(reg_score > 0.5, dim=-1)
            progression = progression / reg_score.size(-1) * 100
        elif isinstance(self.cls_head, APNClsHead):
            reg_score = reg_score.softmax(dim=-1)
            # # argmax
            # progression = torch.argmax(reg_score, dim=-1).float()
            # progression = progression / reg_score.size(-1) * 100
            # expectation
            progression = (reg_score * torch.arange(0, reg_score.size(-1)).type_as(reg_score)).sum(dim=-1)
            progression = progression / reg_score.size(-1) * 100
        elif isinstance(self.cls_head, APNRegHead):
            progression = reg_score.sigmoid().squeeze(dim=-1) * 100
        elif isinstance(self.cls_head, APNDecHead):
            reg_score = reg_score.softmax(dim=-2).transpose(-1, -2)[..., 1]
            progression = torch.count_nonzero(reg_score > 0.5, dim=-1)
            progression = progression / reg_score.size(-1) * 100
        else:
            raise TypeError(f"unsupported apn head: {type(self.cls_head)}")
        return progression

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


@LOCALIZERS.register_module()
class APNGuided(APN):
    @auto_fp16()
    def forward_train(self, *args, **kwargs):
        losses = APN.forward_train(*args, **kwargs)
        reg_loss, cls_loss = losses['loss_reg'].clone().detach(), losses['loss_cls'].clone().detach()
        g_factor = cls_loss / reg_loss
        losses['loss_reg'] *= g_factor

        return losses
