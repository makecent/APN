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
                 blending=False):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.cls_head = build_head(cls_head)
        self.init_weights()
        self.fp16_enabled = False
        if blending:
            from mmcv.utils import build_from_cfg
            from mmaction.datasets.builder import BLENDINGS
            self.blending = build_from_cfg(blending, BLENDINGS)

    def init_weights(self):
        """Weight initialization for model."""
        self.backbone.init_weights()
        self.cls_head.init_weights()

    @auto_fp16()
    def _forward(self, imgs):
        # [N, 1, C, T, H, W] -> [N, C, T, H, W]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        x = self.backbone(imgs)
        output = self.cls_head(x)

        return output

    def forward_train(self, imgs, progression_label=None, class_label=None):
        hard_class_label = class_label

        if self.blending is not None:
            imgs, class_label, progression_label = self.blending(imgs, class_label, progression_label)

        if progression_label.ndim < 3:
            ordinal_label = torch.zeros(imgs.shape[0], self.cls_head.num_stages).type_as(progression_label)
            denormalized_prog = (progression_label * self.cls_head.num_stages).round().int()
            for i, prog in enumerate(denormalized_prog):
                ordinal_label[i, :prog] = 1.0
            progression_label = ordinal_label

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

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and testing."""
        cls_score, reg_score = self._forward(imgs)
        cls_score = cls_score.softmax(-1)
        reg_score = reg_score.sigmoid()
        progression = decode_progression(reg_score)
        return list(zip(cls_score.detach().cpu().numpy(), progression.detach().cpu().numpy()))

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
