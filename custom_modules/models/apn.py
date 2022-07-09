import torch

from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.recognizers import BaseRecognizer
from mmaction.core import top_k_accuracy, mean_class_accuracy
from ..apn_utils import binary_accuracy, decode_progression, progression_mae


@LOCALIZERS.register_module()
class APN(BaseRecognizer):
    """APN model framework."""

    def __init__(self,
                 backbone,
                 cls_head):
        super(BaseRecognizer, self).__init__()
        self.backbone = build_backbone(backbone)
        self.backbone_from = 'mmaction2'
        self.cls_head = build_head(cls_head)
        self.init_weights()

    def _forward(self, imgs):
        # [N, 1, C, T, H, W] -> [N, C, T, H, W]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        output = self.cls_head(x)

        return output

    def forward_train(self, imgs, progression_label=None, class_label=None):
        cls_score, reg_score = self._forward(imgs)
        class_label = class_label.squeeze(-1)
        losses = {'loss_cls': self.cls_head.loss_cls(cls_score, class_label)}

        cls_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                 class_label.detach().cpu().numpy(),
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
        return list(zip(cls_score.cpu().numpy(), progression.cpu().numpy()))

    def forward_gradcam(self, imgs):
        """Unused, just complete the abstract function of father class"""
        assert self.with_cls_head
        return self._do_test(imgs)
