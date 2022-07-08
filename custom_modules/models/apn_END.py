import torch

from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.localizers import BaseTAGClassifier
from mmaction.core import top_k_accuracy, mean_class_accuracy
from ..apn_utils import binary_accuracy, decode_progression, progression_mae


@LOCALIZERS.register_module()
class APN(BaseTAGClassifier):
    """APN model framework."""

    def __init__(self,
                 backbone,
                 cls_head):
        super(BaseTAGClassifier, self).__init__()
        self.backbone = build_backbone(backbone)
        self.cls_head = build_head(cls_head)
        self.init_weights()

    def _forward(self, imgs):
        # [N, 1, C, T, H, W] -> [N, C, T, H, W]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        output = self.cls_head(x)

        return output

    def forward_train(self, imgs, progression_label=None, class_label=None, end_label=None):
        cls_score, reg_score, end_score = self._forward(imgs)
        class_label, end_label = class_label.squeeze(-1), end_label.squeeze(-1)
        losses = {'loss_cls': self.cls_head.loss_cls(cls_score, class_label),
                  'loss_end': self.cls_head.loss_cls(end_score, end_label)}

        cls_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                 class_label.detach().cpu().numpy(),
                                 topk=(1,))
        end_acc = top_k_accuracy(end_score.detach().cpu().numpy(),
                                 end_label.detach().cpu().numpy(),
                                 topk=(1,))
        losses['cls_acc'] = torch.tensor(cls_acc, device=cls_score.device)
        losses['end_acc'] = torch.tensor(end_acc, device=end_score.device)
        losses['loss_reg'] = self.cls_head.loss_reg(reg_score, progression_label)

        reg_score = reg_score.sigmoid()
        reg_acc = binary_accuracy(reg_score.detach().cpu().numpy(), progression_label.detach().cpu().numpy())
        reg_mae = progression_mae(reg_score.detach().cpu().numpy(), progression_label.detach().cpu().numpy())
        losses[f'reg_acc'] = torch.tensor(reg_acc, device=reg_score.device)
        losses[f'reg_mae'] = torch.tensor(reg_mae, device=reg_score.device)

        return losses

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and testing."""
        cls_score, reg_score, end_score = self._forward(imgs)
        cls_score = cls_score.softmax(-1)
        end_score = end_score.softmax(-1)
        reg_score = reg_score.sigmoid()
        progression = decode_progression(reg_score)
        return list(zip(cls_score.cpu().numpy(), progression.cpu().numpy(), end_score.cpu().numpy()))

