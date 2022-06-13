import torch
from collections import namedtuple

from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.localizers import BaseTAGClassifier
from mmaction.core import top_k_accuracy, mean_class_accuracy
from ..apn_utils import binary_accuracy, decode_progression, progression_mae


# def get_correlated_progressions(output, class_label, dim=1):
#     """
#     Select rows from output with the index of class_label.
#     Example:
#         output = [[[1, 2],
#                    [3, 4]],
#                   [[5, 6],
#                    [7, 8]]]
#         class_label = [0, 1]
#         indexed_output = [[1, 2],
#                           [7, 8]]
#     Args:
#         output: Tensor. [N, A, S]
#         class_label: Tensor. [N] in range of A
#         dim: Int. Determine which dim to select on.
#     Returns:
#         indexed_output: Tensor. [N, S]
#     """
#     index_shape = list(output.shape)
#     index_shape[dim] = 1
#     class_label = class_label.view(-1, *(output.dim() - 1) * [1])
#
#     index = class_label.expand(index_shape)
#     indexed_output = output.gather(dim, index).squeeze(dim)
#     return indexed_output


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

    def forward_train(self, imgs, progression_label=None, class_label=None):
        class_label = class_label.squeeze(-1)
        cls_score, reg_score = self._forward(imgs)
        losses = {'loss_cls': self.cls_head.loss_cls(cls_score, class_label)}

        cls_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                 class_label.detach().cpu().numpy(),
                                 topk=(1,))
        losses[f'cls_acc'] = torch.tensor(cls_acc, device=cls_score.device)

        foreground = ~(class_label == self.cls_head.num_classes)
        reg_score, progression_label = reg_score[foreground], progression_label[foreground]

        if progression_label.numel() > 0:
            losses['loss_reg'] = self.cls_head.loss_reg(reg_score, progression_label)
            reg_score = reg_score.sigmoid()
            reg_acc = binary_accuracy(reg_score.detach().cpu().numpy(), progression_label.detach().cpu().numpy())
            reg_mae = progression_mae(reg_score.detach().cpu().numpy(), progression_label.detach().cpu().numpy())
            losses[f'reg_acc'] = torch.tensor(reg_acc, device=reg_score.device)
            losses[f'reg_mae'] = torch.tensor(reg_mae, device=reg_score.device)
        # else:
        #     losses['loss_reg'] = reg_score * 0

        return losses

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and testing."""
        cls_score, reg_score = self._forward(imgs)
        cls_score = cls_score.softmax(dim=-1)
        reg_score = reg_score.sigmoid()
        progression = decode_progression(reg_score)
        return list(zip(cls_score.cpu().numpy(), progression.cpu().numpy()))

