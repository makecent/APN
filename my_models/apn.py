import numpy as np
import torch

from mmcv.runner import load_checkpoint
from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.utils import get_root_logger
from mmaction.models.localizers import BaseTAGClassifier
##
from mmaction.models.backbones.resnet3d_slowfast import ResNet3dSlowFast


def get_correlated_progressions(output, class_label, dim=1):
    """
    Select rows from output with the index of class_label.
    Example:
        output = [[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]]
        class_label = [0, 1]
        indexed_output = [[1, 2],
                          [7, 8]]
    Args:
        output: Tensor. [N, A, S]
        class_label: Tensor. [N] in range of A
        dim: Int. Determine which dim to select on.
    Returns:
        indexed_output: Tensor. [N, S]
    """
    index_shape = list(output.shape)
    index_shape[dim] = 1
    if output.shape[dim] == 1:
        # proposal generation, pseudo indexing for generality.
        class_label = torch.zeros_like(class_label)
    class_label = class_label.view(-1, *(output.dim() - 1) * [1])

    index = class_label.expand(index_shape)
    indexed_output = output.gather(dim, index).squeeze(dim)
    return indexed_output


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
        # [N, num_clips, C, T, H, W] -> [N*num_clips, C, T, H, W], which make clips training parallely (For TSN).
        # For 2D backbone, there is no 'T' dimension. For our APN, num_clips is always equal to 1.
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        if type(self.backbone) is ResNet3dSlowFast:
            x_fast, x_slow = x
            # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
            x_fast = torch.nn.AdaptiveAvgPool3d((1, 1, 1))(x_fast)
            x_slow = torch.nn.AdaptiveAvgPool3d((1, 1, 1))(x_slow)
            # [N, channel_fast + channel_slow, 1, 1, 1]
            x = torch.cat((x_slow, x_fast), dim=1)
        output = self.cls_head(x, num_segs)

        return output

    def compute_loss(self, output, progression_label=None, class_label=None):
        if self.cls_head.output_type == 'action_classification':
            losses = self.cls_head.loss(output, class_label)
        else:
            losses = self.cls_head.loss(output, progression_label, class_label)
        return losses

    def forward_train(self, imgs, progression_label=None, class_label=None):
        """Defines the computation performed at every call when training."""
        # [N, num_clips, C, T, H, W] -> [N*num_clips, C, T, H, W], which make clips training parallelly.
        # For 2D backbone, there is no 'T' dimension.
        output = self._forward(imgs)
        if self.cls_head.output_type == 'two_heads':
            losses = self.compute_loss(output, progression_label, class_label)
        else:
            losses = {'loss': self.compute_loss(output, progression_label, class_label)}

        return losses

    def forward_test(self, imgs, progression_label=None, class_label=None):
        """Defines the computation performed at every call when evaluation and testing."""
        output = self._forward(imgs)
        progressions = self.cls_head.loss.test_output(output)
        if progression_label is not None:
            # in validation stage, we remove uncorrelated progressions (useless for computing MAE) to save memory
            progressions = get_correlated_progressions(progressions, class_label).cpu().numpy()
        return progressions
