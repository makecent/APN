import torch

from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.localizers import BaseTAGClassifier


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
        # [N, 1, C, T, H, W] -> [N, C, T, H, W]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        output = self.cls_head(x)

        return output

    def forward_train(self, imgs, progression_label=None, class_label=None):
        output = self._forward(imgs)
        losses = {'loss': self.cls_head.loss(output, progression_label, class_label)}

        return losses

    def forward_test(self, imgs, progression_label=None, class_label=None):
        """Defines the computation performed at every call when evaluation and testing."""
        output = self._forward(imgs)
        output = torch.sigmoid(output)

        # decode output to progressions
        num_stage = output.shape[2]
        progressions = torch.count_nonzero(output > 0.5, dim=-1)
        progressions = progressions * 100 / num_stage

        if progression_label is not None:
            # in validation stage, we remove uncorrelated progressions (useless for computing MAE) to save memory
            progressions = get_correlated_progressions(progressions, class_label)
        return progressions.cpu().numpy()
