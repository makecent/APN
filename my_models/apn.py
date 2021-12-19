import numpy as np
import torch

from mmcv.runner import load_checkpoint
from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.utils import get_root_logger
from mmaction.models.localizers import BaseTAGClassifier
##
from mmaction.models.backbones.resnet3d_slowfast import ResNet3dSlowFast


@LOCALIZERS.register_module()
class APN(BaseTAGClassifier):
    """APN model framework."""

    def __init__(self,
                 backbone,
                 cls_head,
                 pretrained=None):
        super(BaseTAGClassifier, self).__init__()
        self.backbone = build_backbone(backbone)
        self.cls_head = build_head(cls_head)
        self.pretrained = pretrained
        self.init_weights()

    def _forward(self, imgs):
        # [N, num_clips, C, T, H, W] -> [N*num_clips, C, T, H, W], which make clips training parallelly.
        # For 2D backbone, there is no 'T' dimension.
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
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

    def forward_validate(self, imgs, progression_label=None, class_label=None):
        """Defines the computation performed at every call when evaluation and testing."""
        output = self._forward(imgs)
        loss = self.compute_loss(output, progression_label, class_label).cpu().numpy()
        progressions = self.cls_head.loss.val_output(output, class_label)
        return list(zip(progressions, np.tile(loss, reps=output.shape[0])))

    def forward_test(self, imgs, progression_label=None, class_label=None):
        """Defines the computation performed at every call when evaluation and testing."""
        output = self._forward(imgs)
        progressions = self.cls_head.loss.test_output(output)
        return progressions

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        outs = (self.cls_head(x),)
        return outs

    def forward(self,
                imgs,
                progression_label=None,
                class_label=None,
                return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(imgs, progression_label, class_label)
        elif progression_label:
            return self.forward_validate(imgs, progression_label, class_label)
        else:
            return self.forward_test(imgs)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        self.backbone.init_weights()
        self.cls_head.init_weights()

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif not self.pretrained:
            pass
        else:
            raise TypeError('pretrained must be a str or None')
