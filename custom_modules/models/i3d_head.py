# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init

from mmaction.models.builder import HEADS
from mmaction.models.heads.base import BaseHead
from mmcv.runner import load_checkpoint


@HEADS.register_module(force=True)
class I3DHead(BaseHead):
    """
    A modified I3DHead:
        1. self.fc_cls is renamed to self.cls_fc
        2. support head-only initialization
    for easy initialization with the MViT pretrained weights
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 pretrained=False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.cls_fc = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None
        self.pretrained = pretrained

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.cls_fc, std=self.init_std)
        if self.pretrained:
            load_checkpoint(self, self.pretrained)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.cls_fc(x)
        # [N, num_classes]
        return cls_score
