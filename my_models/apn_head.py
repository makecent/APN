from abc import ABCMeta

import torch
import torch.nn as nn
from mmaction.models.heads.base import AvgConsensus

from mmcv.cnn import kaiming_init, normal_init, constant_init
from mmaction.models.builder import HEADS, build_loss


class BiasLayer(nn.Module, metaclass=ABCMeta):
    def __init__(self, input_channel, num_bias):
        super(BiasLayer, self).__init__()
        self.input_channel = input_channel
        self.num_bias = num_bias

        self.bias = nn.Parameter(torch.zeros(input_channel, num_bias).float(), requires_grad=True)

    def forward(self, x):
        return x.unsqueeze(-1).repeat_interleave(self.num_bias, dim=-1) + self.bias


@HEADS.register_module()
class APNHead(nn.Module, metaclass=ABCMeta):
    """Regression head for APN.

    Args:
        num_stages (int): Number of stages to be predicted.
        in_channels (int): Number of channels in input feature.
        loss (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes=20,
                 num_stages=100,
                 in_channels=2048,
                 output_type='coral',
                 loss=dict(type='ApnCORALLoss'),
                 spatial_type='avg3d',
                 dropout_ratio=0.5):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.output_type = output_type
        self.loss = build_loss(loss)
        self.num_stages = num_stages
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.consensus = AvgConsensus(dim=1)

        if self.spatial_type == 'avg3d':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif self.spatial_type == 'avg2d':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.output_type == 'regression':
            self.reg_fc = nn.Linear(self.in_channels, self.num_classes)
            self.layers = self.reg_fc
        elif self.output_type == 'classification':  # including cost-sensitive classification
            self.cls_fc = nn.Linear(self.in_channels, self.num_classes * (self.num_stages + 1))
            self.view = nn.Unflatten(dim=-1, unflattened_size=(self.num_classes, self.num_stages + 1))
            self.layers = nn.Sequential(self.cls_fc, self.view)
        elif self.output_type == 'binary_decomposition':
            self.bi_cls_fc = nn.Linear(self.in_channels, self.num_classes * self.num_stages * 2)
            self.view = nn.Unflatten(dim=-1, unflattened_size=(self.num_classes, self.num_stages, 2))
            self.layers = nn.Sequential(self.bi_cls_fc, self.view)
        elif self.output_type == 'coral':
            self.coral_fc = nn.Linear(self.in_channels, self.num_classes, bias=False)
            self.coral_bias = BiasLayer(self.num_classes, self.num_stages)
            self.layers = nn.Sequential(self.coral_fc, self.coral_bias)
        # elif self.output_type == 'two_heads':
        #     prog_cls_fc1 = nn.Linear(self.in_channels, self.in_channels)
        #     prog_cls_fc2 = nn.Linear(self.in_channels, self.num_stages + 1)
        #     action_cls_fc1 = nn.Linear(self.in_channels, self.in_channels)
        #     action_cls_fc2 = nn.Linear(self.in_channels, self.num_classes)
        # elif self.output_type == 'action_classification':
        #     # 'action_classification' does "action classification" only (without progressions).
        #     self.layers = nn.Linear(self.in_channels, self.num_classes)
        else:
            raise ValueError(f"output_type: {self.output_type} not allowed")

    def init_weights(self):
        """Initiate the parameters from scratch."""
        if self.output_type == 'coral':
            kaiming_init(self.coral_fc, a=0, nonlinearity='relu', distribution='uniform')
            constant_init(self.coral_bias, 0)
        else:
            normal_init(self.layers, std=0.01)

    def forward(self, x, num_segs=1):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [M, C, T, H, W] for 3D backbone, or [M, C, H, W] for 2D backbone. M = batch_size(N) * num_clips.
        x = self.avg_pool(x)
        # [N, C, 1, 1, 1] or [N, C, 1, 1]
        x = self.tsn(x, num_segs)  # most cases no effect
        # [N, 1, C, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, C]
        if self.dropout is not None:
            x = self.dropout(x)
        score = self.layers(x)
        return score

    def tsn(self, x, num_segs):
        # [N*num_segs, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        return x
