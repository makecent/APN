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
                 loss=dict(type='ApnCORALLoss', uncorrelated_progs='random'),
                 spatial_type='avg',
                 dropout_ratio=0.5):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss = build_loss(loss)
        self.num_stages = num_stages
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.consensus = AvgConsensus(dim=1)

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.coral_fc = nn.Linear(self.in_channels, self.num_classes, bias=False)
        self.coral_bias = BiasLayer(self.num_classes, self.num_stages)
        self.layers = nn.Sequential(self.coral_fc, self.coral_bias)

    def init_weights(self):
        kaiming_init(self.coral_fc, a=0, nonlinearity='relu', distribution='uniform')
        constant_init(self.coral_bias, 0)

    def forward(self, x):
        # [N, C, T', H', W'] or [N, L, C]
        x = self.avg_pool(x)
        # [N, C, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, C]
        if self.dropout is not None:
            x = self.dropout(x)
        score = self.layers(x)
        return score
