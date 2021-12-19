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
        elif self.output_type == 'classification':
            self.cls_fc = nn.Linear(self.in_channels, self.num_classes * (self.num_stages + 1))
        elif self.output_type == 'binary_classification':
            self.bi_cls_fc = nn.Linear(self.in_channels, self.num_classes * self.num_stages * 2)
        elif self.output_type == 'coral':
            self.coral_fc = nn.Linear(self.in_channels, self.num_classes, bias=False)
            self.coral_bias = BiasLayer(self.num_classes, self.num_stages)
        elif self.output_type == 'two_heads':
            self.prog_cls_fc1 = nn.Linear(self.in_channels, self.in_channels)
            self.prog_cls_fc2 = nn.Linear(self.in_channels, self.num_stages + 1)
            self.action_cls_fc1 = nn.Linear(self.in_channels, self.in_channels)
            self.action_cls_fc2 = nn.Linear(self.in_channels, self.num_classes)

        elif self.output_type == 'action_classification':
            # 'classification' means "stage classification" that do classification on both stages and actions, in
            # contrast to 'action_classification' which do action classification only.
            self.ac_cls_fc = nn.Linear(self.in_channels, self.num_classes)
        else:
            raise ValueError(f"output_type: {self.output_type} not allowed")

    def init_weights(self):
        """Initiate the parameters from scratch."""
        if self.output_type == 'regression':
            normal_init(self.reg_fc, std=0.001)
        elif self.output_type == 'classification':
            normal_init(self.cls_fc, std=0.01)
        elif self.output_type == 'binary_classification':
            normal_init(self.bi_cls_fc, std=0.01)
        elif self.output_type == 'coral':
            kaiming_init(self.coral_fc, a=0, nonlinearity='relu', distribution='uniform')
            constant_init(self.coral_bias, 0)
        elif self.output_type == 'action_classification':
            normal_init(self.ac_cls_fc, std=0.01)
        elif self.output_type == 'two_heads':
            normal_init(self.prog_cls_fc1, std=0.01)
            normal_init(self.prog_cls_fc2, std=0.01)
            normal_init(self.action_cls_fc1, std=0.01)
            normal_init(self.action_cls_fc2, std=0.01)

    def forward(self, x, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [M, C, T, H, W] for 3D backbone, or [M, C, H, W] for 2D backbone. M = batch_size(N) * num_clips.
        x = self.avg_pool(x)
        # [N, C, 1, 1, 1] or [N, C, 1, 1]
        x = self.tsn(x, num_segs)  # this would has no effect in most cases unless tsn is used, i.e. num_clips > 1.
        # [N, 1, C, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, C]
        if self.dropout is not None:
            x = self.dropout(x)

        if self.output_type == 'regression':
            score = self.reg_fc(x)
        elif self.output_type == 'classification':
            x = self.cls_fc(x)
            score = x.view(-1, self.num_classes, self.num_stages + 1)
        elif self.output_type == 'binary_classification':
            x = self.bi_cls_fc(x)
            score = x.view(-1, self.num_classes, self.num_stages, 2)
        elif self.output_type == 'coral':
            x = self.coral_fc(x)
            score = self.coral_bias(x)
        elif self.output_type == 'action_classification':
            score = self.ac_cls_fc(x)
        elif self.output_type == 'two_heads':
            prog_inner = self.prog_cls_fc1(x)
            prog = self.prog_cls_fc2(prog_inner)
            action_inner = self.action_cls_fc1(x)
            action = self.action_cls_fc2(action_inner)
            score = {'prog': prog, 'action': action}
        return score

    def tsn(self, x, num_segs):
        # [M, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        return x
