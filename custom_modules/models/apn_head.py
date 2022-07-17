from abc import ABCMeta

import torch
import torch.nn as nn
from torch.nn import functional as F

from mmcv.cnn import kaiming_init, normal_init, constant_init
from mmaction.models.builder import HEADS, build_loss

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
                 hid_channels=256,
                 loss_cls=dict(type='CrossEntropyLossV2', label_smoothing=0.1),
                 loss_reg=dict(type='BCELossWithLogitsV2', label_smoothing=0.1),
                 dropout_ratio=0.5,
                 avg3d=True):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.num_stages = num_stages
        self.dropout_ratio = dropout_ratio

        self.reg_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.cls_attention1 = nn.MultiheadAttention(in_channels, num_heads=8, batch_first=True)
        self.cls_attention2 = nn.MultiheadAttention(in_channels, num_heads=8, batch_first=True)
        self.reg_attention1 = nn.MultiheadAttention(in_channels, num_heads=8, batch_first=True)
        self.reg_attention2 = nn.MultiheadAttention(in_channels, num_heads=8, batch_first=True)

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) if avg3d else nn.Identity()

        self.cls_fc = nn.Linear(self.in_channels, self.num_classes)

        self.coral_fc = nn.Linear(self.in_channels, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(1, self.num_stages), requires_grad=True)

    def init_weights(self):
        kaiming_init(self.cls_fc, a=0, nonlinearity='relu', distribution='uniform')
        kaiming_init(self.coral_fc, a=0, nonlinearity='relu', distribution='uniform')
        constant_init(self.coral_bias, 0)

    def forward(self, x):
        x = self.avg_pool(x)
        # x = x.view(x.shape[0], -1)

        print(x.shape, x[:, 0, :].shape)
        cls_x = self.cls_attention1(query=x[:, 0, :], key=x, value=x)
        cls_x = self.cls_attention2(query=cls_x[:, 0, :], key=cls_x, value=cls_x)
        cls_token = cls_x[:, 0, :].squeeze(dim=1)

        reg_x = torch.cat((self.reg_token, x[:, 1:, :]))
        reg_x = self.reg_attention1(query=reg_x[:, 0, :], key=reg_x, value=reg_x)
        reg_x = self.reg_attention2(query=reg_x[:, 0, :], key=reg_x, value=reg_x)
        reg_token = reg_x[:, 0, :].squeeze(dim=1)

        cls_token = F.dropout(cls_token, p=self.dropout_ratio)
        reg_token = F.dropout(reg_token, p=self.dropout_ratio)

        cls_score = self.cls_fc(cls_token)
        reg_score = self.coral_fc(reg_token) + self.coral_bias

        return cls_score, reg_score

# class TokenAttentionLayer(nn.Module, metaclass=ABCMeta):
#     def __init__(self, embed_dim):
#         super(TokenAttentionLayer, self).__init__()
#         self.embed_dim = embed_dim
#         self.q_fc = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, cls_token, x):
#         q = self.q_fc(cls_token)
#         k =
