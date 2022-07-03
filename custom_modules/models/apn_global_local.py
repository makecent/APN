import torch

from mmaction.models.builder import LOCALIZERS, build_backbone, build_head
from mmaction.models.localizers import BaseTAGClassifier
from mmaction.core import top_k_accuracy, mean_class_accuracy
from ..apn_utils import binary_accuracy, decode_progression, progression_mae


@LOCALIZERS.register_module()
class APN_GL(BaseTAGClassifier):
    """APN model framework."""

    def __init__(self,
                 global_backbone,
                 local_backbone,
                 cls_head,
                 global_frames=8,
                 local_frames=32):
        super(BaseTAGClassifier, self).__init__()
        self.global_backbone = build_backbone(global_backbone)
        self.local_backbone = build_backbone(local_backbone)
        self.global_frames = global_frames
        self.local_frames = local_frames
        self.cls_head = build_head(cls_head)
        self.init_weights()

    def init_weights(self):
        """Weight initialization for model."""
        self.global_backbone.init_weights()
        self.local_backbone.init_weights()
        self.cls_head.init_weights()

    def _forward(self, imgs):
        # [N, num_frames, C, 1, H, W]
        imgs = imgs.transpose(0, 1).squeeze(dim=3)  # [num_frames, N, C, H, W]
        local_imgs, global_imgs = imgs[:self.local_frames].transpose(0, 1), imgs[self.local_frames:].transpose(0, 1)  # [N, num_frames, C, H, W]
        global_imgs = global_imgs.flatten(end_dim=1)  # [N, num_frames, C, H, W] -> [N*num_frames, C, H, W]
        local_imgs = local_imgs.transpose(1, 2)  # [N, num_frames, C, H, W] -> [N, C, num_frames, H, W]

        local_feat = self.local_backbone(local_imgs).mean(dim=(-1, -2, -3))
        global_feat = self.global_backbone(global_imgs).unflatten(dim=0, sizes=(-1, self.global_frames)).mean(dim=(-1, -2, -4))

        output = self.cls_head(torch.cat([global_feat, local_feat], dim=-1))

        return output

    def forward_train(self, imgs, progression_label=None, class_label=None):
        cls_score, reg_score = self._forward(imgs)
        class_label = class_label.squeeze(-1)
        losses = {'loss_cls': self.cls_head.loss_cls(cls_score, class_label)}

        cls_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                 class_label.detach().cpu().numpy(),
                                 topk=(1,))
        losses[f'cls_acc'] = torch.tensor(cls_acc, device=cls_score.device)
        losses['loss_reg'] = self.cls_head.loss_reg(reg_score, progression_label)

        reg_score = reg_score.sigmoid()
        reg_acc = binary_accuracy(reg_score.detach().cpu().numpy(), progression_label.detach().cpu().numpy())
        reg_mae = progression_mae(reg_score.detach().cpu().numpy(), progression_label.detach().cpu().numpy())
        losses[f'reg_acc'] = torch.tensor(reg_acc, device=reg_score.device)
        losses[f'reg_mae'] = torch.tensor(reg_mae, device=reg_score.device)

        return losses

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and testing."""
        cls_score, reg_score = self._forward(imgs)
        cls_score = cls_score.softmax(-1)
        reg_score = reg_score.sigmoid()
        progression = decode_progression(reg_score)
        return list(zip(cls_score.cpu().numpy(), progression.cpu().numpy()))
