import torch
from mmaction.models.builder import LOSSES
from mmaction.models.losses import BaseWeightedLoss
from torch.nn import functional as F


@LOSSES.register_module(force=True)
class L1LossWithLogits(BaseWeightedLoss):

    def _forward(self, cls_score, label, **kwargs):
        loss_cls = F.l1_loss(cls_score.sigmoid(), label.reshape_as(cls_score), **kwargs)

        return loss_cls


@LOSSES.register_module(force=True)
class L2LossWithLogits(BaseWeightedLoss):

    def _forward(self, cls_score, label, **kwargs):
        loss_cls = F.mse_loss(cls_score.sigmoid(), label.reshape_as(cls_score), **kwargs)

        return loss_cls


@LOSSES.register_module(force=True)
class CrossEntropyLossV2(BaseWeightedLoss):
    """Cross Entropy Loss.
    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.
    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None, label_smoothing=0.0):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)
        self.label_smoothing = label_smoothing

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.
        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if self.class_weight is not None:
            assert 'weight' not in kwargs, \
                "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        if self.label_smoothing > 0:
            assert 'label_smoothing' not in kwargs, \
                "The 'label_smoothing' is already defined in the loss as a class attribute"
        loss_cls = F.cross_entropy(cls_score, label, label_smoothing=self.label_smoothing, **kwargs)

        return loss_cls


@LOSSES.register_module()
class BCELossWithLogitsV2(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_weight=1.0, class_weight=None, label_smoothing=0.0):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)
        self.label_smoothing = label_smoothing

    @staticmethod
    def _smooth(label, smoothing=0.1):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            label = label * (1.0 - smoothing) + (label * smoothing).mean(dim=-1, keepdim=True)
        return label

    def _forward(self, cls_score, label, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        if self.class_weight is not None:
            assert 'weight' not in kwargs, "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        if self.label_smoothing > 0:
            assert 'label_smoothing' not in kwargs, \
                "The 'label_smoothing' is already defined in the loss as a class attribute"
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, self._smooth(label, self.label_smoothing),
                                                      **kwargs)
        return loss_cls


@LOSSES.register_module()
class FocalLoss(BaseWeightedLoss):
    def __init__(self, gamma=2.0, alpha=0.25, do_onehot=True, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.do_onehot = do_onehot
        self.label_smoothing = label_smoothing

    @staticmethod
    def _smooth(label, smoothing=0.1):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            label = label * (1.0 - smoothing) + (label * smoothing).mean(dim=-1)
        return label

    def _forward(self, cls_score, label, **kwargs):
        if self.do_onehot:
            label = F.one_hot(label, num_classes=cls_score.size(-1))
        loss = F.binary_cross_entropy_with_logits(cls_score, self._smooth(label, self.label_smoothing), **kwargs)
        cls_score = cls_score.sigmoid()
        pt = (1 - cls_score) * label + cls_score * (1 - label)
        focal_weight = (self.alpha * label + (1 - self.alpha) * (1 - label)) * pt.pow(self.gamma)
        return focal_weight * loss
