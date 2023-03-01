import numpy as np
import torch
from mmaction.core.evaluation.accuracy import pairwise_temporal_iou, interpolated_precision_recall

from custom_modules.mmdet_utils import multiclass_nms
from matplotlib import pyplot as plt

def decode_progression(reg_score):
    batch_size, num_stage = reg_score.shape
    if isinstance(reg_score, torch.Tensor):
        progression = torch.count_nonzero(reg_score > 0.5, dim=-1)
        # x1 = torch.cat([torch.ones((batch_size, 1), device=reg_score.device), reg_score], dim=-1)
        # x2 = torch.cat([reg_score, torch.zeros((batch_size, 1), device=reg_score.device)], dim=-1)
        # p = (x1 - x2).clamp(0)
        # v = torch.arange(num_stage+1, device=reg_score.device).repeat((batch_size, 1))
        # progression = (p * v).sum(dim=-1)
    elif isinstance(reg_score, np.ndarray):
        progression = np.count_nonzero(reg_score > 0.5, axis=-1)
    else:
        raise TypeError(f"unsupported reg_score type: {type(reg_score)}")
    progression = progression * 100 / num_stage
    return progression


def progression_mae(reg_score, progression_label):
    progression = decode_progression(reg_score)
    progression_label = decode_progression(progression_label)
    if isinstance(reg_score, torch.Tensor):
        mae = torch.abs(progression - progression_label)
    elif isinstance(reg_score, np.ndarray):
        mae = np.abs(progression - progression_label)
    else:
        raise TypeError(f"unsupported reg_score type: {type(reg_score)}")
    return mae.mean()


def binary_accuracy(pred, label):
    acc = np.count_nonzero((pred > 0.5) == label) / label.size
    return acc


def uniform_sampling_1d(vector, num_sampling, return_idx=False):
    input_type = type(vector)
    if isinstance(vector, (list, tuple, range)):
        vector = np.array(vector)
    idx = np.linspace(0, len(vector) - 1, num_sampling, dtype=int)
    sampled = vector[idx].tolist() if input_type == list else vector[idx]
    return (sampled, idx) if return_idx else sampled


def compute_iou(a, b):
    ov = 0
    union = max(a[1], b[1]) - min(a[0], b[0])
    intersection = min(a[1], b[1]) - max(a[0], b[0])
    if intersection > 0:
        ov = intersection / union
    return ov


def matrix_iou(gt, ads):
    ov_m = np.zeros([gt.shape[0], ads.shape[0]])
    for i in range(gt.shape[0]):
        for j in range(ads.shape[0]):
            ov_m[i, j] = compute_iou(gt[i, :], ads[j, :])
    return ov_m


def cluster_and_flatten(arr, gap):
    # cluster arr wit gap smaller than a value
    m = [[arr[0]]]
    for x in arr[1:]:
        if x - m[-1][0] < gap:
            m[-1].append(x)
        else:
            m.append([x])
    # fetch element from each cluster, have many choices while here we only keep first ele in each cluster.
    r = [c[0] for c in m]
    # r = []
    # for c in m:
    #     r.extend([c[0], c[len(c)//2], c[-1]])
    return r


def score_progression_proposal(proposal, method='mse', backend='numpy'):
    # 1666.66/33.33 is the minimum MSE/MAE error of 100 ranks.
    pytorch = True if backend == 'torch' else False
    template = torch.linspace(0, 100, len(proposal), device=proposal.device) if pytorch else np.linspace(0, 100,
                                                                                                         len(proposal))
    if method == 'mae':
        mae = torch.abs(proposal - template).mean() if pytorch else np.abs(proposal - template).mean()
        score = -mae / 33.33 + 1
    elif method == 'mse':
        mse = ((template - proposal) ** 2).mean()
        score = -mse / 1666.66 + 1
    else:
        raise ValueError(f'cannot understand the scoring method {method}')
    return score


def apn_detection_on_single_video(results):
    video_name, results, rescale_rate, kwargs = results
    search_kwargs = kwargs.get('search', {}).copy()
    search_kwargs['min_L'] /= rescale_rate

    cls_score_all, progression = map(np.array, zip(*results))
    det_bbox, loc_score = apn_detection_on_vector(progression, **search_kwargs)

    if len(det_bbox) == 0:
        return torch.empty([0, 3]), torch.empty([0])

    cls_score = np.array([(cls_score_all[bbox[0]: bbox[1] + 1]).mean(axis=0) for bbox in det_bbox])
    # det_bbox = det_bbox * rescale_rate

    nms_kwargs = kwargs.get('nms', {})
    det_bbox, cls_score, loc_score = map(torch.from_numpy, (det_bbox, cls_score, loc_score))
    score = cls_score * loc_score[:, None]
    det_bbox, det_label = multiclass_nms(
        det_bbox,
        score,
        nms_kwargs.get('score_thr', 0.),
        nms_kwargs.get('nms', dict(iou_thr=0.4)),
        nms_kwargs.get('max_per_video', -1))

    # top5 = np.argpartition(cls_score_all.mean(axis=0), -5)[-5:]
    # remain = np.in1d(det_label, top5)
    # det_bbox = det_bbox[remain]
    # det_label = det_label[remain]
    return det_bbox, det_label


def apn_detection_on_vector(progression_vector, min_e=60, max_s=40, min_L=60, score_threshold=0, method='mse',
                            backbend='numpy'):
    """
    :param progression_vector:
    :param min_e:
    :param max_s:
    :param min_L:
    :return:
    """
    pytorch = True if backbend == 'torch' else False
    if pytorch:
        progression_vector = torch.from_numpy(progression_vector).cuda()
    progs = progression_vector.squeeze()
    start_candidates = torch.where(progs < max_s)[0] if pytorch else np.where(progs < max_s)[0]
    end_candidates = torch.where(progs > min_e)[0] if pytorch else np.where(progs > min_e)[0]
    dets = []
    scores = []

    for start in start_candidates:
        for end in end_candidates:
            this_action_length = end - start + 1
            if this_action_length > min_L:
                score = score_progression_proposal(progs[start:end + 1], method)
                if score > score_threshold:
                    dets.append([start, end])
                    scores.append(score)

    dets = torch.tensor(dets) if pytorch else np.array(dets)
    scores = torch.tensor(scores) if pytorch else np.array(scores)
    descending = -scores.argsort()
    dets = dets[descending].reshape(-1, 2)
    scores = scores[descending]
    return (dets.cpu().numpy(), scores.cpu().numpy()) if pytorch else (dets, scores)


def nms1d(dets, scores, iou_thr=0.0, top_k=np.inf, return_idx=False, backend='numpy'):
    """NMS for temporal proposals.

    Args:
        dets (Tensor): Proposals generated by network.
        iou_thr (Tensor): High threshold for soft nms.
        top_k (Tensor): Top k values to be considered.

    Returns:
        Tensor: The updated proposals.
    """
    on_tensor = True if backend == 'torch' else False
    if dets.size == 0:
        return (dets, scores) if not return_idx else (dets, scores, np.empty([]))
    scores_ = scores.clone() if on_tensor else scores.copy()
    keep = []
    while scores_.max() > 0 and len(keep) < top_k:
        max_index = scores_.argmax(dim=0) if on_tensor else scores_.argmax(axis=0)

        keep.append(max_index)
        scores_[max_index] = 0

        remain = scores_.nonzero(as_tuple=True) if on_tensor else scores_.nonzero()

        ious = one_vs_n_iou(dets[max_index], dets[remain], backend=backend)
        scores_[remain[0][ious > iou_thr]] = 0

    keep = torch.stack(keep) if on_tensor else np.stack(keep)
    dets = dets[keep, :]
    scores = scores[keep]
    if not return_idx:
        return dets, scores
    return dets, scores, keep


def one_vs_n_iou(one_box, n_boxes, backend='numpy'):
    """Compute IoU score between a box and other n boxes.

    (1D)
    """
    one_len = one_box[1] - one_box[0]
    n_len = n_boxes[:, 1] - n_boxes[:, 0]
    if backend == 'torch':
        inter_left = torch.maximum(one_box[0], n_boxes[:, 0])
        inter_right = torch.minimum(one_box[1], n_boxes[:, 1])
        inter_len = torch.clamp(inter_right - inter_left, min=0)
        union_len = one_len + n_len - inter_len
        jaccard = torch.divide(inter_len, union_len)
    else:
        inter_left = np.maximum(one_box[0], n_boxes[:, 0])
        inter_right = np.minimum(one_box[1], n_boxes[:, 1])
        inter_len = (inter_right - inter_left).clip(min=0)
        union_len = one_len + n_len - inter_len
        jaccard = np.divide(inter_len, union_len)
    return jaccard


def plot_detection(detection, video_name, fig=None):
    from matplotlib import pyplot as plt
    import numpy as np
    video_detection = detection[video_name]
    video_detection = np.vstack(video_detection)
    if fig is not None:
        fig.bar(video_detection[:, :2].mean(-1), video_detection[:, -1]*100, width=video_detection[:, 1] - video_detection[:, 0], align='center', alpha=0.5)
    else:
        plt.bar(video_detection[:, :2].mean(-1), video_detection[:, -1]*100, width=video_detection[:, 1] - video_detection[:, 0], align='center', alpha=0.5)



def plot_prediction(prediction, video_name, fig=None):
    from matplotlib import pyplot as plt
    video_prediction = prediction[video_name]
    if fig is not None:
        fig.plot(video_prediction, '-')
    else:
        plt.plot(video_prediction, '-')



def average_precision_at_temporal_iou(ground_truth,
                                      prediction,
                                      temporal_iou_thresholds=(np.linspace(
                                          0.5, 0.95, 10)),
                                      return_tp=False):
    """**** This function is revised from the one with the same name in mmaction2.core.evaluation.accuracy ****
    Compute average precision (in detection task) between ground truth and
    predicted data frames. If multiple predictions match the same predicted
    segment, only the one with highest score is matched as true positive. This
    code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (dict): Dict containing the ground truth instances.
            Key: 'video_id'
            Value (np.ndarray): 1D array of 't-start' and 't-end'.
        prediction (np.ndarray): 2D array containing the information of
            proposal instances, including 'video_id', 'class_id', 't-start',
            't-end' and 'score'.
        temporal_iou_thresholds (np.ndarray): 1D array with temporal_iou
            thresholds. Default: ``np.linspace(0.5, 0.95, 10)``.
        return_tp (Boolean): Return tp if true

    Returns:
        np.ndarray: 1D array of average precision score.
    """
    ap = np.zeros(len(temporal_iou_thresholds), dtype=np.float32)
    if len(prediction) < 1:
        return ap

    num_gts = 0.
    lock_gt = dict()
    for key in ground_truth:
        lock_gt[key] = np.ones(
            (len(temporal_iou_thresholds), len(ground_truth[key]))) * -1
        num_gts += len(ground_truth[key])

    # Sort predictions by decreasing score order.
    prediction = np.array(prediction)
    scores = prediction[:, 4].astype(float)
    sort_idx = np.argsort(scores)[::-1]
    prediction = prediction[sort_idx]

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(temporal_iou_thresholds), len(prediction)),
                  dtype=np.int32)
    fp = np.zeros((len(temporal_iou_thresholds), len(prediction)),
                  dtype=np.int32)

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in enumerate(prediction):

        # Check if there is at least one ground truth in the video.
        if this_pred[0] in ground_truth:
            this_gt = np.array(ground_truth[this_pred[0]], dtype=float)
        else:
            fp[:, idx] = 1
            continue

        t_iou = pairwise_temporal_iou(this_pred[2:4].astype(float), this_gt)
        # We would like to retrieve the predictions with highest t_iou score.
        t_iou_sorted_idx = t_iou.argsort()[::-1]
        for t_idx, t_iou_threshold in enumerate(temporal_iou_thresholds):
            for jdx in t_iou_sorted_idx:
                if t_iou[jdx] < t_iou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[this_pred[0]][t_idx, jdx] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[this_pred[0]][t_idx, jdx] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float32)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float32)
    recall_cumsum = tp_cumsum / num_gts

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(temporal_iou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                  recall_cumsum[t_idx, :])

    return (ap, tp) if return_tp else ap


def plot_gt(video_name, height=100, test_sampling=1000, fig=None):
    from matplotlib import pyplot as plt
    from itertools import repeat
    import pandas as pd
    import numpy as np
    # plot_gt(video_name.rsplit('/', 1)[-1] + '.mp4', 2)
    # plt.plot(cls_score.sum(-1))
    # plt.title(video_name.rsplit('/', 1)[-1])
    # plt.show()
    # plot_gt(video_name.rsplit('/', 1)[-1] + '.mp4', 100)
    # plt.title(video_name.rsplit('/', 1)[-1])
    # plt.plot(progression)
    # plt.show()
    if '/' in video_name:
        video_name = video_name.rsplit('/', 1)[-1]
    if '.' not in video_name:
        video_name = video_name + '.mp4'
    gts = pd.read_csv('/home/louis/PycharmProjects/APN/my_data/thumos14/annotations/apn/apn_test.csv', header=None)
    this_gt = gts[gts[0] == video_name]
    endpoints = this_gt.iloc[:, 2:4].values
    video_length = this_gt[1].iloc[0]
    normalized_endpoints = endpoints/video_length * test_sampling
    normalized_endpoints = np.rint(normalized_endpoints).astype(int)
    start = normalized_endpoints[:, 0]
    end = normalized_endpoints[:, 1]
    if fig is not None:
        fig.bar(normalized_endpoints.mean(-1), height, width=end - start, align='center', alpha=0.5)
    else:
        plt.bar(normalized_endpoints.mean(-1), height, width=end - start, align='center', alpha=0.5)


from mmaction.models.builder import LOSSES
from mmaction.models.losses import BaseWeightedLoss
from torch.nn import functional as F


@LOSSES.register_module(force=True)
class L1LossWithLogits(BaseWeightedLoss):

    def _forward(self, cls_score, label, **kwargs):
        loss_cls = F.l1_loss(cls_score.sigmoid(), label, **kwargs)

        return loss_cls


@LOSSES.register_module(force=True)
class L2LossWithLogits(BaseWeightedLoss):

    def _forward(self, cls_score, label, **kwargs):
        loss_cls = F.mse_loss(cls_score.sigmoid(), label, **kwargs)

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