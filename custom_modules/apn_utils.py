import numpy as np
import torch
from mmaction.core.evaluation.accuracy import pairwise_temporal_iou, interpolated_precision_recall

from custom_modules.mmdet_utils import multiclass_nms
from matplotlib import pyplot as plt

def decode_progression(reg_score):
    num_stage = reg_score.shape[-1]
    if isinstance(reg_score, torch.Tensor):
        progression = torch.count_nonzero(reg_score > 0.5, dim=-1)
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

    cls_score, progression = map(np.array, zip(*results))
    det_bbox, loc_score = apn_detection_on_vector(progression, **search_kwargs)

    if len(det_bbox) == 0:
        return torch.empty([0, 3]), torch.empty([0])

    cls_score = np.array([(cls_score[bbox[0]: bbox[1] + 1]).mean(axis=0) for bbox in det_bbox])
    det_bbox = det_bbox * rescale_rate

    nms_kwargs = kwargs.get('nms', {})
    det_bbox, cls_score, loc_score = map(torch.from_numpy, (det_bbox, cls_score, loc_score))
    det_bbox, det_label = multiclass_nms(
        det_bbox,
        cls_score * loc_score[:, None],
        nms_kwargs.get('score_thr', 0.00),
        nms_kwargs.get('nms', dict(iou_thr=0.4)),
        nms_kwargs.get('max_per_video', -1))
        # score_factors=loc_score)
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
    # start_candidates = cluster_and_flatten(start_candidates, 2)
    end_candidates = torch.where(progs > min_e)[0] if pytorch else np.where(progs > min_e)[0]
    # end_candidates = cluster_and_flatten(end_candidates, 2)
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


def plot_detection(video_prediction, gt, ads):
    from matplotlib import pyplot as plt
    import numpy as np
    plt.figure(figsize=(15, 5))
    plt.plot(video_prediction, '-')
    plt.vlines(gt[:, 0], 0, 100, colors='r', linestyles='solid', label='ground truth')
    plt.vlines(gt[:, 1], 0, 100, colors='r', linestyles='solid', label='ground truth')
    plt.vlines(ads[:, 0], 0, 100, colors='k', linestyles='dashed', label='ground truth')
    plt.vlines(ads[:, 1], 0, 100, colors='k', linestyles='dashed', label='ground truth')
    plt.yticks(np.arange(0, 100, 20.0))
    plt.xlabel('Frame Index')
    plt.ylabel('Completeness')
    plt.grid()
    plt.show()


def plot_prediction(video_prediction):
    from matplotlib import pyplot as plt
    import numpy as np
    plt.figure(figsize=(15, 5))
    plt.plot(video_prediction, '-')
    plt.yticks(np.arange(0, 100, 20.0))
    plt.xlabel('Frame Index')
    plt.ylabel('Completeness')
    plt.grid()
    plt.show()


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


def plot_gt(video_name, height):
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
    gts = pd.read_csv('/home/louis/PycharmProjects/APN/my_data/thumos14/annotations/apn/apn_test.csv', header=None)
    this_gt = gts[gts[0] == video_name]
    endpoints = this_gt.iloc[:, 2:4].values
    video_length = this_gt[1].iloc[0]
    normalized_endpoints = endpoints/video_length * 1000
    normalized_endpoints = np.rint(normalized_endpoints).astype(int)
    start = normalized_endpoints[:, 0]
    end = normalized_endpoints[:, 1]
    plt.bar(normalized_endpoints.mean(-1), height, width=end - start, align='center', alpha=0.5)