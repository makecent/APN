import numpy as np
import torch
from mmaction.core.evaluation.accuracy import pairwise_temporal_iou, interpolated_precision_recall


def uniform_1d_sampling(vector, num_sampling, return_idx=False):
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


def apn_detection_on_single_video(progressions, rescale_rate=1.0, det_kwargs={}):
    sampling = det_kwargs.get('sampling', None)  # experimental argument
    search_kwargs = det_kwargs.get('search', {})
    nms_kwargs = det_kwargs.get('nms', {})

    if sampling:  # experimental argument
        assert isinstance(sampling, (int, float))
        assert rescale_rate == 1.0, "'sampling' is just an experimental argument used to down-sampling 'full' results " \
                                    "to investigate the impact of sampling with less time" \
                                    "If you didn't 'test_sampling' as 'full', don't set this 'sampling' argument"
        rescale_rate = len(progressions) / sampling
    search_kwargs['min_L'] /= rescale_rate

    dets_and_scores = []
    for class_ind, progs_by_class in enumerate(progressions.T):
        if sampling:
            progs_by_class = uniform_1d_sampling(progs_by_class, sampling)
        dets_by_class, scores_by_class = apn_detection_on_vector(progs_by_class, **search_kwargs)
        dets_by_class = dets_by_class * rescale_rate
        dets_by_class, scores_by_class = nms1d(dets_by_class, scores_by_class, **nms_kwargs)
        dets_and_scores.append(np.hstack([dets_by_class, scores_by_class[:, None]]))
    # set 'min_L' back for the following detection because dictionary objects are mutable.
    search_kwargs['min_L'] *= rescale_rate

    return dets_and_scores


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


def eval_ap(detections, gt_by_cls, iou_range, return_decomposed=False):
    """ **** This function is revised from the one with the same name in mmaction2.localization.ssn_utils ****
    Evaluate mean average precisions (mAP) and classification accuracy (Cls. acc.) of the located samples.
    Args:
        detections (dict): Results of detections.
        gt_by_cls (dict): Information of groudtruth.
        iou_range (list or numpy.array): Ranges of iou.
        return_decomposed (boolean): Return loc_precision and cls_accuracy if true, which two decompose the mAP.

    Returns:
        list: Average precision values of classes at ious.
    """
    ap_values = np.zeros((len(detections), len(iou_range)))
    ap_wi_cls = np.ones((len(detections), len(iou_range)))
    mAP_wo_cls = np.ones(len(iou_range))

    for class_idx in detections.keys():
        tps = []
        for class_idx_x in detections.keys():
            # compute AP of proposals with considering classification, i.e. the normal AP.
            ap, tp = average_precision_at_temporal_iou(gt_by_cls[class_idx_x],
                                                       detections[class_idx],
                                                       iou_range,
                                                       return_tp=True)
            if class_idx_x == class_idx:
                ap_values[class_idx, :] = ap
                if not return_decomposed:
                    break
            tps.append(tp)
        tps = np.stack(tps)
        if return_decomposed:
            # compute AP of proposals (that have been detected w/o classification) with considering classification.
            for iou_idx, min_overlap in enumerate(iou_range):
                tps_of_iou = tps[:, iou_idx, :]
                detection_of_cls = detections[class_idx]
                detection_of_cls_sorted = detection_of_cls[detection_of_cls[:, 4].astype(float).argsort()[::-1]]
                detections_located = detection_of_cls_sorted[np.where(tps_of_iou.any(axis=0))[0]]
                ap_wi_cls[class_idx, iou_idx] = average_precision_at_temporal_iou(gt_by_cls[class_idx],
                                                                                 detections_located,
                                                                                 [min_overlap])
            # for iou_idx, min_overlap in enumerate(iou_range):
            # tps_of_iou = tps[:, iou_idx, :]
            # tps_of_located = tps_of_iou[:, np.where(tps_of_iou.any(axis=0))[0]]
            # num_located = tps_of_located.shape[-1]
            # if tps_of_located.shape[-1] > 0:
            #     num_cls = np.count_nonzero(tps_of_located[class_idx])
            #     cls_accuracy[class_idx, iou_idx] = num_cls / num_located
    if return_decomposed:
        # compute mAP of proposals without considering classification
        detections_ignore_cls = list(detections.values())
        detections_ignore_cls = np.vstack(detections_ignore_cls)
        gt_ignore_cls = {}
        for gt_of_cls in gt_by_cls.values():
            for video, gt in gt_of_cls.items():
                gt_ignore_cls.setdefault(video, []).extend(gt)
        mAP_wo_cls = average_precision_at_temporal_iou(gt_ignore_cls,
                                                      detections_ignore_cls,
                                                      iou_range)
    return (ap_values, mAP_wo_cls, ap_wi_cls) if return_decomposed else ap_values


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