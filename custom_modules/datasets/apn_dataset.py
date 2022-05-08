import copy
import os.path as osp
import numpy as np
from multiprocessing import Pool, cpu_count
# from torch.multiprocessing import Pool, set_start_method
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass
from itertools import repeat
from torch.utils.data import Dataset
from mmcv.utils import print_log
from mmcv import dump
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import Compose
from .apn_utils import apn_detection_on_single_video, uniform_1d_sampling, eval_ap
# from mmaction.core import average_recall_at_avg_proposals, top_k_accuracy, \
#     mean_class_accuracy


@DATASETS.register_module()
class APNDataset(Dataset):
    """APN dataset for action detection.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video, start frame, end frame and
    the label the action, which are split with a comma.
    Example of a annotation file:

    .. code-block:: txt

    video_test_0000324,4470,1476,1605,0
    video_test_0000324,4470,3501,3675,0
    video_test_0000664,2579,39,144,0
    video_test_0000664,2579,669,759,0
    video_test_0000664,2579,1704,1791,0
    video_test_0000714,5383,786,897,0


    Args:
        ann_files (str) or Tuple(str, ...): Path to the annotation files.
        pipeline: A sequence of data transforms.
        data_prefixes (str) or Tuple(str, ...): Path to a directories where video frames are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
                            Default: 'RGB'.
    """

    def __init__(self,
                 ann_files,
                 pipeline,
                 num_stages=100,
                 action_index=None,
                 data_prefixes=None,
                 untrimmed=False,
                 test_sampling=1000,
                 proposal_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 start_index=0,
                 modality='RGB',
                 test_mode=False):
        super().__init__()
        self.ann_files = ann_files if isinstance(ann_files,
                                                 (list, tuple)) else [ann_files]
        data_prefixes = data_prefixes if isinstance(data_prefixes,
                                                    (list, tuple)) else [
            data_prefixes]
        self.data_prefixes = [osp.realpath(root) if osp.isdir(root) else root
                              for root in data_prefixes]
        assert len(self.data_prefixes) == len(self.ann_files)
        self.num_stages = num_stages
        self.action_index = action_index
        self.test_sampling = test_sampling
        self.proposal_mode = proposal_mode

        # test_mode is unused, just for compatibility with mmlab;
        # We use self.untrimmed to clarify if test or train/val.
        # test_mode can only distinguish test/val or train
        self.test_mod = test_mode
        self.untrimmed = untrimmed

        self.pipeline = Compose(pipeline)
        self.filename_tmpl = filename_tmpl
        self.start_index = start_index
        assert modality in ['RGB', 'Flow', 'Video']
        self.modality = modality

        self.gt_infos, self.video_infos = self.load_gt_infos()
        self.frame_infos = []
        for ann_file, data_prefix in zip(self.ann_files, self.data_prefixes):
            self.frame_infos.extend(
                self.load_annotations(ann_file, data_prefix))

    def load_gt_infos(self):
        """Get ground truth bounding boxes information by class level and video level
        GT in class level:
        {class_index1:
                {'video_name1': [[s1, e1], [s2, e2], ...],
                 'video_name2': [[s1, e1], [s2, e2], ...],}
        class_index2:
                ...
        ...
        }
        GT in video level:
        {video_name1:
            {'total_frames': int,
             'gt_bboxes': [[s1, e1, c1], [s2, e2, c2], ...]}
        video_name2:
                ...
        ...
        }
        """
        gt_infos = {}
        video_infos = {}
        for ann_file in self.ann_files:
            with open(ann_file, 'r') as fin:
                for line in fin.readlines():
                    line_split = line.strip().split(',')
                    video_name = str(line_split[0])
                    total_frames = int(line_split[1])
                    start_frame = int(line_split[2])
                    end_frame = int(line_split[3])
                    class_label = int(
                        line_split[4]) if not self.proposal_mode else 0

                    if self.modality != 'Video':
                        video_name = video_name.rsplit('.', 1)[0]
                    gt_infos.setdefault(class_label, {}).setdefault(video_name,
                                                                    []).append(
                        [start_frame, end_frame])

                    video_infos.setdefault(video_name, {}).setdefault(
                        'total_frames', total_frames)
                    video_infos.setdefault(video_name, {}).setdefault(
                        'gt_bboxes', []).append(
                        [start_frame, end_frame, class_label])
                    if isinstance(self.test_sampling, int):
                        video_infos.setdefault(video_name, {}).setdefault(
                            'rescale_rate', total_frames / self.test_sampling)
                    else:
                        assert self.test_sampling == 'full'
                        video_infos.setdefault(video_name, {}).setdefault(
                            'rescale_rate', 1.0)
        return gt_infos, video_infos

    def load_annotations(self, ann_file, data_prefix):
        if not self.untrimmed:
            frame_infos = []
            with open(ann_file, 'r') as fin:
                for line in fin.readlines():
                    line_split = line.strip().split(',')
                    video_name = str(line_split[0])
                    total_frames = int(line_split[1])
                    start_frame = int(line_split[2])
                    end_frame = int(line_split[3])
                    class_label = int(
                        line_split[4]) if not self.proposal_mode else 0

                    if self.action_index and (class_label != self.action_index):
                        continue

                    if self.modality != 'Video':
                        video_name = video_name.rsplit('.', 1)[0]
                    video_name = osp.join(data_prefix, video_name)

                    for frm_idx in range(start_frame, end_frame + 1):
                        frame_info = {'class_label': class_label,
                                      'frame_index': frm_idx}
                        if self.modality == 'Video':
                            frame_info['filename'] = video_name
                        else:
                            frame_info['frame_dir'] = video_name
                            frame_info['total_frames'] = total_frames
                        progression = (frm_idx - start_frame) / (
                                end_frame - start_frame)
                        frame_info['progression_label'] = progression
                        frame_infos.append(frame_info)
        else:
            frame_infos = []
            for video_name, video_info in self.video_infos.items():
                video_name = osp.join(data_prefix, video_name)
                total_frames = video_info['total_frames']
                frame_inds = list(range(self.start_index,
                                        self.start_index + total_frames))
                if self.test_sampling != 'full':
                    frame_inds = uniform_1d_sampling(frame_inds, self.test_sampling)

                for frm_idx in frame_inds:
                    frame_info = {'frame_index': frm_idx}
                    if self.modality == 'Video':
                        frame_info['filename'] = video_name
                    else:
                        frame_info['frame_dir'] = video_name
                        frame_info['total_frames'] = total_frames
                    frame_infos.append(frame_info)
        return frame_infos

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return dump(results, out)

    @staticmethod
    def decode_results(results):
        decoded_results = {}
        try:
            assert len(results[0]) == 2
            decoded_results['argmax_progressions'], decoded_results['exception_progressions'] = zip(*results)
        except:
            decoded_results['progressions'] = results

        return decoded_results

    def evaluate(self,
                 results,
                 metrics='MAE',
                 metric_options=dict(
                     mAP=dict(
                         search=dict(
                             min_e=60,
                             max_s=40,
                             min_L=60,
                             method='mse'),
                         nms=dict(iou_thr=0.4),
                         dump_detections=False,
                         dump_evaluation=False,

                     )),
                 logger=None):
        """Evaluation in rawframe dataset.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
            logger (obj): Training logger. Defaults: None.
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        Returns:
            dict: Evaluation results dict.
        """
        # check
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['MAE', 'mAP', 'AR@AN',
                           'top_k_based_progression', 'framecls_top_k_accuracy',
                           'AR@AN_based_on_apn',
                           'framecls_mean_class_accuracy',
                           'framecls_top_k_accuracy_based_on_apn',
                           'framecls_mean_class_accuracy_based_on_apn']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        results = self.decode_results(results)
        eval_results = {}
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'MAE':
                prog_labels = self.get_progression_labels(denormalized=True)
                if 'progressions' in results:
                    progs = results['progressions']
                    MAE = np.abs(progs - prog_labels).mean()
                    eval_results = self.update_and_print_eval(eval_results, MAE, 'MAE',
                                                              logger)
                    continue
                elif 'exception_progressions' in results:
                    arg_progs, excp_progs = results['argmax_progressions'], \
                                            results['exception_progressions']
                    arg_mae = np.mean(np.abs(arg_progs - prog_labels))
                    excep_mae = np.mean(np.abs(excp_progs - prog_labels))
                    eval_results = self.update_and_print_eval(eval_results, arg_mae,
                                                              'argmax_mae', logger)
                    eval_results = self.update_and_print_eval(eval_results, excep_mae,
                                                              'MAE', logger)
                    continue
                else:
                    raise ValueError(f"No progressions in results")

            if metric == 'mAP':
                assert self.untrimmed is True, "To compute mAP, the dataset must be untrimmed videos"
                employ_cls_score = False  # experimental
                if 'progressions' in results:
                    progs = results['progressions']
                else:
                    progs = results['exception_progressions']
                if employ_cls_score:
                    progs = [[p, c] for p, c in
                             zip(progs, results['classification_scores'])]

                progs = self.split_results_by_video(np.array(progs))
                detections = self.action_detection_on_progs_dict(progs, det_kwargs=metric_options.get('mAP', {}))
                detections = self.format_det_for_func_eval_ap(detections)
                dump_detections = metric_options.get('mAP', {}).get('dump_detections', False)
                if dump_detections:
                    print_log(f"\nwriting detection results to {dump_detections}", logger=logger)
                    dump(detections, dump_detections)
                iou_range = np.arange(0.1, 1.0, .1)
                ap_values, mAP_wo_cls, ap_wi_cls = eval_ap(detections, self.gt_infos, iou_range, return_decomposed=True)
                mAP = ap_values.mean(axis=0)
                mAP_wi_cls = ap_wi_cls.mean(axis=0)
                for iou, mAP_iou in zip(iou_range, mAP):
                    eval_results = self.update_and_print_eval(eval_results, mAP_iou,
                                                              f'mAP@{iou:.01f}', logger)
                for iou, pre_iou in zip(iou_range, mAP_wo_cls):
                    eval_results = self.update_and_print_eval(eval_results, pre_iou,
                                                              f'wo_cls@{iou:.01f}', logger)
                for iou, acc_iou in zip(iou_range, mAP_wi_cls):
                    eval_results = self.update_and_print_eval(eval_results, acc_iou,
                                                              f'wi_cls@{iou:.01f}', logger)
                dump_evaluation = metric_options.get('mAP', {}).get('dump_evaluation', False)
                if dump_evaluation:
                    print_log(f"\nwriting evaluation results to {dump_evaluation}", logger=logger)
                    eval_results_with_options = copy.deepcopy(eval_results)
                    eval_results_with_options.update(metric_options.get('mAP'))
                    dump(eval_results_with_options, dump_evaluation)
                continue

        return eval_results

    def split_results_by_video(self, results):
        """Format the plain_progressions.
        After formatting, the return result is a dict:
        {'video_name1': array with shape (num_classes, L1)
         'video_name2': array with shape (num_classes, L2)
                ...}
        """
        split_results = {}
        cum_frames = 0
        for video_name, data in self.video_infos.items():
            if self.test_sampling == 'full':
                total_frames = data['total_frames']
            else:
                total_frames = self.test_sampling
            this_video_framenum = len(
                range(self.start_index, self.start_index + total_frames))
            p_by_video = results[cum_frames: cum_frames + this_video_framenum]
            split_results[video_name] = p_by_video * (100 / self.num_stages)
            cum_frames += this_video_framenum
        assert cum_frames == len(
            results), "total frames from ds.video_infos and results not equal."
        return split_results

    def format_det_for_func_eval_ap(self, detections):
        """ To use the predefined function 'eval_ap' in 'ssn_utils', we format detections to meet requirements."""
        formated_detections = {}
        for class_ind, data in self.gt_infos.items():
            relative_videos = list(data.keys())
            det_by_c = []
            for video_name in relative_videos:
                det_by_v_by_c = detections[video_name][class_ind]
                if det_by_v_by_c.size == 0:
                    continue
                video_name = np.full((len(det_by_v_by_c), 1), video_name)
                class_id = np.full((len(det_by_v_by_c), 1), class_ind)
                det_by_v_by_c = np.hstack([video_name, class_id,
                                           det_by_v_by_c + self.start_index])
                det_by_c.append(det_by_v_by_c)
            if len(det_by_c) == 0:
                formated_detections[class_ind] = np.empty((1, 5))
            else:
                formated_detections[class_ind] = np.vstack(det_by_c)
        return dict(sorted(formated_detections.items()))

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.frame_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        results = copy.deepcopy(self.frame_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = 'RGB' if self.modality == 'Video' else self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)

    def get_progression_labels(self, denormalized=True):
        progression_labels = np.array(
            [frame_info['progression_label'] for frame_info in
             self.frame_infos])  # [N,] in range (0,1)
        if denormalized:
            progression_labels *= self.num_stages  # [N,] in range (0, num_stages)
        return progression_labels

    def get_MAE_on_untrimmed_results(self, results, return_pv=False):
        """ Experimental function, compute MAE based on predicted progressions of untrimmed video"""
        assert self.untrimmed
        results = np.array(results)
        cum_frames = 0
        pre, gt, pv = [], [], []
        for video_name, video_info in self.video_infos.items():
            num_sampling = video_info['total_frames'] if self.test_sampling == 'full' else self.test_sampling
            sampled_frame = np.linspace(0, video_info['total_frames'] - 1, num_sampling, dtype=int)
            for action_start, action_end, class_label in video_info['gt_bboxes']:
                action_frame = np.arange(action_start, action_end + 1)
                progs = np.linspace(0, 1, len(action_frame))
                sampled_frame, sampled_idx = np.unique(sampled_frame, return_index=True)
                idx1 = sampled_idx[np.where(np.in1d(sampled_frame, action_frame))[0]]
                idx2 = np.where(np.in1d(action_frame, sampled_frame))[0]
                if idx1.size == 0:
                    continue
                pre_all_class = results[idx1 + cum_frames]
                pv.append(np.var(pre_all_class, axis=-1))
                pre.append(pre_all_class[:, class_label])
                gt.append(progs[idx2] * self.num_stages)
            cum_frames += self.test_sampling if self.test_sampling != 'full' else video_info['total_frames']
        assert cum_frames == len(results)
        pre, gt, pv = np.hstack(pre), np.hstack(gt), np.hstack(pv)
        MAE = np.mean(np.abs(gt - pre))
        PV = pv.mean()
        return MAE if not return_pv else (MAE, PV)

    @staticmethod
    def update_and_print_eval(eval_results_holder, result, name, logger=None, decimals=4):
        eval_results_holder[name] = result
        log_msg = f'{name}: \t{result:.{decimals}f}'
        print_log(log_msg, logger=logger)
        return eval_results_holder

    def action_detection_on_progs_dict(self, progressions, det_kwargs, nproc=cpu_count()):
        result_dict = {}
        with Pool(nproc) as pool:
            rescale_rates = [video_info['rescale_rate'] for video_info in self.video_infos.values()]
            video_names, progressions_by_video = zip(*progressions.items())
            all_dets = pool.starmap(
                apn_detection_on_single_video,
                zip(progressions_by_video, rescale_rates, repeat(det_kwargs)))

        for video_name, dets_and_scores in zip(video_names, all_dets):
            result_dict[video_name] = dets_and_scores
        # if dump_detections:
        #     dump(result_dict, dump_detections)
        return result_dict

# Previously used but now deleted codes:

# def top_k_accuracy_metric(cls_scores, gt_labels, metric_options, eval_results,
#                           logger):
#     topk = metric_options.setdefault('top_k_accuracy', {}).setdefault('topk',
#                                                                       (1, 5))
#     if not isinstance(topk, (int, tuple)):
#         raise TypeError('topk must be int or tuple of int, '
#                         f'but got {type(topk)}')
#     if isinstance(topk, int):
#         topk = (topk,)
#
#     top_k_acc = top_k_accuracy(cls_scores, gt_labels, topk)
#
#     log_msg = []
#     for k, acc in zip(topk, top_k_acc):
#         eval_results[f'top{k}_acc'] = acc
#         log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
#     log_msg = ''.join(log_msg)
#     print_log(log_msg, logger=logger)
#
#     return eval_results

# def get_correlated_results(self, results):
#     """ Keep the predicted progressions of correlated action class, removing the uncorrelated ones"""
#     class_labels = [frame_info['class_label'] for frame_info in self.frame_infos]
#     gathered_results = np.take_along_axis(np.array(results), np.expand_dims(
#         np.array(class_labels), axis=-1), axis=-1)
#     return np.squeeze(gathered_results).tolist()

#
# def _import_prop_ground_truth(self):
#     """Refer to function with same name in activitynet_dataset.py"""
#     ground_truth = {}
#     for video_id, video_info in self.video_infos.items():
#         this_video_ground_truths = video_info['gt_bboxes']
#         ground_truth[video_id] = np.array(this_video_ground_truths)
#     return ground_truth
#
# def _import_proposals(self, progressions):
#     """Refer to function with same name in activitynet_dataset.py"""
#     progressions = self.split_results_by_video(progressions)
#     detections = {}
#     num_detections = 0
#     rescale_len = 1000
#     for video_name, p_by_v in progressions.items():
#         original_len = p_by_v.shape[1]
#         scale_ratio = original_len / rescale_len
#         p = imresize(p_by_v.squeeze()[np.newaxis, ...].astype(np.float),
#                      (rescale_len, 1)).squeeze()
#         det_by_v_by_c = apn_detection_on_vector(p, min_e=60, max_s=40,
#                                                 min_L=60 / scale_ratio)
#         det_by_v_by_c = nms1d(det_by_v_by_c)
#         det_by_v_by_c[:, :2] *= scale_ratio
#         detections[video_name] = det_by_v_by_c
#         num_detections += det_by_v_by_c.shape[0]
#     return detections, num_detections

# if metric == 'videocls_top_k_based_on_apn':
# sampling_rate = 50
# video_labels = [video_info['gt_bboxes'][0][2] for video_info in
#                 self.video_infos.values()]
# if 'predictions' in results:
#     progs = results['predictions']
# else:
#     progs = results['exception_progressions']
# if 'classification_scores' in results:
#     progs = [[p, c] for p, c in
#              zip(progs, results['classification_scores'])]
# progs = self.split_results_by_video(np.array(progs))
# scores = []
# for video_name, progs_by_v in progs.items():
#     if progs_by_v.ndim == 3:
#         p_by_v = progs_by_v[:, 0, :]
#         cs_by_v = progs_by_v[:, 1, :]
#         cs_by_v = sampling(cs_by_v, sampling_rate)
#         p_by_v = sampling(p_by_v, sampling_rate)
#         cs_by_v = cs_by_v.mean(axis=0)
#
#     else:
#         p_by_v = progs_by_v
#         p_by_v = sampling(p_by_v, sampling_rate)
#         cs_by_v = None
#
#     this_video_length = p_by_v.shape[0]
#     action_template = np.linspace(0, 100, this_video_length)
#     mse = ((action_template - p_by_v.T) ** 2).mean(axis=-1)
#     score = -mse / 1666.66 + 1
#     score = score * cs_by_v
#     scores.append(score)
#     # scores.append(cs_by_v)
# eval_results = top_k_accuracy_metric(scores, video_labels,
#                                      metric_options,
#                                      eval_results, logger)
# continue

# if metric == 'framecls_top_k_accuracy':
#     topk = metric_options.setdefault('top_k_accuracy',
#                                      {}).setdefault('topk', (1, 5))
#     if not isinstance(topk, (int, tuple)):
#         raise TypeError('topk must be int or tuple of int, '
#                         f'but got {type(topk)}')
#     if isinstance(topk, int):
#         topk = (topk,)
#
#     gt_labels = [frame_info['class_label'] for frame_info in
#                  self.frame_infos]
#     top_k_acc = top_k_accuracy(results['classification_scores'],
#                                gt_labels, topk)
#     for k, acc in zip(topk, top_k_acc):
#         eval_results = self.print_result(eval_results, acc,
#                                          f'top{k}_acc', logger)
#     continue

# if metric == 'framecls_mean_class_accuracy':
# gt_labels = [frame_info['class_label'] for frame_info in self.frame_infos]
# mean_acc = mean_class_accuracy(results['classification_scores'],
#                                gt_labels)
# eval_results = self.print_result(eval_results, mean_acc,
#                                  'mean_acc', logger)
# continue

# Refer to 'activitynet_dataset.evaluate(metric='AR@AN')'
# if metric == 'AR@AN':
# temporal_iou_thresholds = metric_options.setdefault(
#     'AR@AN', {}).setdefault('temporal_iou_thresholds',
#                             np.linspace(0.5, 0.95, 10))
# max_avg_proposals = metric_options.setdefault(
#     'AR@AN', {}).setdefault('max_avg_proposals', 100)
# if isinstance(temporal_iou_thresholds, list):
#     temporal_iou_thresholds = np.array(temporal_iou_thresholds)
#
# ground_truth = self._import_prop_ground_truth()
# proposal, num_proposals = self._import_proposals(
#     results['predictions'])
# recall, _, _, auc = (
#     average_recall_at_avg_proposals(
#         ground_truth,
#         proposal,
#         num_proposals,
#         max_avg_proposals=max_avg_proposals,
#         temporal_iou_thresholds=temporal_iou_thresholds))
# eval_results['auc'] = auc
# eval_results['AR@1'] = np.mean(recall[:, 0])
# eval_results['AR@5'] = np.mean(recall[:, 4])
# eval_results['AR@10'] = np.mean(recall[:, 9])
# eval_results['AR@50'] = np.mean(recall[:, 49])
# eval_results['AR@100'] = np.mean(recall[:, 99])
# continue

# if metric == 'AR@AN_based_on_apn':
# progs = results['progressions']
# progs = self.split_results_by_video(np.array(progs))
# proposals = {}
# raw_proposals = {}
# rescale_len = 1000
# for video_name, p_by_v in progs.items():
#     original_len = p_by_v.shape[0]
#     scale_ratio = original_len / rescale_len
#
#     # proposals[video_name] = []
#     # for class_ind, p_by_v_by_c in enumerate(p_by_v.T):
#     # p = imresize(p_by_v_by_c[np.newaxis, ...].astype(np.float), (rescale_len, 1)).squeeze()
#     p = sampling(p_by_v, rescale_len)
#     # p = sampling(p_by_v_by_c, rescale_len)
#     det_by_v_by_c = apn_detection_on_vector(p, min_e=60,
#                                             max_s=40,
#                                             min_L=60 / scale_ratio,
#                                             score_threshold=0,
#                                             method='mse')
#     det_by_v_by_c[:, :2] *= (scale_ratio * self.stride)
#     det_by_v_by_c[:, :2] += float(self.start_index)
#     # proposals[video_name].append(det_by_v_by_c)
#     raw_proposals[video_name] = det_by_v_by_c
#     # raw_proposals[video_name] = np.vstack(proposals[video_name])
#     proposals[video_name] = nms(det_by_v_by_c, 0.8)
#
# if dump_detections:
#     dump(raw_proposals, dump_detections)
# temporal_iou_thresholds = metric_options.setdefault(
#     'AR@AN', {}).setdefault('temporal_iou_thresholds',
#                             np.linspace(0.5, 0.95, 10))
# max_avg_proposals = metric_options.setdefault(
#     'AR@AN', {}).setdefault('max_avg_proposals', 100)
# if isinstance(temporal_iou_thresholds, list):
#     temporal_iou_thresholds = np.array(temporal_iou_thresholds)
#
# ground_truth = self._import_prop_ground_truth()
# recall, _, _, auc = (
#     average_recall_at_avg_proposals(
#         ground_truth,
#         proposals,
#         100,
#         max_avg_proposals=max_avg_proposals,
#         temporal_iou_thresholds=temporal_iou_thresholds))
# eval_results['auc'] = auc
# eval_results['AR@1'] = np.mean(recall[:, 0])
# eval_results['AR@5'] = np.mean(recall[:, 4])
# eval_results['AR@10'] = np.mean(recall[:, 9])
# eval_results['AR@50'] = np.mean(recall[:, 49])
# eval_results['AR@100'] = np.mean(recall[:, 99])
# continue
# if metric == 'loss':
#     loss = sum(results['losses']) / len(results['losses'])
#     eval_results = self.print_result(eval_results, loss, 'loss',
#                                      logger)
#     continue
