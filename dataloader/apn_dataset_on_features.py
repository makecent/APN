import copy
import os.path as osp
import numpy as np
import h5py

from torch.utils.data import Dataset
from mmcv.utils import print_log
from mmcv import imresize, dump

from .registry import DATASETS
from .pipelines import Compose
from ..localization import action_detection, eval_ap, nms
from ..core import average_recall_at_avg_proposals, top_k_accuracy, mean_class_accuracy


def sampling(array, num_frames):
    idx = np.round(np.linspace(0, len(array) - 1, num_frames)).astype(int)
    return array[idx]


@DATASETS.register_module()
class APNFeaturesDataset(Dataset):
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
        ann_files (str): Path to the annotation files.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefixes (str): Path to a directories where video frames are held.
            Default: None.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
                            Default: 'RGB'.
    """

    def __init__(self,
                 ann_files,
                 pipeline,
                 num_stages=100,
                 untrimmed=False,
                 test_mode=False,
                 proposal_mode=False):
        super().__init__()
        self.ann_files = ann_files if isinstance(ann_files, (list, tuple)) else [ann_files]
        self.num_stages = num_stages
        self.untrimmed = untrimmed
        self.proposal_mode = proposal_mode
        # useless
        self.start_index = 0
        self.stride = 1
        self.test_mode = test_mode

        self.pipeline = Compose(pipeline)

        self.gt_infos, self.video_infos = self.load_gt_infos()
        self.snippet_infos = []
        for ann_file in self.ann_files:
            self.snippet_infos.extend(self.load_annotations(ann_file))


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
                    class_label = int(line_split[4]) if not self.proposal_mode else 0

                    gt_infos.setdefault(class_label, {}).setdefault(video_name, []).append([start_frame, end_frame])

                    video_infos.setdefault(video_name, {}).setdefault('total_frames', total_frames)
                    video_infos.setdefault(video_name, {}).setdefault('gt_bboxes', []).append(
                        [start_frame, end_frame, class_label])


        return gt_infos, video_infos

    def load_annotations(self, ann_file):
        if not self.untrimmed:
            snippet_infos = []
            with open(ann_file, 'r') as fin:
                for line in fin.readlines():
                    line_split = line.strip().split(',')
                    video_name = str(line_split[0])
                    total_frames = int(line_split[1])
                    start_frame = int(line_split[2])
                    end_frame = int(line_split[3])
                    class_label = int(line_split[4]) if not self.proposal_mode else 0

                    for frm_idx in range(start_frame, end_frame + 1, self.stride):
                        snippet_info = {'video_name': video_name,
                                        'frame_index%': frm_idx/total_frames,
                                        'class_label': class_label}
                        progression = (frm_idx - start_frame) / (end_frame - start_frame)
                        snippet_info['progression_label'] = progression
                        snippet_infos.append(snippet_info)

        else:
            snippet_infos = []
            for video_name, video_info in self.video_infos.items():
                total_frames = video_info['total_frames']
                for frm_idx in range(self.start_index, self.start_index + total_frames, self.stride):
                    snippet_info = {'video_name': video_name,
                                    'frame_index': frm_idx}
                    snippet_infos.append(snippet_info)
        return snippet_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.snippet_infos[idx])
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.snippet_infos[idx])
        return self.pipeline(results)

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return dump(results, out)

    def decode_results(self, results):
        decoded_results = {}
        # decode results
        if self.untrimmed:
            predictions = results
        else:
            # results include loss
            predictions, decoded_results['losses'] = zip(*results)

        if type(predictions[0]) is not list and type(predictions[0]) is not tuple:
            decoded_results['predictions'] = predictions
        elif len(predictions[0]) == 1:
            # predictions include two different types of progressions
            # decoded_results['argmax_progressions'], decoded_results['exception_progressions'] = zip(*predictions)
            decoded_results['exception_progressions'] = predictions
        elif len(predictions[0]) == 3:
            # predictions include two different types of progressions and classification scores
            decoded_results['argmax_progressions'], decoded_results['exception_progressions'], decoded_results['classification_scores'] = zip(*predictions)
            decoded_results.pop('classification_scores')
        else:
            decoded_results['predictions'] = predictions

        return decoded_results

    def evaluate(self,
                 results,
                 metrics='mae',
                 metric_options={},
                 dataset_name='Val',
                 dump_detections=False,
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
        allowed_metrics = ['mae', 'loss', 'mAP', 'AR@AN', 'top_k_based_progression', 'framecls_top_k_accuracy',
                           'framecls_mean_class_accuracy', 'framecls_top_k_accuracy_based_on_apn', 'framecls_mean_class_accuracy_based_on_apn']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        # decode results
        results = self.decode_results(results)
        # validation main
        eval_results = {}
        for metric in metrics:
            msg = f'Evaluating {metric} on {dataset_name} dataset...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'loss':
                loss = sum(results['losses']) / len(results['losses'])
                eval_results['loss'] = loss
                log_msg = f'\n{dataset_name} loss\t{loss:.2f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mae':
                prog_labels = self.get_progression_labels(denormalized=True)
                if 'predictions' in results:
                    progs = results['predictions']
                    mae = np.mean(np.abs(progs - prog_labels))
                    # Normalized to range [0, 100] if num_stages is not 100
                    mae *= 100 / self.num_stages
                    eval_results['mae'] = mae
                    log_msg = f'\n{dataset_name} mae\t{mae:.2f}'
                    print_log(log_msg, logger=logger)
                    continue
                elif 'exception_progressions' in results:
                    arg_progre, excp_progre = results['argmax_progressions'], results['exception_progressions']
                    arg_mae = np.mean(np.abs(arg_progre - prog_labels))
                    excp_mae = np.mean(np.abs(excp_progre - prog_labels))
                    arg_mae *= 100 / self.num_stages
                    excp_mae *= 100 / self.num_stages
                    eval_results['arg_mae'] = arg_mae
                    eval_results['excp_mae'] = excp_mae
                    eval_results['mae'] = excp_mae
                    log_msg1 = f'\n{dataset_name} arg_mae\t{arg_mae:.2f}'
                    log_msg2 = f'\n{dataset_name} excp_mae\t{excp_mae:.2f}'
                    print_log(log_msg1, logger=logger)
                    print_log(log_msg2, logger=logger)
                    continue
                else:
                    raise ValueError(f"No progressions in results")

            if metric == 'mAP':
                assert self.untrimmed is True, "To compute mAP, the dataset must be untrimmed videos"
                min_e = metric_options.setdefault('mAP', {}).setdefault('min_e', 60)
                max_s = metric_options.setdefault('mAP', {}).setdefault('max_s', 40)
                min_L = metric_options.setdefault('mAP', {}).setdefault('min_L', 60)
                score_T = metric_options.setdefault('mAP', {}).setdefault('score_T', 0.0)
                highest_iou = metric_options.setdefault('mAP', {}).setdefault('highest_iou', 0.4)
                top_k = metric_options.setdefault('mAP', {}).setdefault('top_k', np.inf)
                method = metric_options.setdefault('mAP', {}).setdefault('method', 'mse')

                if 'predictions' in results:
                    progs = results['predictions']
                else:
                    progs = results['exception_progressions']
                if 'classification_scores' in results:
                    progs = [[p, c] for p, c in zip(progs, results['classification_scores'])]

                progs = self.split_results_by_video(np.array(progs))
                detections = {}
                rescale_len = 1000
                for video_name, progs_by_v in progs.items():
                    original_len = progs_by_v.shape[0]
                    if progs_by_v.ndim == 3:
                        p_by_v = progs_by_v[:, 0, :]
                        cs_by_v = progs_by_v[:, 1, :]
                    else:
                        p_by_v = progs_by_v
                        cs_by_v = None
                    scale_ratio = original_len / rescale_len

                    detections[video_name] = {}
                    for class_ind, p_by_v_by_c in enumerate(p_by_v.T):
                        p = imresize(p_by_v_by_c[np.newaxis, ...].astype(np.float), (rescale_len, 1)).squeeze()
                        # p = sampling(p_by_v_by_c, rescale_len)
                        det_by_v_by_c = action_detection(p, min_e=min_e, max_s=max_s, min_L=min_L / scale_ratio,
                                                         score_threshold=score_T, method=method)
                        det_by_v_by_c[:, :2] *= (scale_ratio * self.stride)
                        det_by_v_by_c[:, :2] += float(self.start_index)
                        # if cs_by_v is not None:
                        #     cs_by_v_by_c = cs_by_v[:, class_ind]
                        #     cs_by_det = []
                        #     for row in det_by_v_by_c:
                        #         cs_by_det.append((cs_by_v_by_c[row[0].astype(int): row[1].astype(int)+1]).mean())
                        #     det_by_v_by_c[:, 2] *= cs_by_det
                        #
                        # det_by_v_by_c = nms(det_by_v_by_c, highest_iou, top_k)
                        detections[video_name][class_ind] = det_by_v_by_c

                if dump_detections:
                    dump(detections, dump_detections)
                detections = self.format_det_for_func_eval_ap(detections)
                iou_range = np.arange(0.1, 1.0, .1)
                ap_values = eval_ap(detections, self.gt_infos, iou_range)
                mAP = ap_values.mean(axis=0)
                for iou, map_iou in zip(iou_range, mAP):
                    eval_results[f'mAP@{iou:.02f}'] = map_iou
                continue

            if metric == 'videocls_top_k_based_on_apn':
                video_labels = [video_info['gt_bboxes'][0][2] for video_info in self.video_infos.values()]
                progressions_dict = self.split_results_by_video(results['predictions'])
                scores = []
                for video_name, progs_by_v in progressions_dict.items():
                    this_video_length = progs_by_v.shape[1]
                    action_template = np.linspace(0, 100, this_video_length)
                    mse = ((action_template - progs_by_v) ** 2).mean(axis=1)
                    score_by_v = -mse / 1666.66 + 1
                    scores.append(score_by_v)
                eval_results = top_k_accuracy_metric(scores, video_labels, metric_options, eval_results, logger)
                continue

            if metric == 'framecls_top_k_accuracy':
                gt_labels = [frame_info['class_label'] for frame_info in self.snippet_infos]
                eval_results = top_k_accuracy_metric(results['predictions'], gt_labels, metric_options, eval_results, logger)
                continue

            if metric == 'framecls_mean_class_accuracy':
                gt_labels = [frame_info['class_label'] for frame_info in self.snippet_infos]
                mean_acc = mean_class_accuracy(results['predictions'], gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'framecls_top_k_accuracy_based_on_apn':
                gt_labels = [frame_info['class_label'] for frame_info in self.snippet_infos]
                eval_results = top_k_accuracy_metric(results['classification_scores'], gt_labels, metric_options,
                                                     eval_results, logger)
                continue

            if metric == 'framecls_mean_class_accuracy_based_on_apn':
                gt_labels = [frame_info['class_label'] for frame_info in self.snippet_infos]
                mean_acc = mean_class_accuracy(results['classification_scores'], gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            # Refer to 'activitynet_dataset.evaluate(metric='AR@AN')'
            if metric == 'AR@AN':
                temporal_iou_thresholds = metric_options.setdefault(
                    'AR@AN', {}).setdefault('temporal_iou_thresholds',
                                            np.linspace(0.5, 0.95, 10))
                max_avg_proposals = metric_options.setdefault(
                    'AR@AN', {}).setdefault('max_avg_proposals', 100)
                if isinstance(temporal_iou_thresholds, list):
                    temporal_iou_thresholds = np.array(temporal_iou_thresholds)

                ground_truth = self._import_prop_ground_truth()
                proposal, num_proposals = self._import_proposals(results['predictions'])
                recall, _, _, auc = (
                    average_recall_at_avg_proposals(
                        ground_truth,
                        proposal,
                        num_proposals,
                        max_avg_proposals=max_avg_proposals,
                        temporal_iou_thresholds=temporal_iou_thresholds))
                eval_results['auc'] = auc
                eval_results['AR@1'] = np.mean(recall[:, 0])
                eval_results['AR@5'] = np.mean(recall[:, 4])
                eval_results['AR@10'] = np.mean(recall[:, 9])
                eval_results['AR@50'] = np.mean(recall[:, 49])
                eval_results['AR@100'] = np.mean(recall[:, 99])
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
            total_frames = data['total_frames']
            selected_frames = len(range(self.start_index, self.start_index + total_frames, self.stride))
            p_by_video = results[cum_frames: cum_frames + selected_frames]
            split_results[video_name] = p_by_video * (100 / self.num_stages)
            cum_frames += selected_frames
        assert cum_frames == len(results), "total frames from ds.video_infos and results not equal."
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
                det_by_v_by_c = np.hstack([video_name, class_id, det_by_v_by_c * self.stride + self.start_index])
                det_by_c.append(det_by_v_by_c)
            if len(det_by_c) == 0:
                formated_detections[class_ind] = np.empty((1, 5))
            else:
                formated_detections[class_ind] = np.vstack(det_by_c)
        return formated_detections

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.snippet_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)
        else:
            return self.prepare_train_frames(idx)

    def get_progression_labels(self, denormalized=True):
        progression_labels = np.array(
            [frame_info['progression_label'] for frame_info in self.snippet_infos])  # [N,] in range [0,1]
        if denormalized:
            progression_labels *= self.num_stages  # [N,] in range [0, num_stages]
        return progression_labels

    def _import_prop_ground_truth(self):
        """Refer to function with same name in activitynet_dataset.py"""
        ground_truth = {}
        for video_id, video_info in self.video_infos.items():
            this_video_ground_truths = video_info['gt_bboxes']
            ground_truth[video_id] = np.array(this_video_ground_truths)
        return ground_truth

    def _import_proposals(self, progressions):
        """Refer to function with same name in activitynet_dataset.py"""
        progressions = self.split_results_by_video(progressions)
        detections = {}
        num_detections = 0
        rescale_len = 1000
        for video_name, p_by_v in progressions.items():
            original_len = p_by_v.shape[1]
            scale_ratio = original_len / rescale_len
            p = imresize(p_by_v.squeeze()[np.newaxis, ...].astype(np.float), (rescale_len, 1)).squeeze()
            det_by_v_by_c = action_detection(p, min_e=60, max_s=40, min_L=60 / scale_ratio)
            det_by_v_by_c = nms(det_by_v_by_c)
            det_by_v_by_c[:, :2] *= scale_ratio
            detections[video_name] = det_by_v_by_c
            num_detections += det_by_v_by_c.shape[0]
        return detections, num_detections

    def split_result_to_trimmed(self, results):
        video_start_index = {}
        cum_frames = 0
        for video_name, video_info in self.video_infos.items():
            video_start_index[video_name] = cum_frames
            cum_frames += video_info['total_frames']

        trimmed_results = []
        for frame_info in self.snippet_infos:
            video_name = frame_info['frame_dir'].split('/')[-1]
            frame_idx = frame_info['frame_index']
            start_index = video_start_index[video_name]
            trimmed_results.append(results[start_index+frame_idx])

        return trimmed_results

def top_k_accuracy_metric(cls_scores, gt_labels, metric_options, eval_results, logger):
    topk = metric_options.setdefault('top_k_accuracy',{}).setdefault('topk', (1, 5))
    if not isinstance(topk, (int, tuple)):
        raise TypeError('topk must be int or tuple of int, '
                        f'but got {type(topk)}')
    if isinstance(topk, int):
        topk = (topk,)

    top_k_acc = top_k_accuracy(cls_scores, gt_labels, topk)

    log_msg = []
    for k, acc in zip(topk, top_k_acc):
        eval_results[f'top{k}_acc'] = acc
        log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
    log_msg = ''.join(log_msg)
    print_log(log_msg, logger=logger)

    return eval_results

