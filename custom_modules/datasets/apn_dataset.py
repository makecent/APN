import copy
import os.path as osp
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import repeat
from torch.utils.data import Dataset
from mmcv.utils import print_log
from mmcv import dump
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import Compose
from mmaction.localization.ssn_utils import eval_ap
from .apn_utils import apn_detection_on_single_video, uniform_sampling_1d


@DATASETS.register_module()
class APNDataset(Dataset):
    """APN dataset for action detection.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    [video_name, total_frames, start_frame, end_frame, class_label]
    Example:
    .. code-block:: txt
    video_test_0000324.mp4,4470,1476,1605,0
    video_test_0000324.mp4,4470,3501,3675,0
    video_test_0000664.mp4,2579,39,144,0
    video_test_0000664.mp4,2579,669,759,0
    video_test_0000664.mp4,2579,1704,1791,0
    video_test_0000714.mp4,5383,786,897,0

    Args:
        ann_files (str) or Tuple(str, ...): Path to the annotation files.
        pipeline: A sequence of data transforms.
        data_prefixes (str) or Tuple(str, ...): Path to a directories where video frames are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
                            Default: 'RGB'.
    """

    def __init__(self,
                 ann_files,
                 pipeline,
                 num_stages=100,
                 data_prefixes=None,
                 untrimmed=False,
                 test_sampling=1000,
                 filename_tmpl='img_{:05}.jpg',
                 start_index=0,
                 modality='RGB',
                 test_mode=False):
        super().__init__()
        self.ann_files = ann_files if isinstance(ann_files, (list, tuple)) else [ann_files]
        data_prefixes = data_prefixes if isinstance(data_prefixes, (list, tuple)) else [data_prefixes]
        self.data_prefixes = [osp.realpath(root) if osp.isdir(root) else root for root in data_prefixes]
        assert len(self.data_prefixes) == len(self.ann_files)
        self.num_stages = num_stages
        self.test_sampling = test_sampling
        self.untrimmed = untrimmed
        self.filename_tmpl = filename_tmpl
        self.start_index = start_index
        assert modality in ['RGB', 'Flow', 'Video']
        self.modality = modality
        self.pipeline = Compose(pipeline)

        self.gt_infos, self.video_infos = self.load_gt_infos()
        self.frame_infos = []
        for ann_file, data_prefix in zip(self.ann_files, self.data_prefixes):
            self.frame_infos.extend(
                self.load_annotations(ann_file, data_prefix))

    def load_gt_infos(self):
        """Generate ground truth from the ann_file, class-level and video-level for different purposes.

        GT in class-level:
        {class_index1:
                {'video_name1': [[s1, e1], [s2, e2], ...],
                 'video_name2': [[s1, e1], [s2, e2], ...],}
        class_index2:
                ...
        ...}

        GT in video-level:
        {video_name1:
            {'total_frames': int,
             'gt_bboxes': [[s1, e1, c1], [s2, e2, c2], ...]}
        video_name2:
                ...
        ...}
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
                    class_label = int(line_split[4])
                    if self.modality != 'Video':
                        video_name = video_name.rsplit('.', 1)[0]

                    gt_infos.setdefault(class_label, {}).setdefault(video_name, []).append([start_frame, end_frame])

                    video_infos.setdefault(video_name, {}).setdefault('total_frames', total_frames)
                    video_infos.setdefault(video_name, {}).setdefault('gt_bboxes', []).append(
                        [start_frame, end_frame, class_label])
                    video_infos.setdefault(video_name, {}).setdefault('rescale_rate', total_frames / self.test_sampling)

        return gt_infos, video_infos

    def load_annotations(self, ann_file, data_prefix):
        # Validation dataset (trimmed)
        if not self.untrimmed:
            frame_infos = []
            with open(ann_file, 'r') as fin:
                for line in fin.readlines():
                    line_split = line.strip().split(',')
                    video_name = str(line_split[0])
                    total_frames = int(line_split[1])
                    start_frame = int(line_split[2])
                    end_frame = int(line_split[3])
                    class_label = int(line_split[4])

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
                        progression_label = (frm_idx - start_frame) / (end_frame - start_frame)
                        frame_info['progression_label'] = progression_label
                        frame_infos.append(frame_info)
        # Testing dataset (untrimmed)
        else:
            frame_infos = []
            for video_name, video_info in self.video_infos.items():
                video_name = osp.join(data_prefix, video_name)
                total_frames = video_info['total_frames']
                frame_inds = list(range(self.start_index, self.start_index + total_frames))
                frame_inds = uniform_sampling_1d(frame_inds, self.test_sampling)

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
                         nms=dict(iou_thr=0.4))),
                 logger=None):
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['MAE', 'mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        results = np.array(results)
        eval_results = {}
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'MAE':
                prog_labels = self.get_progression_labels(denormalized=True)
                MAE = np.abs(results - prog_labels).mean()
                eval_results = self.update_and_print_eval(eval_results, MAE, 'MAE', logger)

            if metric == 'mAP':
                assert self.untrimmed is True, "To compute mAP, the dataset must be untrimmed"
                progs = self.split_results_by_video(results)
                detections = self.action_detection_by_progs(progs, det_kwargs=metric_options.get('mAP', {}))
                detections = self.format_detections(detections)

                iou_range = np.arange(0.1, 1.0, .1)
                ap_values = eval_ap(detections, self.gt_infos, iou_range)
                mAP = ap_values.mean(axis=0)
                for iou, mAP_iou in zip(iou_range, mAP):
                    eval_results = self.update_and_print_eval(eval_results, mAP_iou, f'mAP@{iou:.01f}', logger)
                continue

        return eval_results

    def split_results_by_video(self, results):
        progs_by_video = {}
        cum_frames = 0
        for video_name, data in self.video_infos.items():
            p_by_video = results[cum_frames: cum_frames + self.test_sampling]
            progs_by_video[video_name] = p_by_video
            cum_frames += self.test_sampling
        assert cum_frames == len(results), "total frames from 'video_infos' and 'results' not equal."
        return progs_by_video

    def format_detections(self, detections):
        """Format the detections to input to the predefined function 'eval_ap' in 'ssn_utils',"""
        formatted_detections = {}
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
                formatted_detections[class_ind] = np.empty((1, 5))
            else:
                formatted_detections[class_ind] = np.vstack(det_by_c)
        return dict(sorted(formatted_detections.items()))

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
        progression_labels = np.array([frame_info['progression_label'] for frame_info in self.frame_infos])
        if denormalized:
            progression_labels *= 100
        return progression_labels

    @staticmethod
    def update_and_print_eval(eval_results, result, name, logger=None, decimals=4):
        eval_results[name] = result
        log_msg = f'{name}: \t{result:.{decimals}f}'
        print_log(log_msg, logger=logger)
        return eval_results

    def action_detection_by_progs(self, progressions, det_kwargs, nproc=cpu_count()):
        result_dict = {}
        with Pool(nproc) as pool:
            rescale_rates = [video_info['rescale_rate'] for video_info in self.video_infos.values()]
            video_names, progressions_by_video = zip(*progressions.items())
            all_dets = pool.starmap(
                apn_detection_on_single_video,
                zip(progressions_by_video, rescale_rates, repeat(det_kwargs)))

        for video_name, dets_and_scores in zip(video_names, all_dets):
            result_dict[video_name] = dets_and_scores

        return result_dict

    def get_MAE_on_untrimmed_results(self, results, return_pv=False):
        """ Experimental function, compute MAE based on predicted progressions of untrimmed video"""
        assert self.untrimmed
        results = np.array(results)
        cum_frames = 0
        pre, gt, pv = [], [], []
        for video_name, video_info in self.video_infos.items():
            num_sampling = video_info['total_frames'] if self.test_sampling == 'full' else self.test_sampling
            sampled_frame = np.linspace(0, video_info['total_frames'] - 1, num_sampling, dtype=int)
            sampled_frame, sampled_idx = np.unique(sampled_frame, return_index=True)
            for action_start, action_end, class_label in video_info['gt_bboxes']:
                action_frame = np.arange(action_start, action_end + 1)
                progs = np.linspace(0, 1, len(action_frame))
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
