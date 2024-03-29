import copy
import os.path as osp
from itertools import repeat
from multiprocessing import cpu_count
from collections import OrderedDict
import numpy as np
from mmaction.evaluation import top_k_accuracy
from mmaction.registry import DATASETS
from mmengine.dataset import Compose
from mmengine import print_log, dump, track_parallel_progress
from torch.utils.data import Dataset
from warnings import warn

from ..apn_utils import apn_detection_on_single_video, uniform_sampling_1d
from custom_modules.mmdet_utils import bbox2result
from custom_modules.mmdet_utils import eval_map


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
        self.frame_infos = self.load_annotations()

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
        for ann_file, data_prefix in zip(self.ann_files, self.data_prefixes):
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

                    gt_infos.setdefault(class_label, {}).setdefault(video_name, []).append([start_frame, end_frame])

                    video_infos.setdefault(video_name, {}).setdefault('total_frames', total_frames)
                    video_infos.setdefault(video_name, {}).setdefault('gt_bboxes', []).append([start_frame, end_frame])
                    video_infos.setdefault(video_name, {}).setdefault('gt_labels', []).append(class_label)
                    video_infos.setdefault(video_name, {}).setdefault('rescale', total_frames / self.test_sampling)

        return gt_infos, video_infos

    def load_annotations(self):
        if not self.untrimmed:
            # Training and Validation dataset (trimmed)
            frame_infos = []
            for video_name, video_info in self.video_infos.items():
                total_frames, gt_bboxes, gt_labels = video_info['total_frames'], video_info['gt_bboxes'], video_info[
                    'gt_labels']
                for (start_f, end_f), label in zip(gt_bboxes, gt_labels):
                    # if end_f - start_f + 1 <= 30:
                    #     continue
                    for frm_idx in range(start_f, end_f + 1):
                        frame_info = dict(filename=video_name) if self.modality == 'Video' else dict(
                            frame_dir=video_name,
                            total_frames=total_frames)
                        progression_label = (frm_idx - start_f) / (end_f - start_f)
                        frame_info.update(dict(frame_index=frm_idx,
                                               class_label=label,
                                               progression_label=progression_label))
                        frame_infos.append(frame_info)
                        # if progression_label < 0.05:
                        #     frame_infos.extend(([frame_info] * 1))
        else:
            # Testing dataset (untrimmed)
            frame_infos = []
            for video_name, video_info in self.video_infos.items():
                # video_name = osp.join(data_prefix, video_name)
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
                         iou_thr=0.5,
                         search=dict(
                             min_e=60,
                             max_s=40,
                             min_L=60,
                             method='mse'),
                         nms=dict(
                             score_thr=0,
                             max_per_video=-1,
                             nms=dict(iou_thr=0.4)))),
                 logger=None):
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['top_k_accuracy', 'MAE', 'mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                warn(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy', {}).setdefault('topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk,)

                cls_score, _ = map(np.array, zip(*results))
                del _
                if self.untrimmed:
                    sampled_idx_pre, _, cls_label = self.get_sample_points_on_untrimmed(return_cls_label=True)
                    cls_score = cls_score[sampled_idx_pre]
                else:
                    cls_label = np.array([frame_info['class_label'] for frame_info in self.frame_infos])

                top_k_acc = top_k_accuracy(cls_score, cls_label, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                del cls_score, cls_label, sampled_idx_pre, _
                continue

            if metric == 'MAE':
                _, progression = map(np.array, zip(*results))
                del _
                if self.untrimmed:
                    sampled_idx_pre, _, gt_progression = self.get_sample_points_on_untrimmed(return_gt_progs=True)
                    progression = progression[sampled_idx_pre]
                    del sampled_idx_pre, _
                else:
                    gt_progression = np.array(
                        [frame_info['progression_label'] * 100 for frame_info in self.frame_infos])
                gt_progression, progression = map(np.squeeze, (gt_progression, progression))
                MAE = np.abs(gt_progression - progression).mean()
                eval_results['MAE'] = MAE
                log_msg = f'\nMAE\t{MAE:.2f}'
                print_log(log_msg, logger=logger)
                del progression, gt_progression
                continue

            if metric == 'mAP':
                # Derive bboxes and scores
                assert self.untrimmed is True, "To compute mAP, the dataset must be untrimmed"
                results_vs_video = self.split_results_by_video(results)
                det_results = self.apn_action_detection(results_vs_video, **metric_options.get('mAP', {}))

                if False:
                    from mmcv import dump
                    video_names = [vn.split('/')[-1] for vn in self.video_infos.keys()]
                    cls_score, progs = zip(*results)
                    dump({k: v for k, v in zip(video_names, self.split_results_by_video(progs))}, "progression.pkl")
                    dump({k: v for k, v in zip(video_names, self.split_results_by_video(cls_score))}, "classification.pkl")
                    dump({k: v for k, v in zip(video_names, det_results)}, "detection.pkl")

                # Computer mAP
                iou_thr = metric_options.get('mAP', {}).get('iou_thr', 0.5)
                iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
                mean_aps = []
                annotations = self.get_ann_info()
                for iou_thr in iou_thrs:
                    print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                    mean_ap, _ = eval_map(det_results,
                                          annotations,
                                          iou_thr=iou_thr,
                                          dataset=self.CLASSES,
                                          logger=logger)
                    mean_aps.append(mean_ap)
                    eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                eval_results[f'mAP'] = sum(mean_aps) / len(mean_aps)
                continue

        return eval_results

    def split_results_by_video(self, results):
        results_vs_video = []
        for i in range(0, len(self), self.test_sampling):
            results_by_video = results[i: i + self.test_sampling]
            results_vs_video.append(results_by_video)
        return results_vs_video

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

    def get_ann_info(self):
        ann_info = []
        for video_info in self.video_infos.values():
            ann = dict(bboxes=np.array(video_info['gt_bboxes']) / video_info['rescale'],
                       labels=np.array(video_info['gt_labels']))
            ann_info.append(ann)
        return ann_info

    def apn_action_detection(self, results, nproc=cpu_count(), **kwargs):
        rescale = [video_info['rescale'] for video_info in self.video_infos.values()]
        video_names = list(self.video_infos.keys())  # For debug purpose
        det_results = track_parallel_progress(apn_detection_on_single_video,
                                              list(zip(video_names, results, rescale, repeat(kwargs))),
                                              nproc,
                                              keep_order=True)
        # TODO: improve by making each video only can detect no more than 5 classes of actions?
        det_results = [bbox2result(det_bboxes, det_labels, len(self.gt_infos)) for det_bboxes, det_labels in
                       det_results]

        return det_results

    def get_sample_points_on_untrimmed(self, return_cls_label=False, return_gt_progs=False):
        """ Only action frames have progressions labels."""
        assert self.untrimmed
        cum_frames = 0
        pre_idx, gt_idx, gt_progs, cls_labels = [], [], [], []
        for video_name, video_info in self.video_infos.items():
            sampled_frame = np.linspace(0, video_info['total_frames'] - 1, self.test_sampling, dtype=int)
            sampled_frame, sampled_idx = np.unique(sampled_frame, return_index=True)
            for gt_bbox, class_label in zip(video_info['gt_bboxes'], video_info['gt_labels']):
                action_start, action_end = gt_bbox
                action_frame = np.arange(action_start, action_end + 1)
                progs_by_action = np.linspace(0, 1, len(action_frame))
                cls_label_by_action = np.full(len(action_frame), class_label)
                idx1 = sampled_idx[np.where(np.in1d(sampled_frame, action_frame))[0]]
                idx2 = np.where(np.in1d(action_frame, sampled_frame))[0]
                if idx1.size == 0:
                    continue
                pre_idx.extend(idx1 + cum_frames)
                gt_idx.extend(idx2)
                gt_progs.extend(progs_by_action[idx2] * 100)
                cls_labels.extend(cls_label_by_action[idx2])
            cum_frames += self.test_sampling
        result = [pre_idx, gt_idx]
        if return_cls_label:
            result.append(cls_labels)
        if return_gt_progs:
            result.append(gt_progs)
        return list(map(np.array, result))


@DATASETS.register_module()
class THUMOS14(APNDataset):
    CLASSES = ('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
               'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
               'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
               'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
               'SoccerPenalty', 'TennisSwing', 'ThrowDiscus',
               'VolleyballSpiking')


@DATASETS.register_module()
class ActivityNet(APNDataset):
    CLASSES = (
        'Applying sunscreen', 'Archery', 'Arm wrestling', 'Assembling bicycle', 'BMX', 'Baking cookies', 'Ballet',
        'Bathing dog', 'Baton twirling', 'Beach soccer', 'Beer pong', 'Belly dance', 'Blow-drying hair',
        'Blowing leaves',
        'Braiding hair', 'Breakdancing', 'Brushing hair', 'Brushing teeth', 'Building sandcastles', 'Bullfighting',
        'Bungee jumping', 'Calf roping', 'Camel ride', 'Canoeing', 'Capoeira', 'Carving jack-o-lanterns',
        'Changing car wheel', 'Cheerleading', 'Chopping wood', 'Clean and jerk', 'Cleaning shoes', 'Cleaning sink',
        'Cleaning windows', 'Clipping cat claws', 'Cricket', 'Croquet', 'Cumbia', 'Curling', 'Cutting the grass',
        'Decorating the Christmas tree', 'Disc dog', 'Discus throw', 'Dodgeball', 'Doing a powerbomb', 'Doing crunches',
        'Doing fencing', 'Doing karate', 'Doing kickboxing', 'Doing motocross', 'Doing nails', 'Doing step aerobics',
        'Drinking beer', 'Drinking coffee', 'Drum corps', 'Elliptical trainer', 'Fixing bicycle', 'Fixing the roof',
        'Fun sliding down', 'Futsal', 'Gargling mouthwash', 'Getting a haircut', 'Getting a piercing',
        'Getting a tattoo',
        'Grooming dog', 'Grooming horse', 'Hammer throw', 'Hand car wash', 'Hand washing clothes', 'Hanging wallpaper',
        'Having an ice cream', 'High jump', 'Hitting a pinata', 'Hopscotch', 'Horseback riding', 'Hula hoop', 'Hurling',
        'Ice fishing', 'Installing carpet', 'Ironing clothes', 'Javelin throw', 'Kayaking', 'Kite flying', 'Kneeling',
        'Knitting', 'Laying tile', 'Layup drill in basketball', 'Long jump', 'Longboarding', 'Making a cake',
        'Making a lemonade', 'Making a sandwich', 'Making an omelette', 'Mixing drinks', 'Mooping floor',
        'Mowing the lawn',
        'Paintball', 'Painting', 'Painting fence', 'Painting furniture', 'Peeling potatoes', 'Ping-pong', 'Plastering',
        'Plataform diving', 'Playing accordion', 'Playing badminton', 'Playing bagpipes', 'Playing beach volleyball',
        'Playing blackjack', 'Playing congas', 'Playing drums', 'Playing field hockey', 'Playing flauta',
        'Playing guitarra', 'Playing harmonica', 'Playing ice hockey', 'Playing kickball', 'Playing lacrosse',
        'Playing piano', 'Playing polo', 'Playing pool', 'Playing racquetball', 'Playing rubik cube',
        'Playing saxophone',
        'Playing squash', 'Playing ten pins', 'Playing violin', 'Playing water polo', 'Pole vault',
        'Polishing forniture',
        'Polishing shoes', 'Powerbocking', 'Preparing pasta', 'Preparing salad', 'Putting in contact lenses',
        'Putting on makeup', 'Putting on shoes', 'Rafting', 'Raking leaves', 'Removing curlers',
        'Removing ice from car',
        'Riding bumper cars', 'River tubing', 'Rock climbing', 'Rock-paper-scissors', 'Rollerblading',
        'Roof shingle removal', 'Rope skipping', 'Running a marathon', 'Sailing', 'Scuba diving', 'Sharpening knives',
        'Shaving', 'Shaving legs', 'Shot put', 'Shoveling snow', 'Shuffleboard', 'Skateboarding', 'Skiing',
        'Slacklining',
        'Smoking a cigarette', 'Smoking hookah', 'Snatch', 'Snow tubing', 'Snowboarding', 'Spinning', 'Spread mulch',
        'Springboard diving', 'Starting a campfire', 'Sumo', 'Surfing', 'Swimming', 'Swinging at the playground',
        'Table soccer', 'Tai chi', 'Tango', 'Tennis serve with ball bouncing', 'Throwing darts',
        'Trimming branches or hedges', 'Triple jump', 'Tug of war', 'Tumbling', 'Using parallel bars',
        'Using the balance beam', 'Using the monkey bar', 'Using the pommel horse', 'Using the rowing machine',
        'Using uneven bars', 'Vacuuming floor', 'Volleyball', 'Wakeboarding', 'Walking the dog', 'Washing dishes',
        'Washing face', 'Washing hands', 'Waterskiing', 'Waxing skis', 'Welding', 'Windsurfing', 'Wrapping presents',
        'Zumba')
