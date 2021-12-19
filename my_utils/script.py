# import mmcv
# import numpy as np
# import pandas as pd
# import json
# import time
# from matplotlib import pyplot as plt
# from mmcv import Config
# from pathlib import Path
# from mmaction.models import build_model, builder
# from mmaction.datasets import build_dataset
# from mmaction.localization import eval_ap
# from mmaction.localization.apn_utils import plot_detection, plot_prediction


# %% check if generated raw frames correct
# from pathlib import Path
# from mmcv import imread
# import numpy as np
# p = Path("/home/louis/PycharmProjects/APN/my_data/ucf101/rawframes")
# shape = []
# for a in p.iterdir():
#     for b in a.iterdir():
#         i = list(b.glob("img_*"))[0]
#         s = imread(i).shape
#         shape.append(s)
#         imgs = len(list(b.glob("img_*")))
#         flow_x = len(list(b.glob("flow_x_*")))
#         flow_y = len(list(b.glob("flow_y_*")))
#         if flow_x ==0 or flow_y ==0 or imgs == 0:
#             raise ValueError('c')
#         if flow_x != flow_y:
#             raise ValueError
#         if imgs != flow_x+1:
#             raise ValueError("b")
# shape = np.array(shape)
# u_shape = np.unique(shape, axis=0)
# %% ucf101 apn file generation
# p = "my_data/ucf101/annotations/trainlist01.txt"
# new = "my_data/ucf101/annotations/apn_trainlist01.csv"
# frames = "my_data/ucf101/rawframes"
# action_ind = pd.read_csv("my_data/ucf101/annotations/classInd.txt", header=None, index_col=1, sep=' ').T
#
# with open(p, 'r') as f:
#     with open(new, 'w') as n:
#         for line in f:
#             folder = line.split('.')[0]
#             action = folder.split('/')[0]
#             ind = action_ind[action].values[0] - 1
#             raw_frames = Path(frames, folder)
#             total_frame = len(list(raw_frames.glob("flow_y_*.jpg")))
#             n.write(f"{folder},{total_frame},0,{total_frame-1},{ind}\n")

# %% activitynet1.3 apn file generation
# raw_frame_path = 'my_data/ActivityNet/rawframes_5fps'
# with open("my_data/ActivityNet/annotations/activity_1_3.json", 'r') as f:
#     ann = json.load(f)['database']
#
# with open("my_data/ActivityNet/annotations/aty_classind.json", 'r') as f2:
#     classind = json.load(f2)
#
# with open("my_data/ActivityNet/apn_aty_train_5fps.csv", 'w') as n:
#     for video_id, video_info in ann.items():
#         if video_info['subset'] == 'training':
#             total_frames = len(list(Path(raw_frame_path, 'v_'+video_id).iterdir()))
#             for seg in video_info['annotations']:
#                 start, end = np.array(seg['segment'])/video_info['duration']*total_frames
#                 indlabel = classind[seg['label']]
#                 n.write(f"{video_id},{total_frames},{round(start)},{round(end)},{indlabel}\n")
#
# with open("my_data/ActivityNet/apn_aty_test_5fps.csv", 'w') as n:
#     for video_id, video_info in ann.items():
#         if video_info['subset'] == 'testing':
#             total_frames = len(list(Path(raw_frame_path, 'v_'+video_id).iterdir()))
#             start, end = 0, 0
#             indlabel = 0
#             n.write(f"{video_id},{total_frames},{start},{end},{indlabel}\n")

# cfg = Config.fromfile("configs/localization/apn/apn_coral_r3dsony_64x2x1_10e_thumos14_flow.py")
# model = build_model(cfg.model)
# ds = build_dataset(cfg.data.train)
# ds.__getitem__(70)

# ## Check the completeness of the activitynet1-3
# import json
# with open('/media/louis/louis4-harddisk/Datasets/ActivityNet/annotations/activity_1_3.json', 'r') as f:
#     t = json.load(f)
# data = t['database']
# del t


### Check score threshhold after nms
# from mmcv import Config
# from mmaction.localization import eval_ap
# import json
# from mmaction.datasets import build_dataset
# import numpy as np
#
# model_name = 'apn_sord3_r3dsony_48x1x1_10e_thumos14_rgb'
# cfg = Config.fromfile(f"configs/localization/apn/{model_name}.py")
# ds = build_dataset(cfg.data.test)
# work_dir = f'work_dirs/{model_name}'
# with open(f"{work_dir}/detections_wo_classificationscores.json", 'r') as f:
#     det = json.load(f)
#
#
# def format_det(det, score_thre):
#     for det_by_v in det.copy().values():
#         for class_ind in list(det_by_v):
#             det_by_v_by_c = np.array(det_by_v.pop(class_ind)).reshape(-1, 3)
#             det_by_v_by_c = det_by_v_by_c[np.where(det_by_v_by_c[:, -1].astype(float) > score_thre)]
#             det_by_v[int(class_ind)] = det_by_v_by_c
#     return det
#
#
#
# mAP = {}
# max = 0
# for score_thre in np.arange(0, 1, 0.1):
#     # score_thre = 0
#     formated_det = format_det(det, score_thre)
#     formated_det = ds.format_det_for_func_eval_ap(formated_det)
#     # iou_range = np.arange(0.1, 1.0, .1)
#     iou_range = (0.5,)
#     ap_values = eval_ap(formated_det, ds.gt_infos, iou_range)
#     grade = ap_values.mean(axis=0)
#     mAP[score_thre] = grade
#     if grade > max:
#         max = grade
#         thre = score_thre

# thre = 0.0 is the best


## Resize Images
# from pathlib import Path
# import mmcv
# p = Path("my_data/DFMAD-70/Images/test")
# for img_folder in p.iterdir():
#     for img_p in img_folder.iterdir():
#         new_path = list(img_p.parts)
#         new_path[img_p.parts.index('test')] = 'resized_test'
#         new_path = Path(*new_path)
#         img = mmcv.imread(img_p)
#         out_img = mmcv.imresize(img, (320, 180))
#         mmcv.imwrite(out_img, f'{new_path}')

## Generate standard proposal
#
# import argparse
#
# import mmcv
# import numpy as np
#
# from mmaction.core import pairwise_temporal_iou
#
#
# def load_annotations(ann_file):
#     """Load the annotation according to ann_file into video_infos."""
#     video_infos = []
#     anno_database = mmcv.load(ann_file)
#     for video_name in anno_database:
#         video_info = anno_database[video_name]
#         video_info['video_name'] = video_name
#         video_infos.append(video_info)
#     return video_infos
#
#
# def import_ground_truth(video_infos, activity_index):
#     """Read ground truth data from video_infos."""
#     ground_truth = {}
#     for video_info in video_infos:
#         video_id = video_info['video_name']
#         this_video_ground_truths = []
#         for ann in video_info['annotations']:
#             t_start, t_end = ann['segment']
#             label = activity_index[ann['label']]
#             this_video_ground_truths.append([t_start, t_end, label])
#         ground_truth[video_id] = np.array(this_video_ground_truths)
#     return ground_truth
#
#
# def import_proposals(result_dict):
#     """Read predictions from result dict."""
#     proposals = {}
#     num_proposals = 0
#     for video_id in result_dict:
#         result = result_dict[video_id]
#         this_video_proposals = []
#         for proposal in result:
#             t_start, t_end = proposal['segment']
#             score = proposal['score']
#             this_video_proposals.append([t_start, t_end, score])
#             num_proposals += 1
#         proposals[video_id] = np.array(this_video_proposals)
#     return proposals, num_proposals
#
#
# def dump_formatted_proposal(video_idx, video_id, num_frames, fps, gts,
#                             proposals, tiou, t_overlap_self,
#                             formatted_proposal_file):
#     """dump the formatted proposal file, which is the input proposal file of
#     action classifier (e.g: SSN).
#     Args:
#         video_idx (int): Index of video.
#         video_id (str): ID of video.
#         num_frames (int): Total frames of the video.
#         fps (float): Fps of the video.
#         gts (np.ndarray[float]): t_start, t_end and label of groundtruths.
#         proposals (np.ndarray[float]): t_start, t_end and score of proposals.
#         tiou (np.ndarray[float]): 2-dim array with IoU ratio.
#         t_overlap_self (np.ndarray[float]): 2-dim array with overlap_self
#             (union / self_len) ratio.
#         formatted_proposal_file (open file object): Open file object of
#             formatted_proposal_file.
#     """
#
#     formatted_proposal_file.write(
#         f'#{video_idx}\n{video_id}\n{num_frames}\n{fps}\n{gts.shape[0]}\n')
#     for gt in gts:
#         formatted_proposal_file.write(f'{int(gt[2])} {int(gt[0])} {int(gt[1])}\n')
#     formatted_proposal_file.write(f'{proposals.shape[0]}\n')
#
#     best_iou = np.amax(tiou, axis=0)
#     best_iou_index = np.argmax(tiou, axis=0)
#     best_overlap = np.amax(t_overlap_self, axis=0)
#     best_overlap_index = np.argmax(t_overlap_self, axis=0)
#
#     for i in range(proposals.shape[0]):
#         index_iou = best_iou_index[i]
#         index_overlap = best_overlap_index[i]
#         label_iou = gts[index_iou][2]
#         label_overlap = gts[index_overlap][2]
#         if label_iou != label_overlap:
#             label = label_iou if label_iou != 0 else label_overlap
#         else:
#             label = label_iou
#         if best_iou[i] == 0 and best_overlap[i] == 0:
#             formatted_proposal_file.write(
#                 f'0 0 0 {int(proposals[i][0])} {int(proposals[i][1])}\n')
#         else:
#             formatted_proposal_file.write(
#                 f'{int(label)} {best_iou[i]} {best_overlap[i]} '
#                 f'{int(proposals[i][0])} {int(proposals[i][1])}\n')
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='convert proposal format')
#     parser.add_argument(
#         '--ann-file',
#         type=str,
#         default='/home/louis/PycharmProjects/mmaction2/my_data/dfmad70/Annotations/for_bmn/dfmad_anno_train.json',
#         help='name of annotation file')
#     parser.add_argument(
#         '--activity-index-file',
#         type=str,
#         default='../../../my_data/ActivityNet/anet_activity_indexes_val.txt',
#         help='name of activity index file')
#     parser.add_argument(
#         '--proposal-file',
#         type=str,
#         default='/home/louis/PycharmProjects/GTAD/output/test/gtad_dfmad_train_proposal.json',
#         help='name of proposal file, which is the'
#              'output of proposal generator (BMN)')
#     parser.add_argument(
#         '--formatted-proposal-file',
#         type=str,
#         default='/home/louis/PycharmProjects/mmaction2/dfmad_gtad_train_200_proposal.txt',
#         help='name of formatted proposal file, which is the'
#              'input of action classifier (SSN)')
#     args = parser.parse_args()
#
#     return args
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     formatted_proposal_file = open(args.formatted_proposal_file, 'w')
#
#     # The activity index file is constructed according to
#     # 'https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_classification.py'
#     # activity_index, class_idx = {}, 0
#     # for line in open(args.activity_index_file).readlines():
#     #     activity_index[line.strip()] = class_idx
#     #     class_idx += 1
#     activity_index = {
#         0: 0,
#         1: 1,
#         2: 2,
#     }
#
#     video_infos = load_annotations(args.ann_file)
#     ground_truth = import_ground_truth(video_infos, activity_index)
#     proposal, num_proposals = import_proposals(
#         mmcv.load(args.proposal_file)['results'])
#     video_idx = 0
#
#     for video_info in video_infos:
#         video_id = video_info['video_name']
#         num_frames = video_info['duration_frame']
#         fps = video_info['fps']
#         ground_truth[video_id][:, :2] *= 30
#         tiou, t_overlap = pairwise_temporal_iou(proposal[video_id][:, :2].astype(float),
#                                                 ground_truth[video_id][:, :2].astype(float),
#                                                 calculate_overlap_self=True)
#
#         dump_formatted_proposal(video_idx, video_id, num_frames, fps,
#                                 ground_truth[video_id], proposal[video_id],
#                                 tiou, t_overlap, formatted_proposal_file)
#         video_idx += 1
#     formatted_proposal_file.close()

# ## load model
# import numpy as np
# import torch
# import time
# from mmcv import Config, dump
# from mmcv.runner import get_dist_info, init_dist, load_checkpoint
# from mmaction.datasets import build_dataset
# from mmaction.models import build_model
#
# model_name = 'apn_coralrandom_r3dsony_32x4_10e_thumos14_flow'
# cfg = Config.fromfile(f"configs/localization/apn/{model_name}.py")
# model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
# load_checkpoint(model, f"work_dirs/{model_name}/epoch_10.pth", map_location='cpu')

# # change level 1 to level 2
# from pathlib import Path
# import pandas as pd
# from mmcv import track_iter_progress
# p = '/media/louis/louis-portable2/kinetics-dataset/train'
# f = '/home/louis/PycharmProjects/APN/my_data/kinetics400/annotations/kinetics_train.csv'
#
# p = Path(p)
# f = pd.read_csv(f)
#
# for i in track_iter_progress(list(p.iterdir())):
#     id = '_'.join(i.stem.split('_')[:-2])
#     label = f.loc[f['youtube_id'] == id, 'label'].to_list()
#     if len(label) == 1:
#         pass
#     elif len(label) > 1:
#         print(f"{id} got multiple labels: {label}")
#     elif len(label) == 0:
#         print(f"{id} got no labels")
#     assert len(label) == 1
#     label = label[0]
#     p.joinpath(label).mkdir(exist_ok=True)
#     i.rename(p.joinpath(label).joinpath(i.name))



# #%% check resized videos
# #find -name "*.mp4" -exec sh -c "echo '{}' >> errors.log; ffmpeg -v error -i '{}' -map 0:1 -f null - 2>> errors.log" \;
# from pathlib import Path
# from mmcv import track_iter_progress
# r = "/home/louis/PycharmProjects/APN/my_data/fineaction/webm_resized"
# v = "/home/louis/PycharmProjects/APN/my_data/fineaction/videos_webm"
# l = list(i.name for i in Path(v).iterdir())
#
# def get_framenum(path):
#     n = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
#     return n
#
# for i in track_iter_progress(l):
#     if 'mp4' in i:
#         continue
#     t1 = get_framenum(str(r + f'/{i.replace(".webm", ".mp4")}'))
#     t2 = get_framenum(str(v + f'/{i}'))
#     if t1 / t2 > 1.02 or t1 / t2 < 0.98:
#         print('\n', i, t1, t2)

import my_models
a = 1