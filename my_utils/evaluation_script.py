import numpy as np
import json
import time
import pickle
from mmcv import Config, dump
from mmaction.datasets import build_dataset
from dataloader.apn_utils import uniform_1d_sampling


model_name = 'apn_coralrandom_r3dsony_32x4_10e_dfmad_flow'
# if 'sord' in model_name:
#     results_component = ('argmax_progressions', 'exception_progressions')
# elif 'coral' in model_name or 'mae' in model_name:
#     results_component = ('progressions',)
# else:
#     raise ValueError

cfg = Config.fromfile(f"configs/localization/apn/{model_name}.py")
# cfg.data.test.untrimmed = False
ds = build_dataset(cfg.data.test)
work_dir = f'work_dirs/{model_name}'

# Action search and compute mAP
with open(f'results.pkl', 'rb') as f:
    results = pickle.load(f)
print(np.var(results, axis=-1).mean())
# print(np.abs((np.array(results)-50).mean(-1)).mean())
# results = [i[:2] for i in results]
# results = ds.split_result_to_trimmed(results)
# results = ds.gather_results_with_class(results)
# dump_detections = f"{work_dir}/detections.pkl"
# dump_detections = None
metric_options = dict(
    mAP=dict(
        sampling=1000,
        search=dict(
            min_e=80,
            max_s=20,
            min_L=600,
            score_threshold=0.0,
            method='mse'),
        nms=dict(iou_thr=0.4))
)

dump_detections = 'detections.pkl'
before = time.time()
eval_results = ds.evaluate(results,
                           metrics='mAP',
                           # results_component=results_component,
                           metric_options=metric_options,
                           dump_detections=dump_detections,
                           logger=None)
execution_time = time.time() - before
print(f"Takes {execution_time} seconds")
print(eval_results)

eval_results['options'] = metric_options['mAP']
eval_results['execution time'] = execution_time
dump(eval_results, f"{work_dir}/evaluations.json")
# #
# 0 for coral 32x4 rgb
# 1 for coral 32x4 flow
# 2 for sord3 32x1 flow
#
#
#
#
# # Validate
# import torch
# from mmaction.apis import single_gpu_test
# from mmaction.datasets import build_dataloader, build_dataset
# from mmaction.models import build_model
# from mmcv import Config
# from mmcv.parallel import MMDataParallel
# model_name = 'apn_sord2_r3dsony_48x1x1_10e_thumos14_flow'
# pth = f'work_dirs/{model_name}/epoch_1.pth'
# cfg = Config.fromfile(f"configs/localization/apn/{model_name}.py")
# cfg.model.pretrained = pth
#
# eval_cfg = cfg.get('evaluation', {})
# val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
# dataloader_setting = dict(
#     videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
#     workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
#     num_gpus=2,
#     shuffle=False)
# dataloader_setting = dict(dataloader_setting,
#                           **cfg.data.get('val_dataloader', {}))
# val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
# val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
#
# if cfg.get('cudnn_benchmark', False):
#     torch.backends.cudnn.benchmark = True
# cfg.gpu_ids = range(1)
# distributed = False
#
# model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
# model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
# results = single_gpu_test(model, val_dataloader)
# key_score = val_dataset.evaluate(results)
#
#
#
#
#
#
#
# Load detection results
import json
import pickle
import numpy as np
from mmcv import Config, dump
from mmaction.datasets import build_dataset
from mmaction.localization import eval_ap
model_name = 'apn_coralrandom_r3dsony_32x4_10e_thumos14_flow'
cfg = Config.fromfile(f"configs/localization/apn/{model_name}.py")
ds = build_dataset(cfg.data.test)
work_dir = f'work_dirs/{model_name}'
# def format_det(det):
#     for det_by_v in det.copy().values():
#         for class_ind in list(det_by_v):
#             det_by_v[int(class_ind)] = np.array(det_by_v.pop(class_ind)).reshape(-1, 3)
#     return det
# with open(f"{work_dir}/mean_fuse/detections.json", 'r') as f:
#     det = format_det(json.load(f))
with open(f"{work_dir}/mean_fuse/detections.pkl", 'rb') as f:
    det = pickle.load(f)

det = ds.format_det_for_func_eval_ap(det)
iou_range = np.arange(0.1, 1.0, .1)
ap_values = eval_ap(det, ds.gt_infos, iou_range)

eval_results = {}
for i, iou in enumerate(iou_range):
    eval_results[f'mAP@{iou:.02f}'] = ap_values.mean(0)[i]
# dump(eval_results, f"{work_dir}/evaluations.json")



## Fusion 1
from mmcv import dump
import numpy as np
import json
import time
from mmcv import Config
from mmaction.datasets import build_dataset



model_name = 'apn_coralrandom_r3dsony_32x4_10e_thumos14_flow'
cfg = Config.fromfile(f"configs/localization/apn/{model_name}.py")
cfg.data.test.untrimmed = False
ds = build_dataset(cfg.data.test)
work_dir = f'work_dirs/{model_name}'

# load results
with open(f'{work_dir}/results.json', 'r') as f:
    flow_results = json.load(f)
with open(f'work_dirs/apn_coralrandom_r3dsony_32x4_10e_thumos14_rgb/results.json', 'r') as f:
    rgb_results = json.load(f)
with open(f'work_dirs/apn_coral_r3dsony_32x4_10e_thumos14_flow/results.json', 'r') as f:
    extra1 = json.load(f)
# with open(f'work_dirs/apn_coral_r3dsony_32x4_10e_thumos14_rgb/results.json', 'r') as f:
#     extra2 = json.load(f)
# with open(f'work_dirs/apn_sordrandom_r3dsony_32x4_10e_thumos14_flow/results.json', 'r') as f:
#     extra3 = json.load(f)
#     extra3 = [i[1] for i in extra3]
# with open(f'work_dirs/apn_bcerandom_r3dsony_32x4_10e_thumos14_flow/results.json', 'r') as f:
#     extra4 = json.load(f)

def take_mean(list_of_prog):
    t = np.array(list_of_prog)
    mean = t.mean(axis=0)
    return mean.tolist()
#
results = take_mean([flow_results, rgb_results,extra1])
pv = np.var(results, axis=-1).mean()
print(pv)
# print(np.var(flow_results, axis=-1).mean())
# print(np.var(results, axis=-1).mean())
# t = take_mean([rgb_results, flow_results, extra1])
# print(np.var(t, axis=-1).mean())
# del rgb_results, flow_results

results = ds.split_result_to_trimmed(results)
results = ds.gather_results_with_class(results)
# t = ds.split_result_to_trimmed(t)
# t = ds.gather_results_with_class(t)
# flow_results = ds.split_result_to_trimmed(flow_results)
# flow_results = ds.gather_results_with_class(flow_results)
# rgb_results = ds.split_result_to_trimmed(rgb_results)
# rgb_results = ds.gather_results_with_class(rgb_results)
# dump_detections = f"{work_dir}/mean_fuse/detection.pkl"
dump_detections = None
metric_options = dict(
    mAP=dict(
    min_e=60,
    max_s=40,
    min_L=60,
    sampling=1000,
    score_T=0.0,
    highest_iou=0.4,
    top_k=np.inf,
    method='mse',
))
before = time.time()
eval_results = ds.evaluate(results,
                           metrics=['mae'],
                           metric_options=metric_options,
                           results_component=('progressions',),
                           dataset_name='Test',
                           dump_detections=dump_detections,
                           logger=None)
execution_time = time.time()-before
print(f"Takes {execution_time} seconds")

eval_results['options'] = metric_options['mAP']
eval_results['execution time'] = execution_time
eval_results['variance'] = pv
dump(eval_results, f"{work_dir}/cross_fuse/random_mean+ignoreflow.json")





# ##%  Group Fusion
import numpy as np
import pickle
from mmcv import Config
from mmaction.datasets import build_dataset
from mmaction.localization import action_detection, eval_ap, nms, nms_with_cls


def format_det(det):
    for det_by_v in det.copy().values():
        for class_ind in list(det_by_v):
            det_by_v[int(class_ind)] = np.array(det_by_v.pop(class_ind)).reshape(-1, 3)
    return det


def evaluate_det(detections):
    # build return detection without change original detection dict.
    temp_detections = {}
    for key in detections.keys():
        temp_detections[key] = {}
        for i in range(20):
            temp_detections[key][i] = []

        for video_name, det_by_v in detections.items():
            for class_ind, det_by_v_by_c in det_by_v.items():
                # det_by_v_by_c = np.c_[np.ones(det_by_v_by_c.shape[0]) * class_ind, det_by_v_by_c]
                det_by_v_by_c = nms(det_by_v_by_c, 0.4)
                temp_detections[video_name][class_ind] = det_by_v_by_c
                # un_nmsed_det_by_v.append(det_by_v_by_c)

            # nmsed_det_by_v = nms(np.vstack(un_nmsed_det_by_v), 1.0)
            # for class_ind in det_by_v.keys():
            #     detections[video_name][class_ind] = nmsed_det_by_v[nmsed_det_by_v[:, 0] == class_ind]

    eval_results = evaluate_det(temp_detections)
    return eval_results


def evaluate_mAP(detections):
    eval_results = {}
    detections= ds.format_det_for_func_eval_ap(detections)
    iou_range = np.arange(0.1, 1.0, .1)
    ap_values = eval_ap(detections, ds.gt_infos, iou_range)
    mAP = ap_values.mean(axis=0)
    for iou, map_iou in zip(iou_range, mAP):
        eval_results[f'mAP@{iou:.02f}'] = map_iou

    return eval_results


def fuse_det(detctions1, detections2, iou=0.4):
    fused_det = {}
    for video_name, det_by_v in detctions1.items():
        fused_det[video_name] = {}
        for class_ind, det_by_v_by_c in det_by_v.items():
            det_by_v_by_c2 = detections2[video_name][class_ind]
            fuse_det_by_v_by_c = np.vstack([det_by_v_by_c, det_by_v_by_c2])
            fused_det[video_name][class_ind] = nms(fuse_det_by_v_by_c, iou)
    return fused_det

rgb_model_name = 'apn_coralrandom_r3dsony_32x4_10e_thumos14_rgb'
flow_model_name = 'apn_coralrandom_r3dsony_32x4_10e_thumos14_flow'
rgb_work_dir = f'work_dirs/{rgb_model_name}'
flow_work_dir = f'work_dirs/{flow_model_name}'

cfg = Config.fromfile(f"configs/localization/apn/{rgb_model_name}.py")
ds = build_dataset(cfg.data.test)


with open(f"{rgb_work_dir}/detections.pkl", 'rb') as f1:
    rgb_det = pickle.load(f1)
with open(f"{flow_work_dir}/detections.pkl", 'rb') as f2:
    flow_det = pickle.load(f2)
eval_results = evaluate_mAP(rgb_det)
print(eval_results)
eval_results = evaluate_mAP(flow_det)
print(eval_results)
iou = 0.5
fused_det = fuse_det(rgb_det, flow_det, iou)

eval_results = evaluate_mAP(fused_det)
print(eval_results)







# Test proposals
from mmcv import dump
from mmaction.core import average_recall_at_avg_proposals
import numpy as np
import json
import pickle
import time
from mmcv import Config
from mmaction.datasets import build_dataset
from mmaction.localization import nms


model_name = 'apn_prop_coral_r3dsony_32x4_10e_thumos14_flow'
cfg = Config.fromfile(f"configs/localization/apn/{model_name}.py")
ds = build_dataset(cfg.data.test)
work_dir = f'work_dirs/{model_name}'

with open(f'{work_dir}/raw_proposals.pkl', 'rb') as f:
    proposals = pickle.load(f)

num_proposal_list = []
iou = 0.9
for video_name, prop_by_v in proposals.items():
    proposals[video_name] = nms(prop_by_v, iou)
    num_proposal_list.append(len(proposals[video_name]))

num_proposals = sum(num_proposal_list)
# dump_detections = f"{work_dir}/proposals.json"
dump_detections = None
metric_options = {}
eval_results = {}
before = time.time()
temporal_iou_thresholds = metric_options.setdefault(
                    'AR@AN', {}).setdefault('temporal_iou_thresholds',
                                            np.linspace(0.5, 0.95, 10))
max_avg_proposals = metric_options.setdefault(
    'AR@AN', {}).setdefault('max_avg_proposals', 1000)
if isinstance(temporal_iou_thresholds, list):
    temporal_iou_thresholds = np.array(temporal_iou_thresholds)

ground_truth = ds._import_prop_ground_truth()
recall, _, _, auc = (
    average_recall_at_avg_proposals(
        ground_truth,
        proposals,
        num_proposals,
        max_avg_proposals=max_avg_proposals,
        temporal_iou_thresholds=temporal_iou_thresholds))
eval_results['auc'] = auc
eval_results['AR@1'] = np.mean(recall[:, 0])
eval_results['AR@5'] = np.mean(recall[:, 4])
eval_results['AR@10'] = np.mean(recall[:, 9])
eval_results['AR@20'] = np.mean(recall[:, 19])
eval_results['AR@50'] = np.mean(recall[:, 49])
eval_results['AR@100'] = np.mean(recall[:, 99])
end = time.time()
print(f"Takes {end-before} seconds")
print(eval_results)
dump(eval_results, f"{work_dir}/prop_evaluations_{iou:.2}_1000.json")