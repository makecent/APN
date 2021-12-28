import numpy as np
import json
import pickle
import os
import time
from mmcv import dump, Config
from mmaction.datasets import build_dataset
from dataloader.apn_utils import uniform_1d_sampling, eval_ap
from my_utils.evaluation import evaluate_results


def evaluate_detections(cfg_file="configs/localization/apn/apn_coral_r3dsony_16x1_10e_thumos14_flow.py",
                        detections_file="auto"):
    # Init
    cfg = Config.fromfile(cfg_file)
    ds = build_dataset(cfg.data.test)
    eval_results = {}

    run_name = cfg_file.split('/')[-1].split('.')[0]
    if detections_file == 'auto':
        detections_file = f"work_dirs/{run_name}/detections.pkl"
    with open(detections_file, 'rb') as f:
        detections = pickle.load(f)
    if not isinstance(list(detections.keys())[0], int):
        detections = ds.format_det_for_func_eval_ap(detections)
        dump(detections, detections_file)
    detections = dict(sorted(detections.items()))
    iou_range = np.arange(0.1, 1.0, .1)
    ap_values, loc_acc, cls_acc = eval_ap(detections, ds.gt_infos, iou_range, return_loc_cls=True)
    mAP = ap_values.mean(axis=0)
    cls_acc = cls_acc.mean(axis=0)
    for iou, mAP_iou in zip(iou_range, mAP):
        eval_results = ds.update_and_print_eval(eval_results, mAP_iou, f'mAP@{iou:.01f}')
    print('\n\n')
    for iou, loc_iou in zip(iou_range, loc_acc):
        eval_results = ds.update_and_print_eval(eval_results, loc_iou, f'loc_pre@{iou:.01f}')
    print('\n\n')
    for iou, cls_iou in zip(iou_range, cls_acc):
        eval_results = ds.update_and_print_eval(eval_results, cls_iou, f'cls_acc@{iou:.01f}')


evaluate_detections()


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
# # Load detection results
# import json
# import pickle
# import numpy as np
# from mmcv import Config, dump
# from mmaction.datasets import build_dataset
# from mmaction.localization import eval_ap
# model_name = 'apn_coral+random_r3dsony_32x4_10e_thumos14_flow'
# cfg = Config.fromfile(f"configs/localization/apn/{model_name}.py")
# ds = build_dataset(cfg.data.test)
# work_dir = f'work_dirs/{model_name}'
# # def format_det(det):
# #     for det_by_v in det.copy().values():
# #         for class_ind in list(det_by_v):
# #             det_by_v[int(class_ind)] = np.array(det_by_v.pop(class_ind)).reshape(-1, 3)
# #     return det
# # with open(f"{work_dir}/mean_fuse/detections.json", 'r') as f:
# #     det = format_det(json.load(f))
# with open(f"{work_dir}/mean_fuse/detections.pkl", 'rb') as f:
#     det = pickle.load(f)
#
# det = ds.format_det_for_func_eval_ap(det)
# iou_range = np.arange(0.1, 1.0, .1)
# ap_values = eval_ap(det, ds.gt_infos, iou_range)
#
# eval_results = {}
# for i, iou in enumerate(iou_range):
#     eval_results[f'mAP@{iou:.02f}'] = ap_values.mean(0)[i]
# # dump(eval_results, f"{work_dir}/evaluations.json")


# # Fusion by Mean


def mean_fusion():
    # load results
    main_cfg = 'work_dirs/apn_coral+random_r3dsony_32x4_10e_thumos14_flow/progressions.pkl'
    ass_cfg = 'work_dirs/apn_coral+random_r3dsony_32x4_10e_thumos14_rgb/progressions.pkl'
    save_dir = main_cfg.rsplit('/', 1)[0]

    with open(main_cfg, 'rb') as f:
        results_first = pickle.load(f)
    with open(ass_cfg, 'rb') as f:
        results_second = pickle.load(f)

    def take_mean(list_of_prog):
        t = np.array(list_of_prog)
        mean = t.mean(axis=0)
        return mean.tolist()

    results = take_mean([results_first, results_second])
    dump(results, f'{save_dir}/mean_fuse/rgb+flow_progressions.pkl')
    del results_first, results_second, results

    evaluate_results(cfg_file=main_cfg,
                     results_file=f'{save_dir}/mean_fuse/mean_progressions.pkl',
                     metric_options_override=dict(
                         mAP=dict(
                            dump_detections=f'{save_dir}/mean_fuse/rgb+flow_detections.pkl',
                            dump_evaluation=f'{save_dir}/mean_fuse/rgb+flow_evaluation.json')))
# mean_fusion()


# # ##%  Group Fusion
# import numpy as np
# import pickle
# from mmcv import Config
# from mmaction.datasets import build_dataset
# from mmaction.localization import action_detection, eval_ap, nms, nms_with_cls
#
#
# def format_det(det):
#     for det_by_v in det.copy().values():
#         for class_ind in list(det_by_v):
#             det_by_v[int(class_ind)] = np.array(det_by_v.pop(class_ind)).reshape(-1, 3)
#     return det
#
#
# def evaluate_det(detections):
#     # build return detection without change original detection dict.
#     temp_detections = {}
#     for key in detections.keys():
#         temp_detections[key] = {}
#         for i in range(20):
#             temp_detections[key][i] = []
#
#         for video_name, det_by_v in detections.items():
#             for class_ind, det_by_v_by_c in det_by_v.items():
#                 # det_by_v_by_c = np.c_[np.ones(det_by_v_by_c.shape[0]) * class_ind, det_by_v_by_c]
#                 det_by_v_by_c = nms(det_by_v_by_c, 0.4)
#                 temp_detections[video_name][class_ind] = det_by_v_by_c
#                 # un_nmsed_det_by_v.append(det_by_v_by_c)
#
#             # nmsed_det_by_v = nms(np.vstack(un_nmsed_det_by_v), 1.0)
#             # for class_ind in det_by_v.keys():
#             #     detections[video_name][class_ind] = nmsed_det_by_v[nmsed_det_by_v[:, 0] == class_ind]
#
#     eval_results = evaluate_det(temp_detections)
#     return eval_results
#
#
# def evaluate_mAP(detections):
#     eval_results = {}
#     detections= ds.format_det_for_func_eval_ap(detections)
#     iou_range = np.arange(0.1, 1.0, .1)
#     ap_values = eval_ap(detections, ds.gt_infos, iou_range)
#     mAP = ap_values.mean(axis=0)
#     for iou, map_iou in zip(iou_range, mAP):
#         eval_results[f'mAP@{iou:.02f}'] = map_iou
#
#     return eval_results
#
#
# def fuse_det(detctions1, detections2, iou=0.4):
#     fused_det = {}
#     for video_name, det_by_v in detctions1.items():
#         fused_det[video_name] = {}
#         for class_ind, det_by_v_by_c in det_by_v.items():
#             det_by_v_by_c2 = detections2[video_name][class_ind]
#             fuse_det_by_v_by_c = np.vstack([det_by_v_by_c, det_by_v_by_c2])
#             fused_det[video_name][class_ind] = nms(fuse_det_by_v_by_c, iou)
#     return fused_det
#
# rgb_model_name = 'apn_coral+random_r3dsony_32x4_10e_thumos14_rgb'
# flow_model_name = 'apn_coral+random_r3dsony_32x4_10e_thumos14_flow'
# rgb_work_dir = f'work_dirs/{rgb_model_name}'
# flow_work_dir = f'work_dirs/{flow_model_name}'
#
# cfg = Config.fromfile(f"configs/localization/apn/{rgb_model_name}.py")
# ds = build_dataset(cfg.data.test)
#
#
# with open(f"{rgb_work_dir}/detections.pkl", 'rb') as f1:
#     rgb_det = pickle.load(f1)
# with open(f"{flow_work_dir}/detections.pkl", 'rb') as f2:
#     flow_det = pickle.load(f2)
# eval_results = evaluate_mAP(rgb_det)
# print(eval_results)
# eval_results = evaluate_mAP(flow_det)
# print(eval_results)
# iou = 0.5
# fused_det = fuse_det(rgb_det, flow_det, iou)
#
# eval_results = evaluate_mAP(fused_det)
# print(eval_results)


# # Test proposals
# from mmcv import dump
# from mmaction.core import average_recall_at_avg_proposals
# import numpy as np
# import json
# import pickle
# import time
# from mmcv import Config
# from mmaction.datasets import build_dataset
# from mmaction.localization import nms
#
#
# model_name = 'apn_prop_coral_r3dsony_32x4_10e_thumos14_flow'
# cfg = Config.fromfile(f"configs/localization/apn/{model_name}.py")
# ds = build_dataset(cfg.data.test)
# work_dir = f'work_dirs/{model_name}'
#
# with open(f'{work_dir}/raw_proposals.pkl', 'rb') as f:
#     proposals = pickle.load(f)
#
# num_proposal_list = []
# iou = 0.9
# for video_name, prop_by_v in proposals.items():
#     proposals[video_name] = nms(prop_by_v, iou)
#     num_proposal_list.append(len(proposals[video_name]))
#
# num_proposals = sum(num_proposal_list)
# # dump_detections = f"{work_dir}/proposals.json"
# dump_detections = None
# metric_options = {}
# eval_results = {}
# before = time.time()
# temporal_iou_thresholds = metric_options.setdefault(
#                     'AR@AN', {}).setdefault('temporal_iou_thresholds',
#                                             np.linspace(0.5, 0.95, 10))
# max_avg_proposals = metric_options.setdefault(
#     'AR@AN', {}).setdefault('max_avg_proposals', 1000)
# if isinstance(temporal_iou_thresholds, list):
#     temporal_iou_thresholds = np.array(temporal_iou_thresholds)
#
# ground_truth = ds._import_prop_ground_truth()
# recall, _, _, auc = (
#     average_recall_at_avg_proposals(
#         ground_truth,
#         proposals,
#         num_proposals,
#         max_avg_proposals=max_avg_proposals,
#         temporal_iou_thresholds=temporal_iou_thresholds))
# eval_results['auc'] = auc
# eval_results['AR@1'] = np.mean(recall[:, 0])
# eval_results['AR@5'] = np.mean(recall[:, 4])
# eval_results['AR@10'] = np.mean(recall[:, 9])
# eval_results['AR@20'] = np.mean(recall[:, 19])
# eval_results['AR@50'] = np.mean(recall[:, 49])
# eval_results['AR@100'] = np.mean(recall[:, 99])
# end = time.time()
# print(f"Takes {end-before} seconds")
# print(eval_results)
# dump(eval_results, f"{work_dir}/prop_evaluations_{iou:.2}_1000.json")
