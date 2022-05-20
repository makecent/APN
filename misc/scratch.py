import heapq
import os.path as osp
from collections import namedtuple

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmaction.datasets.builder import build_dataset
from mmcv import Config
from tqdm import tqdm

from custom_modules.mmdet_utils import bbox2result, eval_map
from custom_modules.mmdet_utils import multiclass_nms

cfg_file = Config.fromfile("configs/apn_r3dsony_32x4_10e_thumos14_flow.py")
det_before_nms = mmcv.load(osp.join(cfg_file.work_dir, 'det_before_nms.pkl'))
ds = build_dataset(cfg_file.data.test)
annotations = ds.get_ann_info()
classes = ds.CLASSES


def nms(det_bbox, cls_score, loc_score, score_thr, nms_thr, max_per_video):
    det_bbox, cls_score, loc_score = map(torch.from_numpy, (det_bbox, cls_score, loc_score))
    det_bbox, det_label = multiclass_nms(
        det_bbox,
        cls_score,
        score_thr,
        dict(iou_thr=nms_thr),
        max_per_video,
        score_factors=loc_score)
    return det_bbox, det_label


class Param:
    def __init__(self, name, grid):
        self.name = name
        self.grid = grid
        self.num = len(grid)
        self.grid_name = [f"{x:.2f}" if isinstance(x, float) else f"{x}" for x in grid]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.grid[idx]


class SearchSpace:

    def __init__(self, params):
        self.params = params
        self.space = [len(p) for p in params]
        self.total_num = len(self)
        self.num_params = len(params)
        self.param_names = [p.name for p in params]
        self.P = namedtuple('Params', [p.name for p in params])
        self.result = []

    def __len__(self):
        return int(np.prod(self.space))

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        inds = np.unravel_index(idx, self.space)
        return self.P(*[p[inds[i]] for i, p in enumerate(self.params)])

    def boxplot(self, param_specify, **kwargs):
        if isinstance(param_specify, int):
            param_ind = param_specify
            param_name = self.param_names[param_ind]
        elif isinstance(param_specify, str):
            param_ind = self.param_names.index(param_specify)
            param_name = param_specify
        else:
            raise TypeError
        result = np.array(self.result).reshape(self.space)
        data = [d.flatten() for d in np.split(result, self.space[param_ind], axis=param_ind)]
        plt.boxplot(data, **kwargs)
        plt.xticks(list(range(1, len(data) + 1)), self.params[param_ind].grid_name)
        plt.xlabel('Choice')
        plt.ylabel('Score')
        plt.title(param_name)
        plt.grid()
        plt.show()

    def topkplot(self, k=10):
        assert k <= self.total_num
        x = list(range(self.num_params))
        yy = []
        topk = heapq.nlargest(k, enumerate(self.result), key=lambda x: x[1])
        for n, (idx, value) in enumerate(topk):
            y = np.unravel_index(idx, self.space)
            yy.append(y)
            plt.plot(x, y, label=f'top{n + 1} {value:.2f}')
        plt.legend()
        plt.xticks(list(range(self.num_params + 2)), self.param_names + ['', ''])
        plt.yticks(list(range(np.max(yy) + 2)))
        plt.title(f"Top {k} settings")
        plt.ylabel('Choice')
        plt.grid()
        plt.show()


param1 = Param('nms_thr', np.arange(0.25, 0.46, 0.05))
param2 = Param('score_thr', np.arange(0, 0.5, 0.05))
param3 = Param('max_per_video', np.hstack([0, np.arange(150, 300, 30)]))
search_space = SearchSpace([param1, param2, param3])
for params in tqdm(search_space):
    det_results = []
    for results_by_video in det_before_nms:
        det_bbox, cls_score, loc_score = results_by_video
        det_bbox, det_label = nms(det_bbox, cls_score, loc_score,
                                  params.score_thr,
                                  params.nms_thr,
                                  params.max_per_video)
        det_results.append((det_bbox, det_label))
    det_results = [bbox2result(det_bboxes, det_labels, len(classes)) for det_bboxes, det_labels in det_results]
    iou_thr = 0.5
    mean_ap, _ = eval_map(det_results, annotations, iou_thr=iou_thr, dataset=classes)
    mean_ap = round(mean_ap, 3)
    # mean_ap = np.random.rand()
    search_space.result.append(mean_ap)

search_space.boxplot(0)
search_space.topkplot(5)
print('finished')
