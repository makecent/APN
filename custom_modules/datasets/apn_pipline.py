import numpy as np
from mmaction.datasets.builder import PIPELINES
from mmaction.datasets.pipelines import SampleFrames
from mmaction.datasets.pipelines import ThreeCrop as _ThreeCrop
from mmaction.datasets.pipelines.augmentations import _init_lazy_if_proper
from mmaction.datasets import BLENDINGS, MixupBlending, CutmixBlending
from mmcv.utils import build_from_cfg
import torch
import warnings
from torch.nn import functional as F

@PIPELINES.register_module()
class FetchStackedFrames(object):

    def __init__(self,
                 clip_len,
                 num_clips=1,
                 frame_interval=1,
                 out_of_bound_opt='clamp',
                 test_mode=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.num_clips = num_clips
        assert self.out_of_bound_opt in ['loop', 'clamp']

    def __call__(self, results):
        """Perform the FetchFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        clip_len = self.clip_len * self.num_clips
        start_index = results['start_index']
        total_frames = results['total_frames']
        frame_inds = results['frame_index'] + np.arange(-clip_len/2, clip_len/2, dtype=int) * self.frame_interval

        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'clamp':
            frame_inds = np.clip(frame_inds, start_index, start_index + total_frames - 1)

        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['num_clips'] = self.num_clips
        results['frame_interval'] = self.frame_interval
        return results


@PIPELINES.register_module()
class FetchGlobalFrames:
    def __init__(self, sample_range=640, *args, **kwargs):
        self.sample_range = sample_range
        self.tsn_sampler = SampleFrames(*args, **kwargs)

    def __call__(self, results):
        """Perform the FetchFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        start_index = results['start_index']
        total_frames = results['total_frames']
        frame_inds = self.tsn_sampler(dict(total_frames=self.sample_range, start_index=start_index))['frame_inds']
        frame_inds += results['frame_index'] - self.sample_range//2
        frame_inds = np.clip(frame_inds, start_index, start_index + total_frames - 1)

        results['frame_inds'] = np.concatenate([results['frame_inds'], frame_inds])
        results['clip_len'] = 1
        results['num_clips'] = 1
        return results

@PIPELINES.register_module()
class LabelToOrdinal(object):

    def __init__(self, num_stages=100):
        self.num_stages = num_stages

    @staticmethod
    def _get_prog(results):
        clip_center = results['frame_inds'].reshape([results['num_clips'], results['clip_len']]).mean(axis=-1)
        prog = np.clip(clip_center / results['total_frames'], a_min=0, a_max=1)
        return prog

    def __call__(self, results):
        """Convert progression_label to ordinal label. e.g., 0.031 => [1, 1, 1, 0, ...]."""
        ordinal_label = np.full(self.num_stages, fill_value=0.0, dtype='float32')
        prog = results['progression_label'] if 'progression_label' in results else self._get_prog(results)

        denormalized_prog = round(prog * self.num_stages)
        ordinal_label[:denormalized_prog] = 1.0
        results['progression_label'] = ordinal_label
        return results


@BLENDINGS.register_module(force=True)
class MixupBlendingProg(MixupBlending):

    def __call__(self, imgs, class_label, progression_label):
        one_hot_label = F.one_hot(class_label, num_classes=self.num_classes)

        mixed_imgs, mixed_class_label, mixed_prog_label = self.do_blending(imgs, one_hot_label, progression_label)

        return mixed_imgs, mixed_class_label, mixed_prog_label

    def do_blending(self, imgs, class_label, progression_label):

        lam = self.beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_class_label = lam * class_label + (1 - lam) * class_label[rand_index, :]
        mixed_prog_label = lam * progression_label + (1 - lam) * progression_label[rand_index, :]

        return mixed_imgs, mixed_class_label, mixed_prog_label


@BLENDINGS.register_module(force=True)
class CutmixBlendingProg(CutmixBlending):

    def __call__(self, imgs, class_label, progression_label):
        one_hot_label = F.one_hot(class_label, num_classes=self.num_classes)

        mixed_imgs, mixed_class_label, mixed_prog_label = self.do_blending(imgs, one_hot_label, progression_label)

        return mixed_imgs, mixed_class_label, mixed_prog_label

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - torch.div(cut_w, 2, rounding_mode='floor'), 0, w)
        bby1 = torch.clamp(cy - torch.div(cut_h, 2, rounding_mode='floor'), 0, h)
        bbx2 = torch.clamp(cx + torch.div(cut_w, 2, rounding_mode='floor'), 0, w)
        bby2 = torch.clamp(cy + torch.div(cut_h, 2, rounding_mode='floor'), 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, class_label, progression_label):

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                                  bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        mixed_class_label = lam * class_label + (1 - lam) * class_label[rand_index, :]
        mixed_prog_label = lam * progression_label + (1 - lam) * progression_label[rand_index, :]

        return imgs, mixed_class_label, mixed_prog_label


@BLENDINGS.register_module()
class BatchAugBlendingProg:
    """Implementing
        https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf
        Only support repeated blending.
    """

    def __init__(self,
                 blendings=(dict(type='MixupBlendingProg', num_classes=200, alpha=.8),
                            dict(type='CutmixBlendingProg', num_classes=200, alpha=1.))):
        self.blendings = [build_from_cfg(bld, BLENDINGS) for bld in blendings]

    def __call__(self, imgs, class_label, progression_label):
        repeated_imgs = []
        repeated_cls_label = []
        repeated_prog_label = []

        for bld in self.blendings:
            mixed_imgs, mixed_class_label, mixed_prog_label = bld(imgs, class_label, progression_label)
            repeated_imgs.append(mixed_imgs)
            repeated_cls_label.append(mixed_class_label)
            repeated_prog_label.append(mixed_prog_label)
        return torch.cat(repeated_imgs), torch.cat(repeated_cls_label), torch.cat(repeated_prog_label)

@PIPELINES.register_module(force=True)
class ThreeCrop(_ThreeCrop):
    """
    Support short-side center crop now;
    Repeat progression label three times;
    """
    def __call__(self, results):
        _init_lazy_if_proper(results, False)
        if 'gt_bboxes' in results or 'proposals' in results:
            warnings.warn('ThreeCrop cannot process bounding boxes')

        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size
        assert crop_h <= img_h and crop_w <= img_w

        if img_w >= img_h:
            w_step = (img_w - crop_w) // 2
            h_offset = (img_h - crop_h) // 2
            offsets = [
                (0, h_offset),  # left
                (2 * w_step, h_offset),  # right
                (w_step, h_offset),  # middle
            ]
        else:
            h_step = (img_h - crop_h) // 2
            w_offset = (img_w - crop_w) // 2
            offsets = [
                (w_offset, 0),  # top
                (w_offset, 2 * h_step),  # down
                (w_offset, h_step),  # middle
            ]

        cropped = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = cropped
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]
        if 'prog_label' in results:
            results['prog_label'] = np.repeat(results['prog_label'], 3)

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str