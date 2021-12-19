import numpy as np
import os
from mmaction.datasets.builder import PIPELINES


@PIPELINES.register_module()
class FetchStackedFrames(object):

    def __init__(self,
                 clip_len,
                 num_clips=1,
                 frame_interval=1,
                 out_of_bound_opt='repeat_last',
                 test_mode=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.num_clips = num_clips
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

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
        elif self.out_of_bound_opt == 'repeat_last':
            frame_inds = np.clip(frame_inds, start_index, start_index + total_frames - 1)
        else:
            raise ValueError('Illegal out_of_bound option.')
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['num_clips'] = self.num_clips
        results['frame_interval'] = self.frame_interval
        return results


@PIPELINES.register_module()
class FetchStackedFeatures(object):

    def __init__(self,
                 feat_path,
                 clip_len,
                 num_clips=1,
                 frame_interval=1,
                 out_of_bound_opt='repeat_last',
                 test_mode=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.num_clips = num_clips
        self.feat_path = feat_path
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def __call__(self, results):
        """Perform the FetchFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        clip_len = self.clip_len * self.num_clips
        video_feature = np.load(os.path.join(self.feat_path, f"{results['video_name']}.npy"))
        # video_feature = pd.read_csv(os.path.join(self.feat_path, f"{results['video_name']}.csv")).values.astype("float32")
        len_feature = len(video_feature)

        feat_inds = round(len_feature * results['frame_index%'])

        feat_inds = feat_inds + np.arange(-clip_len/2, clip_len/2, dtype=int) * self.frame_interval

        if self.out_of_bound_opt == 'loop':
            feat_inds = np.mod(feat_inds, len_feature)
        elif self.out_of_bound_opt == 'repeat_last':
            feat_inds = np.clip(feat_inds, 0, len_feature - 1)
        else:
            raise ValueError('Illegal out_of_bound option.')

        results['snippet'] = video_feature[feat_inds].T
        results['clip_len'] = self.clip_len
        results['num_clips'] = self.num_clips
        results['frame_interval'] = self.frame_interval
        return results



@PIPELINES.register_module()
class LabelToOrdinal(object):

    def __init__(self,
                 num_stages=100,
                 test_mode=False):
        self.num_stages = num_stages
        self.test_mode = test_mode

    def __call__(self, results):
        """Convert progression_label to Ordinal matrix.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        denormalized_prog = round(results['progression_label'] * self.num_stages)
        ordinal_label = np.full(self.num_stages, fill_value=0.0, dtype='float32')
        ordinal_label[:denormalized_prog] = 1.0
        results['progression_label'] = ordinal_label
        return results


@PIPELINES.register_module()
class LabelToInt(object):

    def __init__(self,
                 num_stages=100,
                 test_mode=False):
        self.num_stages = num_stages
        self.test_mode = test_mode

    def __call__(self, results):
        """Convert progression_label to one-hot matrix for classification.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        denormalized_prog = round(results['progression_label'] * self.num_stages)
        results['progression_label'] = denormalized_prog
        return results
