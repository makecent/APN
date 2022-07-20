import numpy as np
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.video_dataset import VideoDataset
from mmcv.utils import print_log
import copy


@DATASETS.register_module()
class VideoDataset_MAE(VideoDataset):
    def evaluate(self, results, **kwargs):
        if 'MAE' in kwargs.get('metrics', []):
            cls_score, reg_mae = list(zip(*results))
            _kwargs = copy.deepcopy(kwargs)
            _kwargs['metrics'].remove('MAE')
            eval_results = super().evaluate(list(cls_score), **_kwargs)

            msg = f'Evaluating MAE ...'
            logger = kwargs.get('logger', None)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            MAE = np.array(reg_mae).mean()
            eval_results[f'MAE'] = MAE
            print_log(f'\nMAE\t{MAE:.2f}', logger=logger)
        else:
            eval_results = super().evaluate(results, **kwargs)

        return eval_results
