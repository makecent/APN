
from custom_modules.datasets.random_erasing import RandomErasing
import torch

pp = RandomErasing()
results = dict()
results['imgs'] = torch.ones(3, 8, 224, 224)
t = pp(results)
print('haha')
