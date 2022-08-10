import torch
from torch import nn

from mmaction.models.builder import BACKBONES
from custom_modules.models.backbones.slowfast.models import MODEL_REGISTRY
from custom_modules.models.backbones.slowfast.config.defaults import get_cfg
from mmcv.runner import _load_checkpoint, load_state_dict
cfg = get_cfg()


@BACKBONES.register_module()
class MViT2(torch.nn.Module):
    def __init__(self, pretrained=True, flow_input=False, num_frames=32):
        super().__init__()
        if num_frames == 32:
            cfg.merge_from_file("custom_modules/models/backbones/slowfast/config/configs/Kinetics/MVITv2_B_32x3.yaml")
        elif num_frames == 16:
            cfg.merge_from_file("custom_modules/models/backbones/slowfast/config/configs/Kinetics/MVITv2_S_16x4.yaml")
        else:
            raise TypeError("Only 32 or 16 frame input are supported")
        name = cfg.MODEL.MODEL_NAME
        model = MODEL_REGISTRY.get(name)(cfg)
        if pretrained:
            if num_frames == 32:
                state_dict = _load_checkpoint("https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth", map_location=lambda storage, loc: storage)
            elif num_frames == 16:
                state_dict = _load_checkpoint("https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth", map_location=lambda storage, loc: storage)
            else:
                raise TypeError("Only 32 or 16 frame input are supported")
            load_state_dict(model, state_dict['model_state'])
        model.head = nn.Identity()
        self.model = model
        if flow_input:
            w = model.patch_embed.proj.weight
            ww = w.mean(dim=1, keepdim=True)
            ww = torch.cat([ww, ww], dim=1)
            conv_flow = torch.nn.Conv3d(96, 2, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
            conv_flow.weight = torch.nn.Parameter(ww, requires_grad=True)
            conv_flow.bias = model.patch_embed.proj.bias
            self.model.patch_embed.proj = conv_flow

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.model(x)
        return x

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        pass
