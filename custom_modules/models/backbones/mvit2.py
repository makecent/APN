import torch
from torch import nn

from mmaction.models.builder import BACKBONES
from custom_modules.models.backbones.slowfast.models import MODEL_REGISTRY
from custom_modules.models.backbones.slowfast.config.defaults import get_cfg
from mmcv.runner import _load_checkpoint, load_state_dict, load_checkpoint, _load_checkpoint_with_prefix
cfg = get_cfg()


@BACKBONES.register_module()
class MViT2(torch.nn.Module):
    def __init__(self, pretrained=True, pretrain_prefix=None, flow_input=False, num_frames=32):
        super().__init__()
        if num_frames == 32:
            cfg.merge_from_file("custom_modules/models/backbones/slowfast/config/configs/Kinetics/MVITv2_B_32x3.yaml")
        elif num_frames == 16:
            cfg.merge_from_file("custom_modules/models/backbones/slowfast/config/configs/Kinetics/MVITv2_S_16x4.yaml")
        else:
            raise TypeError("Only 32 or 16 frame input are supported")
        name = cfg.MODEL.MODEL_NAME
        model = MODEL_REGISTRY.get(name)(cfg)

        if flow_input:
            model.patch_embed.proj = torch.nn.Conv3d(96, 2, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))

        if pretrained:
            if not isinstance(pretrained, str):
                if num_frames == 32:
                    state_dict = _load_checkpoint("https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth", map_location=lambda storage, loc: storage)
                elif num_frames == 16:
                    state_dict = _load_checkpoint("https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth", map_location=lambda storage, loc: storage)
                else:
                    raise TypeError("Only 32 or 16 frame input are supported")
                state_dict = state_dict['model_state']
                if flow_input:
                    rgb_proj = state_dict['patch_embed.proj.weight']
                    flow_proj = rgb_proj.mean(dim=1, keepdim=True)
                    state_dict['patch_embed.proj.weight'] = torch.cat([flow_proj, flow_proj], dim=1)
                load_state_dict(model, state_dict)
            else:
                if pretrain_prefix:
                    state_dict = _load_checkpoint_with_prefix(pretrain_prefix, pretrained)
                    load_state_dict(model, state_dict)
                else:
                    load_checkpoint(model, pretrained)
        model.head = nn.Identity()
        self.model = model

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.model(x)
        return x

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        pass
