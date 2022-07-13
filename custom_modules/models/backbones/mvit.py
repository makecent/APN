import torch

from mmaction.models.builder import BACKBONES


@BACKBONES.register_module()
class MViTB(torch.nn.Module):
    def __init__(self, pretrained=True, flow_input=False):
        super().__init__()
        model = torch.hub.load("facebookresearch/pytorchvideo", model="mvit_base_16x4", pretrained=pretrained)
        model.head = None
        self.model = model
        if flow_input:
            w = model.patch_embed.patch_model.weight
            ww = w.mean(dim=1, keepdim=True)
            ww = torch.cat([ww, ww], dim=1)
            conv_flow = torch.nn.Conv3d(96, 2, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
            conv_flow.weight = torch.nn.Parameter(ww, requires_grad=True)
            conv_flow.bias = model.patch_embed.patch_model.bias
            self.model.patch_embed.patch_model = conv_flow


    def forward(self, x):
        x = self.model(x)
        return x[:, 0, :]

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        pass
