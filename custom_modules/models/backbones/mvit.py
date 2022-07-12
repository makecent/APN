import torch

from mmaction.models.builder import BACKBONES


@BACKBONES.register_module()
class MViTB(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = torch.hub.load("facebookresearch/pytorchvideo", model="mvit_base_16x4", pretrained=pretrained)
        model.head = None
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x[:, 0, :]

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        pass
