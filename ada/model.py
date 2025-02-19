
from torch import nn
from src.dlmodule.unet import UNet

Models = {
    'UNet': UNet,
}

class ADAseg(nn.Module):
    def __init__(self, network, **kwargs):
        super().__init__()
        self.network = Models[network](**kwargs)

    def forward(self, x):
        out = self.network(x)
        return out