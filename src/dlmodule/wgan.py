import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from src.dlmodule.util import get_activation


def weighted_conv2d(in_channels, out_channels, kernel=5, stride=2, padding=2, activation='trelu'):
    layers = [weight_norm(nn.Conv2d(in_channels, out_channels, kernel, stride, padding))]
    if activation is not None:
        layers.append(get_activation(activation))
    return nn.Sequential(*layers)


class WGAN(nn.Module):
    def __init__(self):
        super(WGAN, self).__init__()

        self.conv0 = weighted_conv2d(6, 16)
        self.conv1 = weighted_conv2d(16, 32)
        self.conv2 = weighted_conv2d(32, 64)
        self.conv3 = weighted_conv2d(64, 128)
        self.conv4 = weighted_conv2d(128, 1, activation=None)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 1)
        return x