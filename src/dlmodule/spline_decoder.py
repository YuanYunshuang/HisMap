import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm as WN

from src.dlmodule.util import get_activation


def get_block_settings(depth):
    try:
        return {
        18: (BasicBlock, [2,2,2,2]),
        34: (BasicBlock, [3,4,6,3]),
        50: (Bottleneck, [3,4,6,3]),
        101:(Bottleneck, [3,4,23,3]),
        152:(Bottleneck, [3,8,36,3]),
    }[depth]
    except:
        raise ValueError(f"Resnet depth should be either 18, 34, 50, 101, 152.")


def conv3x3(in_planes, out_planes, stride=1, norm='BN', activation='relu'):
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)]
    if norm == 'BN':
        layers.append(nn.BatchNorm2d(out_planes))
    elif norm == 'WN':
        layers[0] = WN(layers[0])
    if activation is not None:
        layers.append(get_activation(activation))
    return nn.Sequential(*layers)


def conv1x1(in_planes, out_planes, stride=1, norm='BN', activation='relu'):
    layers = [nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)]
    if norm == 'BN':
        layers.append(nn.BatchNorm2d(out_planes))
    if activation == 'relu':
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='BN', activation='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, norm, activation)
        self.conv2 = conv3x3(planes, planes, 1, norm, None)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv1x1(in_planes, self.expansion*planes, stride, norm, None)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='BN', activation='relu'):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes, 1, norm, activation)
        self.conv2 = conv3x3(planes, planes, stride, norm, activation)
        self.conv3 = conv1x1(planes, self.expansion*planes, 1, norm, None)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = conv1x1(in_planes, self.expansion*planes, stride, norm, None)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.activation(out)

        return out


class SplineDecoder(nn.Module):
    def __init__(self, num_inputs, depth, num_outputs, norm='BN', activation='relu'):
        super(SplineDecoder, self).__init__()
        self.in_planes = 64
        self.norm = norm
        self.activation = activation

        block, num_blocks = get_block_settings(depth)

        self.conv1 = conv3x3(num_inputs, 64, 2, norm, activation)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.upsample = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.PixelShuffle(2),
            nn.ReLU()
        )
        self.out_layer = nn.Conv2d(64, num_outputs, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm, self.activation))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upsample(x)
        x = self.out_layer(x)
        x = torch.sigmoid(x)
        return x
