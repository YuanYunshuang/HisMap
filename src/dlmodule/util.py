import torch
from torch import nn
import torch.nn.functional as F


def get_activation(name):
    try:
        return {
            'relu': nn.ReLU(),
            'trelu': TReLU()
        }[name]
    except:
        return ValueError("Supported activation: relu, trelu")


class TReLU(nn.Module):
    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x