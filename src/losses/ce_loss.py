import torch
from .base_loss import BaseLoss
import torch.nn.functional as F


class CrossEntropyLoss(BaseLoss):
    def __init__(self, **kwargs):
        """
        :param beta: The threshold in the piecewise function.
            Defaults to 1.0.
        :param kwargs:
        """
        super(CrossEntropyLoss, self).__init__(**kwargs)
        self.activation = 'softmax'

    def loss(self, pred, target, **kwargs):
        loss = F.cross_entropy(pred, target.long())
        return loss