import numpy as np
import torch


def update_repr_for_np_and_torch():
    torch.Tensor.__repr__ = lambda self: f"{self.shape}_{torch.Tensor.__repr__(self)}"
    np.ndarray.__repr__ = lambda self: f"{self.shape}_{np.array2string(self)}"