import torch
from torch import nn
from typing import List, Dict, Optional
import importlib


def build_module(module_cfg):
    module_full_path=module_cfg['type']
    package, module_name = module_full_path.rsplit('.', 1)
    module = importlib.import_module(f'src.dlmodule.{package}')
    cls_obj = getattr(module, module_name, None)
    assert cls_obj is not None, f'Class \'{module_name}\' not found.'
    try:
        inst = cls_obj(**module_cfg)
    except Exception as e:
        raise Exception(f"{module_name}:{e.__repr__()}")
    return inst


class BaseModule(nn.Module):
    def __init__(self, freeze=False, **kwargs):
        super(BaseModule, self).__init__()
        self.freeze = freeze

    def to_gpu(self, gpu_id):
        self.to(gpu_id)
        addtional_sync_func = nn.SyncBatchNorm.convert_sync_batchnorm
        return None

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        """This must be implemented in head module."""
        # TODO: Create Head base module.
        pass

