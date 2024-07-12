from .focal_loss import *
from .l1_loss import *
from .edl import *
from .ce_loss import *


def build_loss(type, **kwargs):
    return globals()[type](**kwargs)