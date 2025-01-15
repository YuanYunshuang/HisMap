import torch
from torch import nn

from src.dlmodule import vit
from src.losses import build_loss, pred_to_conf_unc


class VitClassifier(nn.Module):
    def __init__(self, vit_type, loss_cfg, **kwargs):
        super().__init__()
        use_fused_attn = True if kwargs.get('mode', 'train') == 'train' else False
        self.vit = getattr(vit, f"crossvit_{vit_type}")(fused_attn=use_fused_attn)
        self.loss_fn = build_loss(**loss_cfg)

    def forward(self, data, **kwargs):
        x = data['img']
        out = self.vit(x)
        data['pred'] = out
        data['conf'], data['unc'] = pred_to_conf_unc(
            data['pred'], self.loss_fn.activation, False)
        self.out_dict = data
        return data

    def loss(self, data, epoch, **kwargs):
        pred = data['pred']
        gt = data['label'].reshape(-1, 1)
        total_loss = self.loss_fn(pred, gt, temp=epoch)
        loss_dict = {
            'total_loss': total_loss
        }
        return total_loss, loss_dict