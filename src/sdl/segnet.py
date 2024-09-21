import torch
from torch import nn

from src.dlmodule.unet import UNet
from src.losses import build_loss, pred_to_conf_unc


class SegNet(nn.Module):
    def __init__(self, loss_cfg, **kwargs):
        super().__init__()
        self.unet = UNet(**kwargs)
        self.loss_neg_pos_ratio = loss_cfg.pop('neg_pos_ratio', None)
        self.loss_fn = build_loss(**loss_cfg)

    def forward(self, data, **kwargs):
        x = data['input_img']
        out = self.unet(x)
        data['pred_img'] = out
        is_edl = 'edl' in self.loss_fn.name.lower()
        data['conf_img'], data['unc_img'] = pred_to_conf_unc(
            data['pred_img'].permute(0, 2, 3, 1), self.loss_fn.activation, is_edl)
        self.out_dict = data
        return data

    def loss(self, data, epoch, **kwargs):
        pred = data['pred_img'].flatten(2, 3).permute(0, 2, 1).flatten(0, 1)
        gt = data['gt_img']
        if gt.ndim == 3:
            gt = gt.flatten()
            cared = self.sample_target(gt)
        else:
            gt = gt.permute(0, 2, 3, 1).flatten(0, 2)
            cared = torch.ones_like(gt[:, 0]).bool()
        total_loss = self.loss_fn(pred[cared], gt[cared], temp=epoch)
        loss_dict = {
            'total_loss': total_loss
        }
        return total_loss, loss_dict

    def sample_target(self, gt):
        cared = torch.ones_like(gt).bool()
        if self.loss_neg_pos_ratio != None:
            pos = gt > 0
            neg = gt == 0
            n_pos = pos.sum()
            n_neg = neg.sum()
            if n_neg > n_pos:
                neg_inds = torch.where(neg)[0]
                dont_care = neg_inds[torch.randperm(len(neg_inds))[:n_neg - n_pos]]
            else:
                pos_inds = torch.where(pos)[0]
                dont_care = pos_inds[torch.randperm(len(pos_inds))[:n_pos - n_neg]]
            cared[dont_care] = False
            if cared.sum() == 0:
                cared = torch.rand_like(gt) < 0.1
        return cared
