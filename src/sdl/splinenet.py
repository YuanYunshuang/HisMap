import torch
from torch import nn

from src.dlmodule.spline_decoder import SplineDecoder
from src.dlmodule.renderer import SplineRenderer
from src.utils.splines import paint_splines
from src.losses import build_loss


class SplineNet(nn.Module):
    def __init__(self, n_params=25, **kwargs):
        super().__init__()
        self.n_params = n_params
        self.spliner = SplineDecoder(1, 18, self.n_params)
        self.renderer = SplineRenderer(self.n_params)
        self.painter = paint_splines
        self.loss_fn = build_loss('SmoothL1Loss')

    def forward(self, data, **kwargs):
        x = data['img']
        splines = self.spliner(x)
        rendered_img = self.renderer(splines)
        painted_img = self.painter(splines)

        self.out_dict = {
            'input_img': x.squeeze(),
            'splines': splines,
            'rendered_img': rendered_img,
            'painted_img': painted_img
        }

    def loss(self):
        render_loss = self.loss_fn(self.out_dict['rendered_img'], self.out_dict['painted_img'])
        spliner_loss = self.loss_fn(self.out_dict['rendered_img'], self.out_dict['input_img'])
        total_loss = render_loss + spliner_loss
        loss_dict = {
            'render_loss': render_loss,
            'spliner_loss': spliner_loss,
            'total_loss': total_loss
        }
        return total_loss, loss_dict