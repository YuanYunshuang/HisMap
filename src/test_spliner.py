import os.path

import cv2
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from src.dlmodule.renderer import FCN, ShapeRenderer, SplineRenderer
from src.dlmodule.resnet import ResNet
from src.dlmodule.spline_decoder import SplineDecoder
from src.utils import paint, splines
from src.utils.train import freeze
import torch.optim as optim
import torchvision

writer = SummaryWriter("../train_log/src")

use_cuda = torch.cuda.is_available()
step = 0
n_params = 25
ckpt_path = "../data/spliner.pkl"

renderer = freeze(SplineRenderer(n_params, state_dict_path="../data/spline_renderer.pkl")).cuda()
net = SplineDecoder(1, 18, n_params)
optimizer = optim.Adam(net.parameters(), lr=3e-6)
batch_size = 256

X = cv2.imread("/home/yuan/Downloads/3922_2017_forest_bdry.tif")[..., 0] / 255
H, W = X.shape
H = W = 1024
sample_width = 128
samples = np.stack(np.where(X), axis=-1)
valid = (samples[:, 0] < H - sample_width) & (samples[:, 1] < W - sample_width)
samples = samples[valid]


def load_weights():
    if os.path.exists(ckpt_path):
        pretrained_dict = torch.load(ckpt_path)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)


load_weights()
net.cuda()
net.eval()
Bsqrt = int(np.sqrt(batch_size))

# X = torch.from_numpy(X[:1024, :1024]).float()[None, None].cuda()
# with torch.no_grad():
#     out = net(X)
#     canvas = renderer(out.permute(0, 2, 3, 1))
#     cv2.imshow("canvas", canvas.detach().cpu().numpy())
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

inds_h = np.arange(0, H, 120)
inds_w = np.arange(0, W, 120)
inds_h[-1] = H - sample_width
inds_w[-1] = W - sample_width
Y = np.zeros_like(X[:H,:W])
Z = np.zeros_like(X[:H,:W])

samples = []
inds = []
for h in inds_h:
    for w in inds_w:
        samples.append(X[h:h+sample_width, w:w+sample_width])
        inds.append([h, w])

for i in range(0, len(samples) // batch_size + 1):
    cur_batch = samples[i * batch_size : min((i + 1) * batch_size, len(samples))]
    cur_inds = inds[i * batch_size : min((i + 1) * batch_size, len(samples))]
    cur_batch = torch.from_numpy(np.stack(cur_batch, axis=0)).unsqueeze(1).float().cuda()

    gen = net(cur_batch)
    canvas = renderer(gen.permute(0, 2, 3, 1)).detach().cpu().numpy()
    vec_map = []
    for param in gen.detach().cpu().numpy():
        mask = param[0] > 0.25
        mask_param = param[1:].transpose(1, 2, 0).reshape(16, 16, 8, 3)
        mask_param[..., -1] = 1 / 4
        vec_map.append(splines.draw_splines_on_canvas(mask, mask_param, 8))

    for (h, w), canv, vm in zip(cur_inds, canvas, vec_map):
        Y[h+4:h + sample_width, w+4:w + sample_width] = canv[4:, 4:]
        Z[h+4:h + sample_width, w+4:w + sample_width] = vm[4:, 4:]

cv2.imwrite("/home/yuan/Downloads/3922_2017_forest_bdry_pred_renderer.jpg", np.uint8(Y * 255))
cv2.imwrite("/home/yuan/Downloads/3922_2017_forest_bdry_pred_drawer.jpg", np.uint8(Z * 255))


for h in range(H // sample_width // Bsqrt):
    h1 = h * sample_width * Bsqrt
    h2 = min((h + 1) * sample_width * Bsqrt, H)
    for w in range(W // sample_width // Bsqrt):
        w1 = w * sample_width * Bsqrt
        w2 = min((w + 1) * sample_width * Bsqrt, W)

        patch = torch.from_numpy(X[h1:h2, w1:w2]).float().reshape(Bsqrt, sample_width, Bsqrt, sample_width)
        patch = patch.permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(1).cuda()

        gen = net(patch)
        canvas = renderer(gen.permute(0, 2, 3, 1))
        canvas = canvas.reshape(Bsqrt, Bsqrt, sample_width, sample_width).permute(0, 2, 1, 3)
        canvas = canvas.reshape(Bsqrt * sample_width, Bsqrt * sample_width)
        cv2.imshow("canvas", canvas.detach().cpu().numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()


