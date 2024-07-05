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
batch_size = 64

X = cv2.imread("/home/yuan/Downloads/3922_2017_forest_bdry.tif")[..., 0] / 255
H, W = X.shape
sample_width = 128
samples = np.stack(np.where(X), axis=-1)
valid = (samples[:, 0] < H - sample_width) & (samples[:, 1] < W - sample_width)
samples = samples[valid]


def save_model():
    if use_cuda:
        net.cpu()
    torch.save(net.state_dict(), ckpt_path)
    if use_cuda:
        net.cuda()


def load_weights():
    if os.path.exists(ckpt_path):
        pretrained_dict = torch.load(ckpt_path)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)


def criterion(y_pred, y_true):
    return (y_true - y_pred).abs().sum() / y_true.shape[0] / y_true.shape[2]


load_weights()
while step < 500000:
    net.train()
    train_batch = []
    inds = samples[np.random.choice(samples.shape[0], batch_size, replace=False)]
    for ind in inds:
        train_batch.append(X[ind[0]:ind[0] + sample_width, ind[1]:ind[1] + sample_width])
    train_batch = torch.from_numpy(np.stack(train_batch, axis=0)).float().unsqueeze(1)
    if use_cuda:
        net = net.cuda()
        train_batch = train_batch.cuda()
    gen = net(train_batch)
    canvas = renderer(gen.permute(0, 2, 3, 1)).unsqueeze(1)
    optimizer.zero_grad()
    loss = criterion(canvas, train_batch)
    loss.backward()
    optimizer.step()

    if step < 200000:
        lr = 1e-4
    elif step < 400000:
        lr = 1e-5
    else:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    writer.add_scalar("train/loss", loss.item(), step)
    if step % 100 == 0:
        print(step, loss.item())
        net.eval()
        gen = net(train_batch)
        canvas = renderer(gen.permute(0, 2, 3, 1)).unsqueeze(1)
        loss = criterion(canvas, train_batch)
        writer.add_scalar("val/loss", loss.item(), step)
        canvas_grid = torchvision.utils.make_grid(canvas[:32])
        gt_grid = torchvision.utils.make_grid(train_batch[:32])
        writer.add_image("train/gen", canvas_grid, step, dataformats='CHW')
        writer.add_image("train/gt", gt_grid, step, dataformats='CHW')
    if step % 1000 == 0:
        save_model()
    step += 1
