import os.path

import cv2
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from src.dlmodule.renderer import FCN, ShapeRenderer
from src.dlmodule.resnet import ResNet
from src.utils import paint
from src.utils.train import freeze
import torch.optim as optim
import torchvision

writer = SummaryWriter("../train_log/src")

use_cuda = torch.cuda.is_available()
step = 0
n_params = 6
n_triangle = 5
ckpt_path = "../data/triangle_decomposer.pkl"

renderer = freeze(ShapeRenderer(n_params, state_dict_path="../data/triangle_renderer.pkl")).cuda()
net = ResNet(3, 18, n_triangle * (n_params + 3))
optimizer = optim.Adam(net.parameters(), lr=3e-6)
batch_size = 64


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
    ground_truth = []
    for i in range(batch_size):
        f = np.random.uniform(0, 1, (n_triangle,  (n_params + 3)))
        ground_truth.append(f)
        train_batch.append(paint.paint_polygons_by_param(f).transpose(2, 0, 1))

    train_batch = torch.from_numpy(np.stack(train_batch, axis=0)).float()
    ground_truth = torch.from_numpy(np.stack(ground_truth, axis=0)).float()
    if use_cuda:
        net = net.cuda()
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
    gen = net(train_batch).view(batch_size, n_triangle, -1)
    canvas = paint.paint_polygons_by_renderer(gen, renderer=renderer)
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
        gen = net(train_batch).view(batch_size, n_triangle, -1)
        canvas = paint.paint_polygons_by_renderer(gen, renderer=renderer)
        loss = criterion(canvas, train_batch)
        writer.add_scalar("val/loss", loss.item(), step)
        canvas_grid = torchvision.utils.make_grid(canvas[:32])
        gt_grid = torchvision.utils.make_grid(train_batch[:32])
        writer.add_image("train/gen", canvas_grid, step, dataformats='CHW')
        writer.add_image("train/gt", gt_grid, step, dataformats='CHW')
    if step % 1000 == 0:
        save_model()
    step += 1
