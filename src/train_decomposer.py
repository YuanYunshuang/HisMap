import os.path

import cv2
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from src.dlmodule.fcn import FCN
from src.dlmodule.resnet import ResNet
from src.utils.paint import paint_stroke, paint_polygons
import torch.optim as optim
import torchvision

writer = SummaryWriter("../train_log/src")

use_cuda = torch.cuda.is_available()
step = 0
paint_func = paint_polygons
n_params = 6
n_triangle = 5
ckpt_path = "../data/triangle_decomposer.pkl"

criterion = nn.MSELoss()
renderer = FCN(n_params)
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


load_weights()
while step < 500000:
    net.train()
    train_batch = []
    ground_truth = []
    for i in range(batch_size):
        f = np.random.uniform(0, 1, n_triangle * (n_params + 3))
        train_batch.append(f)
        ground_truth.append(paint_func(f.reshape(n_triangle, -1)))

    train_batch = torch.from_numpy(np.stack(train_batch, axis=0)).float()
    ground_truth = torch.from_numpy(np.stack(ground_truth, axis=0)).float()
    if use_cuda:
        net = net.cuda()
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
    gen = net(train_batch)
    optimizer.zero_grad()
    loss = criterion(gen, ground_truth)
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
        loss = criterion(gen, ground_truth)
        writer.add_scalar("val/loss", loss.item(), step)
        gen_grid = torchvision.utils.make_grid(gen[:32].unsqueeze(1).repeat(1, 3, 1, 1))
        gt_grid = torchvision.utils.make_grid(ground_truth[:32].unsqueeze(1).repeat(1, 3, 1, 1))
        writer.add_image("train/gen", gen_grid, step, dataformats='CHW')
        writer.add_image("train/gt", gt_grid, step, dataformats='CHW')
    if step % 1000 == 0:
        save_model()
    step += 1
