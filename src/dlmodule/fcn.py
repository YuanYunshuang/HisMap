import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, n_params, color=False, state_dict_path=None):
        super(FCN, self).__init__()
        n_color = 3 if color else 1
        self.color = color
        self.fc1 = (nn.Linear(n_params, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32 * n_color, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32 * n_color, 32 * n_color, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8 * n_color, 16 * n_color, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16 * n_color, 16 * n_color, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4 * n_color, 8 * n_color, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8 * n_color, 4 * n_color, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)
        if self.color:
            self.activation = nn.Hardtanh(min_val=0., max_val=1.)
        else:
            self.activation = nn.Sigmoid()

        if state_dict_path is not None and os.path.exists(state_dict_path):
            self.load_state_dict(torch.load(state_dict_path))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))

        x = self.activation(x)
        if self.color:
            return x
        else:
            return 1 - x.view(-1, 128, 128)


def linear_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU()
    )


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
        nn.PixelShuffle(2)
    )


class ShapeRenderer(nn.Module):
    def __init__(self, n_params, state_dict_path=None):
        super(ShapeRenderer, self).__init__()
        self.n_params = n_params
        self.fc1 = linear_layer(n_params, 512)
        self.fc2 = linear_layer(512, 1024)
        self.fc3 = linear_layer(1024, 2048)
        self.fc4 = linear_layer(2048, 4096)
        self.conv1 = conv_block(16, 32)
        self.conv2 = conv_block(8, 16)
        self.conv3 = conv_block(4, 4)

        self.color_encoder = linear_layer(3, 128)
        self.out_layer = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Hardtanh(0, 1),
        )

        if state_dict_path is not None and os.path.exists(state_dict_path):
            self.load_state_dict(torch.load(state_dict_path))

    def forward(self, params):
        B, N, C = params.shape
        params = params.view(-1, C)
        color_emb = self.color_encoder(params[:, -3:])
        x = self.fc1(params[:, :-3])
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(-1, 16, 16, 16)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.sigmoid()

        x = x + color_emb.view(B * N, -1, 1, 1)
        x = self.out_layer(x.view(B, N, -1, 128, 128).max(1).values)

        return x
