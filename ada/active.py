import math
import torch

import numpy as np
import torch.nn.functional as F
from torch import nn

from PIL import Image
from tqdm import tqdm


class SpatialPurity(nn.Module):

    def __init__(self, in_channels=6, padding_mode='zeros', size=3):
        super(SpatialPurity, self).__init__()
        assert size % 2 == 1, "error size"
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=size,
                              stride=1, padding=int(size / 2), bias=False, padding_mode=padding_mode,
                              groups=in_channels)
        a = torch.ones((size, size), dtype=torch.float32)
        a = a.unsqueeze(dim=0).unsqueeze(dim=0)
        a = a.repeat([in_channels, 1, 1, 1])
        a = nn.Parameter(a)
        self.conv.weight = a
        self.conv.requires_grad_(False)

    def forward(self, x):
        summary = self.conv(x)
        # summary: (b, in_channels, h, w)
        count = torch.sum(summary, dim=1, keepdim=True)
        # count: (b, 1, h, w)
        dist = summary / count
        # dist: (b, in_channels, h, w)
        spatial_purity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True)
        # (b, 1, h, w), normally b = 1, (1, 1, h, w)
        return spatial_purity


class FloatingRegionScore(nn.Module):

    def __init__(self, in_channels=6, padding_mode='zeros', size=33):
        """
        purity_conv: size*size
        entropy_conv: size*size
        """
        super(FloatingRegionScore, self).__init__()
        self.in_channels = in_channels
        assert size % 2 == 1, "error size"
        self.purity_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=size,
                                     stride=1, padding=int(size / 2), bias=False,
                                     padding_mode=padding_mode, groups=in_channels)
        weight = torch.ones((size, size), dtype=torch.float32)
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
        weight = weight.repeat([in_channels, 1, 1, 1])
        weight = nn.Parameter(weight)
        self.purity_conv.weight = weight
        self.purity_conv.requires_grad_(False)

        self.entropy_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size,
                                      stride=1, padding=int(size / 2), bias=False,
                                      padding_mode=padding_mode)
        weight = torch.ones((size, size), dtype=torch.float32)
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
        weight = nn.Parameter(weight)
        self.entropy_conv.weight = weight
        self.entropy_conv.requires_grad_(False)

    def forward(self, logit):
        """
        return:
            score, purity, entropy
        """
        logit = logit.squeeze(dim=0)  # [in_channels, h ,w]
        p = torch.softmax(logit, dim=0)  # [in_channels, h, w]

        pixel_entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0).unsqueeze(dim=0).unsqueeze(dim=0) / math.log(
            19)  # [1, 1, h, w]
        region_sum_entropy = self.entropy_conv(pixel_entropy)  # [1, 1, h, w]

        predict = torch.argmax(p, dim=0)  # [h, w]
        one_hot = F.one_hot(predict, num_classes=self.in_channels).float()
        one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)  # [1, 19, h, w]
        summary = self.purity_conv(one_hot)  # [1, 19, h, w]
        count = torch.sum(summary, dim=1, keepdim=True)  # [1, 1, h, w]
        dist = summary / count  # [1, 19, h, w]
        region_impurity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True) / math.log(19)  # [1, 1, h, w]
        prediction_uncertainty = region_sum_entropy / count  # [1, 1, h, w]

        score = region_impurity * prediction_uncertainty
        return score.squeeze(dim=0).squeeze(dim=0), region_impurity.squeeze(dim=0).squeeze(
            dim=0), prediction_uncertainty.squeeze(dim=0).squeeze(dim=0)


def pixel_selection(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()

    active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (1280 * 640) * (2048 * 1024))
    calculate_purity = SpatialPurity(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    mask_radius = cfg.ACTIVE.RADIUS_K

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):

                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                output = output.squeeze(dim=0)
                p = torch.softmax(output, dim=0)
                entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
                pseudo_label = torch.argmax(p, dim=0)
                one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
                one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
                purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
                score = entropy * purity

                score[active] = -float('inf')

                for pixel in range(active_pixels):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    end_w = w + mask_radius + 1
                    end_h = h + mask_radius + 1
                    # mask out
                    score[start_h:end_h, start_w:end_w] = -float('inf')
                    active[start_h:end_h, start_w:end_w] = True
                    selected[h, w] = True
                    # active sampling
                    active_mask[h, w] = ground_truth[h, w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()


def region_selection(cfg, feature_extractor, classifier, tgt_epoch_loader):

    feature_extractor.eval()
    classifier.eval()

    floating_region_score = FloatingRegionScore(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    per_region_pixels = (2 * cfg.ACTIVE.RADIUS_K + 1) ** 2
    active_radius = cfg.ACTIVE.RADIUS_K
    mask_radius = cfg.ACTIVE.RADIUS_K * 2
    active_ratio = cfg.ACTIVE.RATIO / len(cfg.ACTIVE.SELECT_ITER)

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = \
                tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                num_pixel_cur = size[0] * size[1]
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                score, purity, entropy = floating_region_score(output)

                score[active] = -float('inf')

                active_regions = math.ceil(num_pixel_cur * active_ratio / per_region_pixels)

                for pixel in range(active_regions):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    active_start_w = w - active_radius if w - active_radius >= 0 else 0
                    active_start_h = h - active_radius if h - active_radius >= 0 else 0
                    active_end_w = w + active_radius + 1
                    active_end_h = h + active_radius + 1

                    mask_start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    mask_start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    mask_end_w = w + mask_radius + 1
                    mask_end_h = h + mask_radius + 1

                    # mask out
                    score[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
                    active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
                    selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
                    # active sampling
                    active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                        ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()