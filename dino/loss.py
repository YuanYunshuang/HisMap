import torch
from torch import nn
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

from src.losses.focal_loss import FocalLoss

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, distributed=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.distributed = distributed
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, labels, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().split(15)

        teacher_out_label = []
        for i, x in enumerate(teacher_out):
            label = F.one_hot(labels[i, 0].long(), num_classes=5).float()
            tol = torch.cat([x, label.unsqueeze(0)], dim=0)
            teacher_out_label.append(tol)
        teacher_out_label = torch.cat(teacher_out_label, dim=0)

        total_loss = torch.sum(-teacher_out_label * F.log_softmax(student_out, dim=-1), dim=-1).mean()
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if self.distributed:
            dist.all_reduce(batch_center)  # default is op=ReduceOp.SUM
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOFocalLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp=0, teacher_temp=0.4,
                 warmup_teacher_temp_epochs=50, nepochs=100, student_temp=0.1,
                 center_momentum=0.9, distributed=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.distributed = distributed
        # self.register_buffer("center", torch.zeros(1, 256, 256, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(1, 0.6, 41, ),
            np.ones(nepochs - 40) * 0.6
        ))
        self.focal_loss = FocalLoss(use_sigmoid=False)

    def forward(self, student_output, teacher_output, labels, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        if student_output.shape[1] == 5:
            student_output = student_output.permute(0, 2, 3, 1)
            teacher_output = teacher_output.permute(0, 2, 3, 1)

        # student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        # teacher_out = F.sigmoid((teacher_output - self.center) / temp)
        teacher_out = F.sigmoid(teacher_output)
        teacher_out = teacher_out.detach().split(15)

        teacher_out_label = []
        for i, x in enumerate(teacher_out):
            label = labels[i, 0].long().unsqueeze(0)
            x_max, x_max_inds = x.max(dim=-1)
            x_lbls = label.repeat(x.shape[0], 1, 1)
            x_contra_mask = x_max_inds != label
            x_valid_mask = x_max > temp
            x_lbls_update_mask = torch.logical_and(x_contra_mask, x_valid_mask)
            x_lbls[x_lbls_update_mask] = x_max_inds[x_lbls_update_mask]

            tol = torch.cat([x_lbls, label], dim=0)
            tol = F.one_hot(tol, num_classes=5).float()


            teacher_out_label.append(tol)
        teacher_out_label = torch.cat(teacher_out_label, dim=0)
        # cared = torch.cat(cared, dim=0)

        pred = student_output.flatten(0, 2)
        gt = teacher_out_label.flatten(0, 2)
        # gt[gt>0.5] = 1.0

        total_loss = self.focal_loss(pred, gt, temp=epoch)
        # self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if self.distributed:
            dist.all_reduce(batch_center)  # default is op=ReduceOp.SUM
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
