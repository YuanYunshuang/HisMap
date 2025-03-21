import os
import argparse
from datetime import datetime

import torch
import cv2
import numpy as np

from dataset import get_dataloader
from model import ADAseg
from src.utils.train_utils import build_optimizer, build_lr_scheduler
from loss import NegativeLearningLoss, LocalConsistentLoss, FocalLoss


def vis_res(img, res, lbl, downx=3):
    res = res.softmax(1)[0].detach()
    img = img[0].permute(1, 2, 0).detach().cpu().numpy()
    img = ((img + 1) / 2 * 255).astype(np.uint8)
    res = (res.cpu().numpy() * 255).astype(np.uint8)[..., None].repeat(3, axis=3)
    lbl = lbl[0].detach().cpu().numpy()
    lbl = [((lbl==x)*255).astype(np.uint8)[..., None].repeat(3, axis=-1) for x in range(1, 6)]
    vis1 = np.hstack((img, res[0], res[1], res[2], res[3], res[4]))[::downx, ::downx]
    vis2 = np.hstack((img, *lbl))[::downx, ::downx]
    vis = np.vstack((vis1, vis2))
    cv2.imwrite("/home/yuan/Downloads/tmp.png", vis)

def save_ckpt(model, save_path, epoch, lr_scheduler):
    print(f'Saving checkpoint to {save_path}.')
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, save_path)


def main(args):
    now = datetime.now().strftime('%m-%d-%H-%M-%S')
    logdir = os.path.join(args.log_dir, now)
    os.makedirs(logdir)
    dataloader = get_dataloader(args.data_path, args.batch_size)
    model = ADAseg('UNet', num_classes=args.num_cls).cuda().train()
    optimizer = build_optimizer(model, dict(lr=args.lr, weight_decay=1e-2, betas=[0.9, 0.98]))
    lr_scheduler = build_lr_scheduler(optimizer, dict(policy='MultiStepLR', milestones=[10, 15], gamma=0.1),
                                      len(dataloader))

    # losses
    sup_criterion = FocalLoss(bg_idx=0, ignore_idx=255)

    # start train
    epoch = 1
    itr = 1
    total_itr = len(dataloader)
    with torch.autograd.set_detect_anomaly(True):
        for i in range(0, args.epochs):
            for img, lbl in dataloader:
                img = img.cuda()
                lbl = lbl.cuda()
                optimizer.zero_grad()
                res = model(img)

                total_loss = sup_criterion(res, lbl)

                total_loss.backward()
                optimizer.step()

                rec_lr = lr_scheduler.get_last_lr()[0]
                itr += 1

                if itr % 10 == 0:
                    print(f"Epoch{epoch}[{itr}/{total_itr}] LR:{rec_lr:.6f} Loss: {total_loss.item():.6f}\t")
                if itr % 50 == 0:
                    vis_res(img, res, lbl)

            lr_scheduler.step()
            epoch += 1
            itr = 1

            # save checkpoint
            save_ckpt(model, os.path.join(logdir, 'model_latest.pth'), epoch, lr_scheduler)
            if epoch % 10 == 0:
                save_ckpt(model, os.path.join(logdir, f'model_ep{epoch}.pth'), epoch, lr_scheduler)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train | test")
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--log-dir", type=str, default=os.path.join(os.path.dirname(__file__),"../logs"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--data-path", type=str, default="/home/yuan/data/HisMap/MapSAM/train_stage0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--negative-threshold", type=float, default=0.05)
    parser.add_argument("--consistent-loss", type=float, default=0.1)
    parser.add_argument("--negative-loss", type=float, default=1.0)
    parser.add_argument("--num-cls", type=int, default=5)
    args = parser.parse_args()

    main(args)
