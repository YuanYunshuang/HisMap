# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import os, sys
import logging

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob

from smolsam.modeling.dataset import MapDataset
from smolsam.modeling.mapsam import MapSAM
from smolsam import sam_model_registry
from smolsam.utils.vis import vis_train
from smolsam.utils.train_utils import adjust_learning_rate_cosine

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6



def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--root-dir",
        type=str,
        default="/home/yuan/data/HisMap/SMOLdata",
        help="path to training dataset; two subfolders: lbls and imgs",
    )
    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        default="vit_b",
        help="model type",
    )

    cur_path = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument("-t", "--task_name", type=str, default="default", help="task name")
    parser.add_argument("-w", "--checkpoint", type=str, default=f"{cur_path}/../model_zoo/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("-o","--output_dir", type=str, default=f"{cur_path}/logs")
    # train
    parser.add_argument("-e", "--num_epochs", type=int, default=100)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-nw", "--num_workers", type=int, default=4)
    # Optimizer parameters
    parser.add_argument("--weight-decay", type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("--lr", type=float, default=5e-5, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--warmup-epochs", type=int, default=0, metavar="warmup epochs", help="total number of warmup epochs")
    parser.add_argument("--use-wandb", type=bool, default=False, help="use wandb to monitor training")
    parser.add_argument("--use-amp", action="store_true", default=False, help="use amp")
    parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--plot-loss", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    # parser.add_argument("--train-patch-emb", action="store_true", default=False, help="Train patch embedding parameters")
    parser.add_argument("--disable-lr-scheduler", action="store_true", default=False)
    parser.add_argument("--from-scratch", action="store_true", default=False)
    parser.add_argument("--data-mode", type=str, default="train50", help="[train50 | train75 | fewshot] percentage of data split for training.")
    args = parser.parse_args()

    if args.use_wandb:
        import wandb

        wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_path": args.tr_npy_path,
                "model_type": args.model_type,
            },
        )
    return args



def main():
    args = setup_parser()
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.output_dir, args.task_name + "-" + run_id)
    device = torch.device(args.device)
    # %% set up model
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    logger = logging.getLogger("SMOL-MapSAM")
    logging.basicConfig(filename=join(model_save_path, "log.txt"), level=logging.INFO)

    if args.from_scratch:
        args.checkpoint = None
        args.warmup_epochs = 1
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    mapsam = MapSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder
    ).to(device)
    mapsam.train()

    logger.info(">>>>>>>>>>>>>>>>>>CONFIGS<<<<<<<<<<<<<<<<")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"# total parameters: "
                f"{sum(p.numel() for p in mapsam.parameters())}")  # 93735472
    logger.info(f"# trainable parameters: "
                f"{sum(p.numel() for p in mapsam.parameters() if p.requires_grad)}")  # 93729252

    img_mask_encdec_params = list(mapsam.image_encoder.parameters()) + list(
        mapsam.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    logger.info(f"# image encoder and mask decoder parameters: "
                f"{sum(p.numel() for p in img_mask_encdec_params if p.requires_grad)}")  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    lrs = []
    best_loss = 1e10
    train_dataset = MapDataset(args.root_dir, mode=args.data_mode)

    logger.info(f"# training samples: {len(train_dataset)}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            mapsam.load_state_dict(checkpoint["model"])
            if not args.data_mode == "fewshot":
                optimizer.load_state_dict(checkpoint["optimizer"])
                start_epoch = checkpoint["epoch"] + 1
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    logger.info(">>>>>>>>>>>>>>>>>>TART TRAINING <<<<<<<<<<<<<<<<<<")
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        if args.disable_lr_scheduler:
            cur_lr = optimizer.param_groups[0]["lr"]
        else:
            cur_lr = adjust_learning_rate_cosine(optimizer, base_lr=args.lr, epoch=epoch,
                                                 total_epoch=num_epochs,
                                                 warmup_epochs=args.warmup_epochs)

        pbar = tqdm(train_dataloader)
        pbar.set_postfix({'Epoch/Step': f'{epoch}/0', 'LR': f'{cur_lr:e}', 'Loss': f'{0:.6f}'})
        for step, data in enumerate(pbar):
            optimizer.zero_grad()
            img_src, lbl_src, img_tgt, lbl_tgt = [d.to(device) for d in data[:4]]
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    mapsam_pred = mapsam(img_src, lbl_src, img_tgt)
                    loss = (seg_loss(mapsam_pred, lbl_tgt.unsqueeze(1)) +
                            ce_loss(mapsam_pred, lbl_tgt.float().unsqueeze(1)))
                    # loss = ce_loss(mapsam_pred, lbl_tgt.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                mapsam_pred = mapsam(img_src, lbl_src, img_tgt)
                loss = (seg_loss(mapsam_pred, lbl_tgt.unsqueeze(1)) +
                        ce_loss(mapsam_pred, lbl_tgt.float().unsqueeze(1)))
                # loss = ce_loss(mapsam_pred, lbl_tgt.float().unsqueeze(1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

            pbar.set_postfix({'Epoch/Step': f'{epoch}/{step + 1}', 'LR': f'{cur_lr:e}', 'Loss': f'{epoch_loss / (step + 1):.6f}'})

            if (step + 1) % 10 == 0:
                logger.info({'Epoch/Step': f'{epoch}/{step + 1}', 'LR': f'{cur_lr:e}', 'Loss': f'{epoch_loss / (step + 1):.6f}'})
            if args.debug and (step + 1) % 50 == 0:
                vis_train(img_src, lbl_src, img_tgt, lbl_tgt, mapsam_pred)

        epoch_loss /= step
        losses.append(epoch_loss)
        lrs.append(cur_lr)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        logger.info(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        ## save the latest model
        checkpoint = {
            "model": mapsam.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": mapsam.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

        # save loss
        with open(join(model_save_path, "medsam_loss.txt"), "w") as f:
            for x, y in zip(losses, lrs):
                f.write(f'{y:.6f}, {x:.6f}\n')
        # %% plot loss
        if args.plot_loss:
            import matplotlib.pyplot as plt
            plt.plot(losses)
            plt.title("Dice + Cross Entropy Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
            plt.close()


if __name__ == "__main__":
    main()
