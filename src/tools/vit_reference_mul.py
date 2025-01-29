import os, sys
import argparse
import math
import timeit

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import get_dataloader
from src.sdl import get_model
from src.utils.misc import save_json
from src.config import load_config, save_config
from src.utils.train_utils import seed_everything, load_model_dict, load_tensors_to_gpu


torch.backends.cudnn.benchmark = True


def draw_attn_on_img(img, attn, conf, label, axs, title=None):
    # Resize attention map to match the image size (512x512)
    attn = np.clip(attn.cpu().numpy(), 0, 1)
    attn_size = int(math.sqrt(attn.shape[1]))
    attn = attn.reshape(attn.shape[0], attn_size, attn_size)
    s = img.shape[0] // attn_size
    attention_map_resized = np.kron(attn, np.ones((1, s, s)))

    for at, cf, lbl, ax in zip(attn, conf, label, axs):
        ax.imshow(img, extent=(0, img.shape[0], 0, img.shape[1]))  # Show the image
        ax.imshow(at, cmap='jet', alpha=0.5, extent=(0, img.shape[0], 0, img.shape[1]), vmin=0, vmax=1)
        ax.set_title(f"{title}_{lbl.item()}_{cf.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-workers", type=int)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--label-pattern", type=str, default="****")
    parser.add_argument("--num-cls", type=int, default=4)
    args = parser.parse_args()
    args.__setattr__('mode', 'inference')

    if args.seed is not None:
        seed_everything(2023)

    cfgs = load_config(args)
    if args.batch_size is not None:
        cfgs['DATASET']['batch_size'] = args.batch_size
    if args.n_workers is not None:
        cfgs['DATASET']['n_workers'] = args.n_workers
    if args.data_path is not None:
        cfgs['DATASET']['data_path'] = args.data_path
    if args.label_pattern is not None:
        cfgs['DATASET']['label_pattern'] = args.label_pattern

    # Initialize modules
    dataloader = get_dataloader(cfgs['DATASET'], 'inference')
    cfgs['MODEL']['mode'] = args.mode
    model = get_model(cfgs['MODEL'])
    ckpt_dict = torch.load(args.load_from, weights_only=False)
    load_model_dict(model, ckpt_dict['model'])
    model.cuda().eval()

    # print(model.vit)

    # Inference process
    cnt = 1
    res_dict = {}
    with torch.no_grad():
        for data in tqdm(dataloader):
            load_tensors_to_gpu(data)
            res_names = []
            res_labels = []
            res_attn = []
            res_idx = []
            res_conf = []
            for i, (patch, year, x, y, img, label) in enumerate(zip(
                    data['patch'],
                    data['year'],
                    data['x'],
                    data['y'],
                    data['img'],
                    data['label']
            )):
                name = f"{patch.split('/')[-1]}_{year}_{x}_{y}"
                res_names.append(name)
                res_labels.append(label)
                # res_dict[name] = {'label': label.item(), 'conf': [], 'max_vals': [], 'max_inds': []}
            n_patches = (data['img'].shape[-1] // 64) ** 2
            patch_drop_mask = torch.zeros((args.num_cls, len(data['img']), n_patches), dtype=torch.float, device=data['img'].device)
            for n in range(n_patches):
                out_dict = model(data, drop_patch_mask=None if n==0 else patch_drop_mask.bool().float())
                attn = torch.stack([ca.attn_weight[:, 0, 0, 1:] for ca in model.vit.cross_attn], dim=0)
                max_vals, max_inds = attn.max(dim=-1, keepdim=True)
                patch_drop_mask.scatter_(-1, max_inds, max_vals)

                res_attn.append(max_vals.squeeze().permute(1, 0).tolist())
                res_idx.append(max_inds.squeeze().permute(1, 0).tolist())
                res_conf.append(out_dict['conf'].tolist())

                # add result to the dict
                # for i, (conf, max_val, max_ind) in enumerate(zip(out_dict['conf'], max_vals, max_inds)):
                #     name = f"{out_dict['patch'][i].split('/')[-1]}_{out_dict['year'][i]}_{out_dict['x'][i]}_{out_dict['y'][i]}"
                #     res_dict[name]['conf'].append(conf[1].item())
                #     res_dict[name]['max_vals'].append(max_val.item())
                #     res_dict[name]['max_inds'].append(max_ind.item())

                # s = int(math.sqrt(attn.shape[-1]))
                # attn = attn.reshape(-1, s, s)
                if args.visualize and n == n_patches - 1:
                    fig = plt.figure(figsize=(16, 16))
                    axs = fig.subplots(4, 4)
                    for i, (patch, year, x, y, img, conf, label) in enumerate(zip(
                            out_dict['patch'],
                            out_dict['year'],
                            out_dict['x'],
                            out_dict['y'],
                            out_dict['img'],
                            out_dict['conf'],
                            out_dict['label']
                    )):
                        if i >= 4: break
                        img = img.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
                        draw_attn_on_img(img[::2, ::2], patch_drop_mask[:, i], conf, label, axs[i], title=f'{patch}_{year}_{x}_{y}')
                    plt.tight_layout()
                    plt.savefig(f'/home/yuan/data/HisMap/LGLN/dataset/attn/inf_imgs/{cnt}.jpg')
                    plt.close()
            cnt += 1

            for i, name in enumerate(res_names):
                res_dict = {
                    'attn': [x[i] for x in res_attn],
                    'idx': [x[i] for x in res_idx],
                    'conf': [x[i] for x in res_conf],
                    'label': res_labels[i].tolist()
                }
                save_json(res_dict, f"/home/yuan/data/HisMap/LGLN/dataset/attn/inference/{name}.json")

