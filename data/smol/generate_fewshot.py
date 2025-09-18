import os
import json

import cv2
import numpy as np


img_size = 384
margin = 128
path = "/home/yuan/data/HisMap/SMOLdata/fewshot/manual"
outdir = "/home/yuan/data/HisMap/SMOLdata/fewshot"
sheets = os.listdir(path)

def generate_data():
    for sheet in sheets:
        lbl_cnt = 0
        MAP = cv2.imread(f"{path}/{sheet}/{sheet}.png", cv2.IMREAD_UNCHANGED)
        label_files = [x for x in os.listdir(f"{path}/{sheet}") if x!=f"{sheet}.png"]
        LABEL = np.zeros(MAP.shape[:2], dtype=np.uint8)
        for lf in label_files:
            lbl_cnt += 1
            L = cv2.imread(f"{path}/{sheet}/{lf}", cv2.IMREAD_UNCHANGED)
            LABEL[L == 255] = lbl_cnt

        H, W = min(MAP.shape[0], LABEL.shape[0]), min(MAP.shape[1], LABEL.shape[1])
        x = np.arange(0, W, img_size-margin)
        y = np.arange(0, H, img_size-margin)
        x[-1] = W - img_size
        y[-1] = H - img_size
        yy, xx = np.meshgrid(x, y)
        xys = np.vstack((xx.flatten(), yy.flatten())).T

        for xy in xys:
            img = MAP[xy[0]:xy[0] + img_size, xy[1]:xy[1] + img_size]
            lbl = LABEL[xy[0]:xy[0] + img_size, xy[1]:xy[1] + img_size]
            if img.ndim == 2:
                img = img[..., np.newaxis].repeat(3, axis=2)
            img = np.pad(img, ((0, img_size - img.shape[0]), (0, img_size - img.shape[1]), (0, 0)),
                         'constant', constant_values=0)
            lbl = np.pad(lbl, ((0, img_size - lbl.shape[0]), (0, img_size - lbl.shape[1])),
                         'constant', constant_values=255)

            name = 'c' + sheet.replace('_crop', '')
            cv2.imwrite(os.path.join(outdir, 'imgs', f'{name}_{xy[0]}_{xy[1]}.png'), img)
            cv2.imwrite(os.path.join(outdir, 'lbls', f'{name}_{xy[0]}_{xy[1]}.png'), lbl)


def generate_meta_info():
    stats = {}

    lbl_path = os.path.join(outdir, "lbls")
    file_list = sorted(os.listdir(lbl_path))
    train_split = []

    for f in file_list:
        name = f"fewshot.{f.split('.')[0]}"
        lbl = cv2.imread(os.path.join(lbl_path, f), cv2.IMREAD_UNCHANGED)
        stat = {}
        for i in range(5):
            stat[i] = int(np.sum(lbl == i)) / lbl.size
        stats[name] = stat

        x, y = f.split(".")[0].split("_")[-2:]
        train_split.append(name)


    with open(os.path.join(outdir, "dinfo_fewshot.json"), "w") as fh:
        json.dump(stats, fh, indent=4)
    with open(os.path.join(outdir, "fewshot.txt"), "w") as fh:
        fh.write("\n".join(train_split))


def generate_fewshot_list_old_class(n_shot=10):
    data_path = "/home/yuan/data/HisMap/SMOLdata"
    meta = json.load(open(f"{data_path}/dinfo_all.json"))
    samples = {}
    selected = []

    # get siegfried samples
    siegfried_railway = [x.strip() for x in open(f"{data_path}/train_siegfried_railway_fewshot10.txt", "r").readlines()]
    siegfried_vineyard = [x.strip() for x in open(f"{data_path}/train_siegfried_vineyard_fewshot10.txt", "r").readlines()]

    # get hameln and bayern samples
    for k, v in meta.items():
        if k in siegfried_railway or k in siegfried_vineyard:
            continue
        if 'siegfried' in k:
            name = k.split('_')[0]
        else:
            name = k.rsplit('_', 2)[0]
        if name not in samples:
            samples[name] = {}
        for l, r in v.items():
            if r > 0.1 and int(l) > 0:
                if l not in samples[name]:
                    samples[name][l] = [k]
                else:
                    samples[name][l].append(k)

    for k, v in samples.items():
        for l, spls in v.items():
            if 'siegfried' in k:
                selected.extend([spls[x] for x in np.random.choice(len(spls), n_shot-10, replace=len(spls) < n_shot-10)])
            else:
                selected.extend([spls[x] for x in np.random.choice(len(spls), n_shot, replace=len(spls) < n_shot)])

    selected.extend(siegfried_railway)
    selected.extend(siegfried_vineyard)

    with open(f"{data_path}/train_{n_shot}shot.txt", "w") as fh:
        fh.write('\n'.join(selected))


generate_fewshot_list_old_class(n_shot=100)