import os
import json

import cv2
import numpy as np


img_size = 384
margin = 0
path = "/home/yuan/data/HisMap/SMOLdata/fewshot/manual"
outdir = "/home/yuan/data/HisMap/SMOLdata/hameln2"
data_path = "/home/yuan/data/HisMap/SMOLdata"
sheets = os.listdir(path)

def generate_hameln2_data_new_class():
    label_map = {'building': 1, 'industrial': 2, 'sport': 3, 'road1': 4, 'road2': 5}
    for sheet in sheets:
        MAP = cv2.imread(f"{path}/{sheet}/{sheet}.png", cv2.IMREAD_UNCHANGED)
        label_files = [x for x in os.listdir(f"{path}/{sheet}") if x!=f"{sheet}.png"]
        LABEL = np.zeros(MAP.shape[:2], dtype=np.uint8)
        for lf in label_files:
            label_name = lf.split('.png')[0].split('_')[-1]
            label_idx = label_map[label_name]
            L = cv2.imread(f"{path}/{sheet}/{lf}", cv2.IMREAD_UNCHANGED)
            LABEL[L == 255] = label_idx

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


def generate_hameln2_meta_info():
    stats = {}

    lbl_path = os.path.join(outdir, "lbls")
    file_list = sorted(os.listdir(lbl_path))
    train_split = []

    for f in file_list:
        name = f"hameln2.{f.split('.')[0]}"
        lbl = cv2.imread(os.path.join(lbl_path, f), cv2.IMREAD_UNCHANGED)
        stat = {}
        for i in range(6):
            stat[i] = int(np.sum(lbl == i)) / lbl.size
        stats[name] = stat

        x, y = f.split(".")[0].split("_")[-2:]
        train_split.append(name)


    with open(os.path.join(outdir, "dinfo_hameln2.json"), "w") as fh:
        json.dump(stats, fh, indent=4)
    with open(os.path.join(outdir, "hameln2_fewshot.txt"), "w") as fh:
        fh.write("\n".join(train_split))


def generate_hameln2_train_test_list():
    meta = json.load(open(f"{data_path}/dinfo_hameln2.json"))
    samples = {}
    train_split = []
    test_split = []
    selected = []

    # get hameln and bayern samples
    for k, v in meta.items():
        name = k.rsplit('_', 2)[0]
        if name not in samples:
            samples[name] = {}
        for l, r in v.items():
            if r > 0.01 and int(l) > 0:
                if l not in samples[name]:
                    samples[name][l] = [k]
                else:
                    samples[name][l].append(k)

    for k, v in samples.items():
        for l, spls in v.items():
            n_train = len(spls) // 2
            train_samples = [spls[x] for x in np.random.choice(len(spls), n_train, replace=False)]
            test_samples = [x for x in spls if x not in train_samples]
            for s in train_samples:
                if s not in selected:
                    train_split.append(s)
            for s in test_samples:
                if s not in selected:
                    test_split.append(s)

    with open(f"{data_path}/train_hameln2.txt", "w") as fh:
        fh.write("\n".join(train_split))
    with open(f"{data_path}/test_hameln2.txt", "w") as fh:
        fh.write("\n".join(test_split))


def generate_fewshot_list_old_class(n_shot=10, mode="train"):
    meta = json.load(open(f"{data_path}/dinfo_all.json"))
    # get siegfried samples
    siegfried_railway = [x.strip() for x in open(f"{data_path}/train_siegfried_railway_fewshot10.txt", "r").readlines()]
    siegfried_vineyard = [x.strip() for x in open(f"{data_path}/train_siegfried_vineyard_fewshot10.txt", "r").readlines()]
    samples = {}
    selected = []

    if mode == "val":
        with open(f"{data_path}/{mode}_{n_shot}shot.txt", "r") as fh:
            train_samples = [x.strip() for x in fh.readlines()]
        for s in train_samples:
            if s in meta:
                print(s)
                meta.pop(s)

    else:
        selected.extend(siegfried_railway)
        selected.extend(siegfried_vineyard)

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
            if r > 0.05 and int(l) > 0:
                if l not in samples[name]:
                    samples[name][l] = [k]
                else:
                    samples[name][l].append(k)

    for k, v in samples.items():
        n_sample = {k: 0 for k in v.keys()}
        for l, spls in v.items():
            if 'siegfried' in k and mode == "train":
                selected.extend([spls[x] for x in np.random.choice(len(spls), n_shot-10, replace=len(spls) < n_shot-10)])
            else:
                n_choice = max(n_shot - n_sample[l], 0)
                sample_for_l = [spls[x] for x in np.random.choice(len(spls), n_choice, replace=len(spls) < n_choice)]
                for s in sample_for_l:
                    for lbl, ratio in meta[s].items():
                        if ratio > 0.05 and int(lbl) > 0:
                            n_sample[lbl] += 1
                selected.extend(sample_for_l)



    with open(f"{data_path}/{mode}_{n_shot}shot.txt", "w") as fh:
        fh.write('\n'.join(selected))

def count_training_samples():
    data_path = "/home/yuan/data/HisMap/SMOLdata"
    all = len(open(f"{data_path}/train_all.txt", "r").readlines())
    shot10 = len(set(open(f"{data_path}/train_10shot.txt", "r").readlines()))
    shot50 = len(set(open(f"{data_path}/train_50shot.txt", "r").readlines()))
    shot100 = len(set(open(f"{data_path}/train_100shot.txt", "r").readlines()))
    print(f"Ratio of training samples 10 shot: {shot10/ all * 100:.2f}%")
    print(f"Ratio of training samples 50 shot: {shot50/ all * 100:.2f}%")
    print(f"Ratio of training samples 100 shot: {shot100/ all * 100:.2f}%")

def count_fewshot_samples():
    label_file = os.path.join(data_path, "fewshot/manual/3821_2002_crop/3821_2002_crop_road2.png")
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    ratio = np.sum(label > 0) / np.sum(label >= 0)
    print(f"Number of fewshot samples: {np.sum(label > 0)}")
    print(f"Ratio of fewshot samples: {ratio * 100:.2f}%")
# generate_hameln2_data_new_class()
# generate_hameln2_meta_info()
# generate_hameln2_train_test_list()
# generate_fewshot_list_old_class(10, 'val')
# generate_fewshot_list_old_class(50, 'val')
# generate_fewshot_list_old_class(100, 'val')
# count_training_samples()

count_fewshot_samples()