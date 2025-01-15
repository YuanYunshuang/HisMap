import glob
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.tools.map_analyzer import read_tiff
LGLN_CLS=['Woodland', 'Grassland', 'Settlement', 'FlowingWater', 'StandingWater']
LGLN_COLOR = [[85, 166, 48], [221, 223, 0], [176, 137, 104], [144, 224, 239], [3, 4, 255]]


def parse_res(filename):
    res = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'Latex' in line:
            break
        if ":" in line:
            year = line[5:9]
        else:
            values = [float(x) for x in line.split(' ') if '|' not in x and x.strip() != '']
            ious = np.array(values[:5])
            oa = values[5]
            if '189' in year:
                miou = np.mean(ious[:-1])
            else:
                miou = np.mean(ious)
            res[year] = {'miou': miou, 'oa': oa, 'ious': ious}
    return res


def eval_txt_to_latex(filename):
    ss = "Latex:\n"
    with open(filename, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Latex" in line:
            return ''.join(line[i:])

    for line in lines:
        if ":" in line:
            ss += line
            year = line[5:9]
        else:
            values = [float(x) for x in line.split(' ') if '|' not in x and x.strip() != '']
            ious = np.array(values[:5])
            oa = values[5]
            if '189' in year:
                miou = np.mean(ious[:-1])
            else:
                miou = np.mean(ious)

            ss += " & ".join([f'{x:.1f}' for x in ious])
            ss += f" & {miou:.1f} & {oa:.1f}\n"
    with open(filename, 'a') as f:
        f.write(ss)
    return ss


def rm_redundant_ckpt(path):
    files = glob.glob(os.path.join(path, '*/*/epoch*.pth'), recursive=True)
    for f in files:
        ep = int(os.path.basename(f).split('.')[0][5:])
        if ep < 3:
            os.remove(f)


def summarize_result(path):
    files = sorted(glob.glob(os.path.join(path, 'pseudo*/*/epoch5/eval.txt')))
    res = {str(k):[] for k in [1898, 1974, 1982, 1996]}
    for f in files:
        idx = f.index('pseudo-0.')
        unc = float(f[idx+7:idx+10])
        cur_res = parse_res(f)
        for year, v in cur_res.items():
            res[year].append([unc] + v['ious'].tolist() + [v['miou']] + [v['oa']])
    print("mIoU:")
    for year, values in res.items():
        mious = [f"{v[6]:.1f}" for v in values]
        print(f" {year} & {' & '.join(mious)}")
    print("OA:")
    for year, values in res.items():
        oas = [f"{v[7]:.1f}" for v in values]
        print(f" {year} & {' & '.join(oas)}")
    print("mIoU/OA:")
    for year, values in res.items():
        ss = [f"{v[6]:.1f}/{v[7]:.1f}" for v in values[4:]]
        print(f" {year} & {' & '.join(ss)}\\\\\\hline")
    return res

def plot_performance_vs_unc(res, save_dir):
    plt.rcParams.update({'font.size': 14})
    labels = ['WL', 'GL', 'SM', 'FL', 'SW', 'mIoU', 'OA']
    fig = plt.figure(figsize=(20, 5))
    axs = fig.subplots(1, 4)
    years = sorted(res.keys())
    for i, year in enumerate(years):
        v = np.array(res[year])
        for j, label in enumerate(labels):
            if label == 'mIoU' or label == 'OA':
                alpha = 1
                linewidth = 4
                linestyle = '-'
            else:
                alpha = 0.5
                linewidth = 2
                linestyle = '--'

            axs[i].plot(v[:, 0], v[:, j+1], label=label, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        axs[i].set_title(year, fontsize=14)
        axs[i].set_xlabel('Uncertainty')

    axs[0].set_ylabel("IoU / mIoU / OA in %")
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=7, frameon=False)
    # plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'bi_pseudo_unc.pdf'))


def land_use_analysis(path):
    files = glob.glob(os.path.join(path, '*.png'))
    statistic = {}
    for f in files:
        name = os.path.basename(f)
        patch, year = name.split('.')[0].split('_')
        patch, year = int(patch), int(year)
        if patch not in statistic:
            statistic[patch] = {}
        label = cv2.imread(f, cv2.IMREAD_UNCHANGED) / 50
        statistic[patch][year] = []
        for i in range(1, 6):
            n_pos = (label == i).sum()
            statistic[patch][year].append(n_pos)
    fig = plt.figure(figsize=(5, 5))
    axs = fig.subplots(1, 1)
    data = statistic[3922]
    years = sorted(data.keys())
    for i in range(5):
        pixels = [data[y][i] for y in years ]
        axs.plot(years, pixels, label=f'{i+1}')
    axs.legend()
    plt.savefig("/home/yuan/data/out/LGLN/pseudo-0.6/pseudo_12-06-03-57-17/epoch5/statistic.png")


def inconsistency(file1, file2, out_path, down_sample=3):
    img1 = cv2.imread(file1, cv2.IMREAD_UNCHANGED)[::down_sample, ::down_sample][300:3000, :3000] / 50
    img2 = cv2.imread(file2, cv2.IMREAD_UNCHANGED)[::down_sample, ::down_sample][300:3000, :3000] / 50

    for l in range(1, 6):
        mask1 = img1 == l
        mask2 = img2 == l
        img = np.zeros(img1.shape + (3,), dtype=np.uint8)
        img[..., 1] = mask1 * 255
        img[..., 2] = mask2 * 255
        cv2.imwrite(os.path.join(out_path, f"3821_{LGLN_CLS[l-1]}.png"), img)
        iou = np.logical_and(mask1, mask2).sum() / np.logical_or(mask1 , mask2).sum()
        print(l, iou)
    # inconsistency = img1 != img2
    # inconsistency_mask = np.logical_not(inconsistency).astype(float) + inconsistency * 0.3
    # inconsistency_mask = (inconsistency_mask * 255).astype(np.uint8)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2RGBA)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2RGBA)
    # img1[..., -1] = inconsistency_mask
    # img2[..., -1] = inconsistency_mask
    # cv2.imwrite(file1.replace('.jpg', '_crop.png'), img1)
    # cv2.imwrite(file2.replace('.jpg', '_crop.png'), img2)
    # cv2.imwrite(out_file, inconsistency.astype(np.uint8) * 255)


def visual_result(pred_file, label_file, out_path, down_sample=3):
    pred = cv2.imread(pred_file, cv2.IMREAD_UNCHANGED)[::down_sample, ::down_sample][0:1000, 400:1600] / 50
    label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)[::down_sample, ::down_sample][0:1000, 400:1600] / 50
    for l in range(1, 6):
        mask1 = label == l
        mask2 = pred == l
        img = np.zeros(pred.shape + (3,), dtype=np.uint8)
        img[..., 1] = mask1 * 255
        img[..., 2] = mask2 * 255
        cv2.imwrite(os.path.join(out_path, f"3922_1996_{LGLN_CLS[l-1]}.png"), img)

def main():
    year = 1898
    l = 5
    cls = LGLN_CLS[l-1]
    def downsample_and_crop(img, down_sample=4):
        return img[0:3000, 1200:4800][::down_sample, ::down_sample]

    all_pred_b = cv2.imread(f"/home/yuan/data/out/LGLN2023/all/epoch20/holistic_seg_result/3922_{year}.png", cv2.IMREAD_UNCHANGED) / 50
    img = (downsample_and_crop(all_pred_b) == l) * 255
    cv2.imwrite(f"/home/yuan/data/HisMap/LGLN/dataset/icc2025/3922_{year}_{cls}_all_pred.jpg", img)

    map_b = read_tiff(f'/home/yuan/data/HisMap/LGLN/dataset/test/maps/3922_{year}.tif')
    map_b = downsample_and_crop(map_b)
    cv2.imwrite(f"/home/yuan/data/HisMap/LGLN/dataset/icc2025/3922_{year}_map.jpg", map_b)

    label_a = cv2.imread('/home/yuan/data/HisMap/LGLN/dataset/3922_label_2023/3922_2023.png', cv2.IMREAD_UNCHANGED) / 50
    img = (downsample_and_crop(label_a) == l) * 255
    cv2.imwrite(f"/home/yuan/data/HisMap/LGLN/dataset/icc2025/3922_2023_{cls}_l2023.png", img)

    label_b = cv2.imread(f'/home/yuan/data/HisMap/LGLN/dataset/test/label/3922_{year}.png', cv2.IMREAD_UNCHANGED) / 50
    img = (downsample_and_crop(label_b) == l) * 255
    cv2.imwrite(f"/home/yuan/data/HisMap/LGLN/dataset/icc2025/3922_{year}_{cls}_l{year}.png", img)

    pseudo_label_b = cv2.imread(f'/home/yuan/data/out/LGLN2023/pseudo-all/pseudo-0.8/pseudo_12-15-05-10-33/epoch3/holistic_seg_result/3922_{year}.png', cv2.IMREAD_UNCHANGED) / 50
    img = (downsample_and_crop(pseudo_label_b) == l) * 255
    cv2.imwrite(f"/home/yuan/data/HisMap/LGLN/dataset/icc2025/3922_{year}_{cls}_pl{year}.png", img)






if __name__ == '__main__':
    pass
    # rm_redundant_ckpt("/home/yuan/data/out/LGLN2023/pseudo-all")
    # visual_result(
    #     "/home/yuan/data/out/LGLN2023/pseudo-all/pseudo-0.8/pseudo_12-15-05-10-33/epoch3/holistic_seg_result/3922_1996.png",
    #     "/home/yuan/data/HisMap/LGLN/dataset/test/label/3922_1996.png",
    #     "/home/yuan/data/HisMap/LGLN/dataset/icc2025"
    # )
    # files = sorted(glob.glob(os.path.join("/home/yuan/data/out/LGLN",
    #                                       '**/eval.txt'), recursive=True))
    # for f in files:
    #     eval_txt_to_latex(f)
    # eval_txt_to_latex("/home/yuan/data/out/LGLN2023/pseudo-0.8/pseudo_12-15-05-10-33/epoch3/eval.txt")
    res = summarize_result("/home/yuan/data/out/LGLN/trace")
    # plot_performance_vs_unc(res, "/home/yuan/data/out/LGLN2023/pseudo-all")
    # land_use_analysis("/home/yuan/data/out/LGLN/pseudo-0.6/pseudo_12-06-03-57-17/epoch5/holistic_seg_result_")
    # inconsistency(
    #     "/home/yuan/data/HisMap/LGLN/dataset/train/label/3821_1974.png",
    #     "/home/yuan/data/HisMap/LGLN/dataset/train/label2023/3821_2023.png",
    #     "/home/yuan/data/HisMap/LGLN/dataset/train/tmp",
    # )