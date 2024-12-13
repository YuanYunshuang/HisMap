import glob
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from openai import files
from sympy.abc import alpha

from src.tools.map_analyzer import read_tiff


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
    files = glob.glob(os.path.join(path, '*/epoch*.pth'))
    for f in files:
        ep = int(os.path.basename(f).split('.')[0][5:])
        if ep < 5:
            os.remove(f)


def summarize_result(path):
    files = sorted(glob.glob(os.path.join(path, 'pseudo*/*/epoch5/eval.txt')))
    res = {str(k):[] for k in [1898, 1974, 1982, 1996]}
    for f in files:
        idx = f.index('pseudo-')
        unc = float(f[idx+7:idx+10])
        cur_res = parse_res(f)
        for year, v in cur_res.items():
            res[year].append([unc] + v['ious'].tolist() + [v['miou']] + [v['oa']])
    return res

def plot(res, save_dir):
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



if __name__ == '__main__':
    pass
    # files = sorted(glob.glob(os.path.join("/home/yuan/data/out/LGLN",
    #                                       '**/eval.txt'), recursive=True))
    # for f in files:
    #     eval_txt_to_latex(f)
    # eval_txt_to_latex("/home/yuan/data/out/LGLN2023/pre/epoch20/eval.txt")
    # res = summarize_result("/home/yuan/data/out/LGLN")
    # plot(res, "/home/yuan/data/out/LGLN")
    land_use_analysis("/home/yuan/data/out/LGLN/pseudo-0.6/pseudo_12-06-03-57-17/epoch5/holistic_seg_result_")