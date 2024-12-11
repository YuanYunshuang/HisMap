import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from openai import files
from sympy.abc import alpha


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
    labels = ['WL', 'GL', 'SM', 'FL', 'SW', 'mIoU', 'OA']

    for year, v in res.items():
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        v = np.array(v)
        for i, label in enumerate(labels):
            if label == 'mIoU' or label == 'OA':
                alpha = 1
                linewidth = 4
                linestyle = '-'
            else:
                alpha = 0.5
                linewidth = 2
                linestyle = '--'

            ax.plot(v[:, 0], v[:, i+1], label=label, alpha=alpha, linewidth=linewidth, linestyle=linestyle)

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'bi_pseudo_unc_{year}.pdf'))


if __name__ == '__main__':
    # files = sorted(glob.glob(os.path.join("/home/yuan/data/out/LGLN",
    #                                       '**/eval.txt'), recursive=True))
    # for f in files:
    #     eval_txt_to_latex(f)
    res = summarize_result("/home/yuan/data/out/LGLN")
    plot(res, "/home/yuan/data/out/LGLN")