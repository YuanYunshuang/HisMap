import os.path
from glob import glob
import cv2

from src.tools.map_analyzer import read_tiff


def rescale(in_dir):
    out_dir = in_dir + '_thumbnail'
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([x for x in os.listdir(in_dir) if 'png' in x or x.endswith('tif')])

    for file in files:
        print(file)
        if 'tif' in file:
            img = read_tiff(os.path.join(in_dir, file))
        else:
            img = cv2.imread(os.path.join(in_dir, file))
        img = cv2.resize(img, (0, 0), fx = 0.05, fy = 0.05)
        cv2.imwrite(os.path.join(out_dir, file), img)


if __name__=="__main__":
    rescale("/home/yuan/data/HisMap/Hameln/TK_projektiv_entzerrt_1m")