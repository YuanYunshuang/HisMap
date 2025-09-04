import os

import cv2
import numpy as np

from data.utils.imgio import read_tiff
from data.utils.vis import overlay_label_on_image
from data.bayern import COLORS, CLASSES


def process_bayern_data(indir, mapfile, labelfile):
    """
    How to use:
        indir: data root directory with 'lbls' and 'maps' subdirectories.
        mapfile: map file name. Example: '01_489_N_1910'
        labelfile: label file name. Example: '489_1910'
    Processed:
        maps: ['01_489_N_1910', '01_489_N_1939', '01_490_N_1910', '01_490_N_1938', '01_518_F_1941', '01_519_N_1910', '02_7230_N_1959_E0', '02_7330_N_1959']
        labels: ['489_1910', '489_1933-39', '490_1910', '490_1933-39', '518_1933-39', '519_1910', '7230_1959', '7330_1959']
    """
    outdir = indir.replace('original', 'processed')
    MAP = read_tiff(os.path.join(indir, 'maps', mapfile + '.tif'))

    # get test area info
    ltest = read_tiff(os.path.join(indir, 'lbls', labelfile + '_testsite.tif'))
    valid_index = np.where(ltest == 1)
    x_min, x_max = np.min(valid_index[0]), np.max(valid_index[0])
    y_min, y_max = np.min(valid_index[1]), np.max(valid_index[1])

    # crop test area for the map
    MAP = MAP[x_min:x_max, y_min:y_max]

    # get labels
    LABEL = np.zeros(MAP.shape[:2], dtype=np.uint8)
    for i, c in enumerate(CLASSES):
        lc = read_tiff(os.path.join(indir, 'lbls', f'{labelfile}_{c}.tif'))
        if lc is None:
            continue
        lc = lc[x_min:x_max, y_min:y_max]
        mask = lc == 1
        print(f"Find {np.sum(mask)} pixels in label {c}")
        LABEL[mask] = i + 1

    # visualize to check correctness
    if MAP.ndim == 2:
        overlay = cv2.cvtColor(MAP, cv2.COLOR_GRAY2RGB)
    else:
        overlay = np.copy(MAP)
    overlay = overlay_label_on_image(overlay, LABEL, color=COLORS)

    cv2.imwrite(f"{outdir}/vis/{mapfile}_overlay.png", overlay[..., ::-1])

    # save cropped map and label
    cv2.imwrite(f"{outdir}/maps/{mapfile}.png", MAP)
    cv2.imwrite(f"{outdir}/lbls/{mapfile}.png", LABEL)


if __name__ == '__main__':
    process_bayern_data("/home/yuan/data/HisMap/Bayern/original", '01_489_N_1910',  '489_1910')