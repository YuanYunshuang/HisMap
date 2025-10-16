from PIL import Image
import numpy as np

def convert():
    img = Image.open("smolsam/logs/siegfried_vineyard_e200_r16_a4_fg0.1/test_siegfried_vineyard_2/inf_plots/siegfried.vineyard_540_tile_448_4032.png")
    h, w = img.size
    img = np.array(img)
    pred_prob = img[:, (w+10)*3:(w+10)*3+w]
    pred = (pred_prob > 127).astype(np.uint8) * 255
    new_img = np.zeros((w, (w+10)*6, 3), dtype=np.uint8) + 128
    new_img[:, :(w+10)*4] = img[:, :(w+10)*4]
    new_img[:, (w+10)*4:(w+10)*4+w] = pred
    new_img[:, (w+10)*5:(w+10)*5+w] = img[:, (w+10)*4:]

    new_img = Image.fromarray(new_img)
    new_img.save("/home/yuan/Downloads/vineyard_pred.png")


def attach_mapsam_res():
    img1 = Image.open("/home/yuan/Downloads/rail_pred.png")
    img2_pred = Image.open("/home/yuan/Downloads/10_sie_pred.png").convert("RGB")
    img2_pred_p = Image.open("/home/yuan/Downloads/10_sie_pred_p.png").convert("RGB")

    # convert to numpy
    img1 = np.array(img1)
    img2_pred = np.array(img2_pred)
    img2_pred_p = np.array(img2_pred_p)

    # concat with 10 pixel padding of color gray 128, all images to have the same height
    h1, w1, _ = img1.shape
    h2, w2, _ = img2_pred_p.shape
    h3, w3, _ = img2_pred.shape
    h = max(h1, h2, h3)
    new_img = np.zeros((h, w1 + w2 + w3 + 10, 3), dtype=np.uint8) + 128
    new_img[:h1, :w1] = img1
    new_img[:h2, w1 :w1+ w2] = img2_pred_p
    new_img[:h3, w1 + 10 + w2:w1 + 10 + w2 + w3] = img2_pred
    new_img = Image.fromarray(new_img)

    new_img.save("/home/yuan/Downloads/railway_pred_compapre.png")


def cal_cls_ratios():
    ratios = np.array([198.4979+98.0835, 218.8876+136.5137, 18.4875+5.8441])
    ratios = ratios / ratios.sum()
    print(','.join([f'{r:.2f}' for r in ratios]))

cal_cls_ratios()