# -*- coding: utf-8 -*-
# Author: Yunshuang Yuan.

import os
from typing import Optional, Tuple

import numpy as np
import torch
import cv2

from data.utils.vis import overlay_label_on_image
from smolsam.modeling.dataset import MapTransformTest
from smolsam.modeling.mapsam import MapSAM
from smolsam import sam_model_registry
from smolsam.utils.vis import apply_jet_colormap_transparent


class SamplePredictor:
    def __init__(
        self,
        checkpoint: str,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        sam_model = sam_model_registry["vit_b"]()
        mapsam = MapSAM(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder
        ).to(device)
        assert os.path.isfile(checkpoint)
        ## Map model to be loaded to specified single GPU
        checkpoint = torch.load(checkpoint, map_location=device)
        mapsam.load_state_dict(checkpoint["model"])

        mapsam.eval()

        self.model = mapsam
        self.transform = MapTransformTest()

    def preprocess(self):
        src_img, src_mask = self.transform(self.src_img, self.src_label)
        tgt_img, _ = self.transform(self.tgt_img, self.src_label)

        src_img = src_img.unsqueeze(0).to(self.device)
        tgt_img = tgt_img.unsqueeze(0).to(self.device)
        src_mask = src_mask[::4, ::4].unsqueeze(0).to(self.device)
        return src_img, tgt_img, src_mask

    def load_data(self, src_img, src_lbl, tgt_img, label_idx):
        self.src_img = cv2.imread(src_img, cv2.IMREAD_UNCHANGED)
        src_label = cv2.imread(src_lbl, cv2.IMREAD_UNCHANGED)
        if src_label.max() == 255:
            self.src_label = src_label == 255
        else:
            self.src_label = src_label == label_idx
        self.tgt_img = cv2.imread(tgt_img)
        self.label_idx = label_idx

    def vis_pred(self, tgt_mask):
        labeled_src_img = overlay_label_on_image(self.src_img, self.src_label, color=[255, 255, 255])
        labeled_tgt_img = apply_jet_colormap_transparent(self.tgt_img, tgt_mask)

        vis_img = np.concatenate((labeled_src_img, labeled_tgt_img), axis=1)
        cv2.imwrite(f"{os.environ.get('HOME')}/Downloads/pred.png", labeled_tgt_img)
        cv2.imwrite(f"{os.environ.get('HOME')}/Downloads/src.png", self.src_img)
        cv2.imwrite(f"{os.environ.get('HOME')}/Downloads/tgt.png", self.tgt_img)
        cv2.imwrite(f"{os.environ.get('HOME')}/Downloads/src_lbl.png", (self.src_label * 255).astype(np.uint8))
        cv2.imwrite(f"{os.environ.get('HOME')}/Downloads/tgt_pred.png", (tgt_mask * 255).astype(np.uint8))

    @torch.no_grad()
    def predict(
        self,
        src_img: str,
        src_label: str,
        tgt_img: str,
        label_idx: int
    ) -> np.ndarray:
        self.load_data(src_img, src_label, tgt_img, label_idx)
        src_img, tgt_img, src_mask = self.preprocess()
        tgt_mask = self.model(src_img, src_mask, tgt_img).squeeze().sigmoid()
        tgt_mask = tgt_mask.cpu().numpy()

        self.vis_pred(tgt_mask)

        return tgt_mask


if __name__ == "__main__":

    label = 1
    predictor = SamplePredictor("/home/yuan/code/HisMap/smolsam/logs/fewshot/medsam_model_best.pth")
    pred_mask = predictor.predict(
        "/home/yuan/data/HisMap/SMOLdata/fewshot/imgs/c3821_1898_512_512.png",
        "/home/yuan/data/HisMap/SMOLdata/fewshot/lbls/c3821_1898_512_512.png",
        "/home/yuan/data/HisMap/SMOLdata/hameln/imgs/3922_1898_8448_5760.png",
        label
    )

# industrial: 3821_2002_6144_8448, 3821_2002_5376_8064
# sport: 3821_2002_6144_6912, 3821_2002_6144_7296

# road1: 3821_2002_6144_8448, 3821_2002_5376_8064
# road2: 3821_2002_6144_4608, 3821_2002_6144_384