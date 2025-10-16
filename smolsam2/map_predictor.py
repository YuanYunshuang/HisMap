# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from data.utils.vis import overlay_label_on_image
from smolsam.modeling.dataset import MapTransformTest
from smolsam.modeling.mapsam import MapSAM
from smolsam import sam_model_registry
from smolsam.utils.vis import apply_jet_colormap_transparent


class MapPredictor:
    def __init__(
        self,
        checkpoint: str,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
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

    def set_src(self, src_img, src_lbl, label_idx):
        assert os.path.isfile(src_img), f"{src_img} does not exist."
        src_img = cv2.imread(src_img, cv2.IMREAD_UNCHANGED)
        src_lbl = cv2.imread(src_lbl, cv2.IMREAD_UNCHANGED)
        self.src_img = src_img
        if src_lbl.max() == 255:
            self.src_lbl = src_lbl == 255
        else:
            self.src_lbl = src_lbl == label_idx
        self.label_idx = label_idx

        src_img, src_mask = self.transform(src_img, self.src_lbl)
        src_img = src_img.unsqueeze(0).to(self.device)
        src_mask = src_mask[::4, ::4].unsqueeze(0).to(self.device)
        self.src_emb = self.model.image_encoder(src_img)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            imgs_emb=self.src_emb,
            masks=src_mask.unsqueeze(1).float(),
        )
        self.sparse_embeddings = sparse_embeddings
        self.dense_embeddings = dense_embeddings

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
        tgt_img: np.array,
    ) -> np.ndarray:
        tgt_img = self.transform(tgt_img).unsqueeze(0).to(self.device)
        tgt_img_embedding = self.model.image_encoder(tgt_img)
        low_res_masks, _ = self.model.mask_decoder(
            image_embeddings=tgt_img_embedding,  # (B, 256, 24, 24)
            image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1, 256, 24, 24)
            sparse_prompt_embeddings=self.sparse_embeddings,  # (B, 0, 256)
            dense_prompt_embeddings=self.dense_embeddings,  # (B, 256, 24, 24)
            multimask_output=False,
        )
        tgt_mask = F.interpolate(
            low_res_masks,
            size=(tgt_img.shape[2], tgt_img.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tgt_mask = tgt_mask.squeeze().sigmoid().cpu().numpy()

        # self.vis_pred(tgt_mask)

        return tgt_mask

    @property
    def device(self) -> torch.device:
        return self.model.image_encoder.pos_embed.device


