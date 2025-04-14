# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from map_sam.conditional_sam.modeling.prompt_encoder import ExampleEncoder


class MapSAM(nn.Module):
    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder=None,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        if prompt_encoder is None:
            self.prompt_encoder = ExampleEncoder(256)
        else:
            self.prompt_encoder = prompt_encoder
        self.freeze_patch_embeddings()

    def freeze_patch_embeddings(self):
        for param in self.image_encoder.patch_embed.parameters():
            param.requires_grad = False

    def forward(self, img_src, lbl_src, img_tgt):
        image = torch.cat([img_src, img_tgt], dim=0)
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        src_img_embedding = image_embedding[:img_src.shape[0]]
        tgt_img_embedding = image_embedding[img_src.shape[0]:]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                imgs_emb=src_img_embedding,
                masks=lbl_src.unsqueeze(1).float(),
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=tgt_img_embedding,  # (B, 256, 24, 24)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 24, 24)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 0, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 24, 24)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        # original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (input_size[0], input_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return masks
