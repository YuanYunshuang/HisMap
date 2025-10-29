# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from syntra.modeling.prompt_encoders.example_encoder import ExampleEncoder


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

    def save_dora_parameters(self) -> dict:
        state_dict = self.state_dict()
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if 'prompt_encoder' in key or 'mask_decoder' in key:
                filtered_state_dict[key] = value
            elif ('qkv' in key) and ('m_' in key or 'lora' in key):
                    filtered_state_dict[key] = value
        return filtered_state_dict

    def load_dora_parameters(self, state_dict: dict) -> None:
        full_state_dict = self.state_dict()
        for key, value in state_dict.items():
            full_state_dict[key] = value
        self.load_state_dict(full_state_dict)

    def forward(self, img_src, lbl_src, img_tgt):
        image = torch.cat([img_tgt.unsqueeze(1), img_src], dim=1) # (B, T, 3, H, W)
        B, T, _, H, W = image.shape
        N = lbl_src.shape[2]
        image_embedding = self.image_encoder(image.flatten(0, 1))  # (B*T, 256, 64, 64)
        image_embedding = image_embedding.view(B, T, *image_embedding.shape[1:])  # (B, T, 256, 64, 64)
        tgt_img_embedding = image_embedding[:, :1].repeat(1, (T-1)*N, 1, 1, 1) # (B, (T-1)*N, 256, 64, 64)
        src_img_embedding = image_embedding[:, 1:].unsqueeze(2).repeat(1, 1, N, 1, 1, 1) # (B, T-1, N, 256, 64, 64)
        # do not compute gradients for prompt encoder
        # with torch.no_grad():
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            imgs_emb=src_img_embedding.flatten(0, 2),  # (B*(T-1)*N, 256, 64, 64)
            masks=lbl_src.flatten(0, 2).unsqueeze(1).float(), # (B*(T-1)*N, 1, H, W
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=tgt_img_embedding.flatten(0, 1),  # (B, (T-1)*N, 256, 64, 64) -> (B*(T-1)*N, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 24, 24)
            sparse_prompt_embeddings=sparse_embeddings,  # (B*(T-1)*N, N_sparse, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B*(T-1)*N, 256, 64, 64)
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(
            low_res_masks,  # (B*(T-1)*N, 1, 96, 96)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).view(B, T-1, N, 1, H, W)  # (B, N, 1, H, W)

        src_lbl_mask = lbl_src.flatten(3).any(dim=3).view(B, T-1, N, 1, 1, 1)  # (B, T-1, N)
        neg = torch.log(torch.tensor(1e-4, device=ori_res_masks.device))
        # set the masks to a very low value where there is no source label
        x, y, z = torch.where(src_lbl_mask == False)[:3]
        ori_res_masks[x, y, z] = neg # (B, T-1, N, H, W)
        ori_res_masks = ori_res_masks.max(dim=1).values  # (B, N, 1, H, W)
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
