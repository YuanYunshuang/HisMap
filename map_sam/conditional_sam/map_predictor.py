# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from torchvision.transforms import v2

from map_sam.conditional_sam.modeling import MapSAM

from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide


class MapPredictor:
    def __init__(
        self,
        mapsam_model: MapSAM,
        img_size: int = 384,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = mapsam_model
        self.img_size = img_size
        self.reset_image()
        color_jitter = 0.05

        self.img_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            # v2.ColorJitter(color_jitter, color_jitter, color_jitter),
        ])

        self.lbl_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=False),
        ])

    def image_to_tensor(self, image):
        pad_x = self.img_size - image.shape[0]
        pad_y = self.img_size - image.shape[1]
        if pad_x > 0 or pad_y > 0:
            image = np.pad(image, ((0, pad_x), (0, pad_y), (0, 0)), mode="constant", constant_values=0)

        # input_image_torch = torch.as_tensor(image, device=self.device, dtype=torch.float).permute(2, 0, 1).contiguous()
        # input_image_torch = tvf.normalize(input_image_torch / 255, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        input_image_torch = self.img_transforms(image)
        input_image_torch = input_image_torch[None, :, :, :]
        return input_image_torch.to(self.device)

    def label_to_tensor(self, label):
        # input_label_torch = torch.as_tensor(label, device=self.device, dtype=torch.float)
        input_label_torch = self.lbl_transforms(label)
        input_label_torch = input_label_torch[None, :, :]
        return input_label_torch.to(self.device)

    @torch.no_grad()
    def set_src_example(
        self,
        image: np.ndarray,
        label: np.ndarray[bool]
    ) -> None:
        """
        Calculates the example embeddings for the provided image-label pair, allowing
        the new given target image to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The example image for showing which pattern should be
            segmented.
          label (np.ndarray): The binary label of the example image.
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        if image.ndim == 2:
            image = image[..., None].repeat(3, axis=2)
        assert label.ndim == 2 and label.dtype == bool, "label must be 2-dimensional and bool."

        input_image = self.image_to_tensor(image)
        downsample_factor = label.shape[0] // (image.shape[0] // 4)
        input_label = self.label_to_tensor(label[::downsample_factor, ::downsample_factor])

        self.reset_image()
        src_img_emb = self.model.image_encoder(input_image)
        self.prompt_emb_sparse, self.prompt_emb_dense = self.model.prompt_encoder(src_img_emb, input_label)
        self.src_example_set = True

    def predict(
        self,
        target_image: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          target_image (np.ndarray): The target image to predict masks for.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.src_example_set:
            raise RuntimeError(
                "An source image-label example must be set with .set_image(...) before mask prediction."
            )

        target_image_tensor = self.image_to_tensor(target_image)
        target_image_emb = self.model.image_encoder(target_image_tensor)
        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=target_image_emb,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=self.prompt_emb_sparse,
            dense_prompt_embeddings=self.prompt_emb_dense,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, (self.img_size, self.img_size))

        if not return_logits:
            masks = masks.sigmoid()

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()

        if target_image.shape[0] < self.img_size or target_image.shape[1] < self.img_size:
            masks_np = masks_np[:, :target_image.shape[0], :target_image.shape[1]]
            low_res_masks_np = low_res_masks_np[:, :target_image.shape[0] // 4, :target_image.shape[1] // 4]
        return masks_np, iou_predictions_np, low_res_masks_np

    @property
    def device(self) -> torch.device:
        return self.model.image_encoder.pos_embed.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
