
import torch

from utils.dora import DoRAWrapper
from . import (
    MaskDecoder,
    ExampleEncoder,
    TwoWayTransformer,
)



def freeze_model(model: torch.nn.Module):
    """Freeze all parameters in the model."""
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_img_encoder(model_type: str, out_chans: int = 256, pretrained: bool = True, freeze: bool = True, dora_r: int = 0, **kwargs):
    """Build SAM or SAM2 image encoder from model type."""
    model_types = ["vit_h", "vit_l", "vit_b", "hiera_t", "hiera_s", "hiera_l", "hiera_bp"]
    assert model_type in model_types, f"Unknown model type: {model_type}"
    if "vit" in model_type:
        from backbones.sam.build import build_sam_img_encoder as build_img_encoder
        img_encoder = build_img_encoder(model_type, pretrained=pretrained, out_chans=out_chans, **kwargs)
        if freeze or dora_r > 0:
            img_encoder = freeze_model(img_encoder)
        if dora_r > 0:
            img_encoder = DoRAWrapper(img_encoder, r=dora_r)
    elif "hiera" in model_type:
        from backbones.sam2.build import build_sam2_img_encoder as build_img_encoder
        img_encoder = build_img_encoder(model_type, pretrained=pretrained, out_chans=out_chans, **kwargs)
        if freeze or dora_r > 0:
            img_encoder = freeze_model(img_encoder)
        if dora_r > 0:
            img_encoder.trunk = DoRAWrapper(img_encoder.trunk, r=dora_r)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Built image encoder: {model_type}, pretrained={pretrained}")
    return img_encoder


def build_prompt_encoder(model_type: str, embed_dim: int = 256, **kwargs):
    model_types = ["example", "notion"]
    assert model_type in model_types, f"Unknown model type: {model_type}"

    if model_type == "example":
        return ExampleEncoder(embed_dim=embed_dim, **kwargs)


def build_mask_decoder(model_type: str = "default", embed_dim: int = 256, **kwargs):
    model_types = ["default"]
    assert model_type in model_types, f"Unknown model type: {model_type}"
    mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
    return mask_decoder


def build_segmentation_model(model_name: str, **kwargs):
    if model_name == "unet":
        raise NotImplementedError("UNet model is not implemented yet.")
    elif model_name == "smol":
        img_encoder = build_img_encoder("vit_b", **kwargs)
        prompt_encoder = build_prompt_encoder("example", **kwargs)
        mask_decoder = build_mask_decoder("default", **kwargs)
        from .models.smol import SMOLModel
        model = SMOLModel(
            image_encoder=img_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
        )
    elif model_name == "smol2":
        pass
    elif model_name == "syntra":
        pass
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    


    

model = build_segmentation_model('smol', pretrained=True, freeze=True, dora_r=16)
print(model)