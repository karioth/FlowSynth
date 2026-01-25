from PIL import Image
import numpy as np
import json
import torch
import os
from ..models.modules.vae import AutoencoderKL

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def safe_blob_dump(fn, result):
    try:
        if os.path.exists(fn):
            os.remove(fn)
        with open(fn, "w") as f:
            json.dump(result, f)
    except:
        print('Failed to write blob:', fn, result)

def load_vae(vae_model_path, image_size):
    data = torch.load(vae_model_path, map_location="cpu")
    if "config" in data:
        raise ValueError("Sigma-VAE checkpoints are not supported in this refactor.")
    input_size = image_size // 16
    latent_size = 16
    flatten_input = False
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=vae_model_path)
    return vae, input_size, latent_size, flatten_input


def image_to_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected (B, C, H, W) tensor, got shape {tuple(x.shape)}")
    bsz, channels, height, width = x.shape
    return x.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)


def sequence_to_image(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected (B, T, C) tensor, got shape {tuple(x.shape)}")
    bsz, seq_len, channels = x.shape
    grid_size = int(seq_len ** 0.5)
    if grid_size * grid_size != seq_len:
        raise ValueError(f"seq_len must be a perfect square, got {seq_len}")
    return x.reshape(bsz, grid_size, grid_size, channels).permute(0, 3, 1, 2)

