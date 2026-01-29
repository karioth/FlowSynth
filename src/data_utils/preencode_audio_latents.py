import os
from pathlib import Path

import torch

from audiotools import AudioSignal


@torch.no_grad()
def cache_audio_latents(
    data_iter,
    model,
    *,
    chunk_size_latents: int,
    overlap_latents: int,
    rank: int = 0,
):
    """
    data_iter yields batches (lists) where each item is either None or:
    (source_str, wav[C,T], sr, out_path_str).
    """
    from src.data_utils.audio_utils import encode_audio_latents

    for batch in data_iter:
        for item in batch:
            if item is None:
                continue

            path_str, wav, sr, out_path_str = item
            out_path = Path(out_path_str)

            if out_path.exists():
                continue

            audio = AudioSignal(wav.unsqueeze(0), sr)
            posterior_params, metadata = encode_audio_latents(
                model,
                audio,
                chunked=True,
                chunk_size_latents=chunk_size_latents,
                overlap_latents=overlap_latents,
            )

            tensor = posterior_params[0].to(torch.float32).cpu()
            latent_length = int(metadata.get("latent_length", tensor.shape[-1]))
            payload = {
                "posterior_params": tensor,
                "latent_length": latent_length,
                "source": path_str,
            }

            os.makedirs(out_path.parent, exist_ok=True)
            tmp = str(out_path) + f".{rank}.{os.getpid()}.tmp"
            try:
                torch.save(payload, tmp)
                os.replace(tmp, out_path)
            finally:
                try:
                    os.remove(tmp)
                except FileNotFoundError:
                    pass
