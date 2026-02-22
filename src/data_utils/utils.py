"""
Core audio VAE utilities for encoding and decoding.

For preprocessing utilities (file scanning, duration filtering, etc.),
see src.data_utils.preprocess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import torchaudio
import tqdm
from audiotools import AudioSignal


@torch.no_grad()
def encode_audio_latents(
    model: torch.nn.Module,
    audio_path_or_signal: Union[str, Path, AudioSignal],
    verbose: bool = False,
    chunked: bool = True,
    chunk_size_latents: int = 512,
    overlap_latents: int = 32,
):
    """
    Chunked VAE encoding for a single file or AudioSignal.

    Always mixes to mono.
    Returns (posterior_params, metadata) where posterior_params = cat([mean, logvar], dim=1).
    """
    audio_signal = audio_path_or_signal
    if isinstance(audio_signal, (str, Path)):
        audio_signal = AudioSignal.load_from_file_with_ffmpeg(str(audio_signal))

    model.eval()
    original_device = audio_signal.device

    audio_signal = audio_signal.clone()
    original_sr = audio_signal.sample_rate
    original_length = audio_signal.signal_length

    # Resample to model SR
    audio_signal.resample(model.sample_rate)

    # Always mix to mono: [B, C, T] -> [B, 1, T]
    audio = audio_signal.audio_data
    if audio.shape[1] > 1:
        audio = audio.mean(dim=1, keepdim=True)

    # Ensure hop-aligned length (pads at end only)
    audio = model._pad(audio)

    B, C, T = audio.shape
    assert C == 1, "expected mono after mixing"

    samples_per_latent = int(model.hop_length)
    total_size = int(T)
    latent_size = total_size // samples_per_latent

    params_dim = int(model.quantizer.in_proj.out_channels)
    posterior_params = torch.zeros(
        (B, params_dim, latent_size),
        dtype=audio.dtype,
        device=original_device,
    )

    device = next(model.parameters()).device
    use_chunking = bool(chunked) and (latent_size > int(chunk_size_latents))

    def _encode_params(x_b1t: torch.Tensor) -> torch.Tensor:
        z = model.encoder(x_b1t)
        mean, scale = model.quantizer.in_proj(z).chunk(2, dim=1)
        stdev = F.softplus(scale) + 1e-4
        logvar = 2.0 * torch.log(stdev)
        return torch.cat([mean, logvar], dim=1)

    if not use_chunking:
        params = _encode_params(audio.to(device))
        posterior_params.copy_(params.to(original_device))
    else:
        chunk_latents = int(chunk_size_latents)
        overlap_lat = int(overlap_latents)

        if chunk_latents <= overlap_lat:
            raise ValueError("chunk_size_latents must be > overlap_latents")
        if overlap_lat < 2:
            raise ValueError("When chunked=True, overlap_latents must be >= 2")
        if overlap_lat % 2 != 0:
            raise ValueError("overlap_latents must be even when chunked=True")

        chunk_size = chunk_latents * samples_per_latent          # samples
        overlap = overlap_lat * samples_per_latent               # samples
        hop = chunk_size - overlap                               # samples
        assert hop > 0

        spans = []
        last_i = None
        it = tqdm.trange(0, max(0, total_size - chunk_size) + 1, hop) if verbose else range(
            0, max(0, total_size - chunk_size) + 1, hop
        )
        for i in it:
            spans.append((int(i), int(i + chunk_size)))
            last_i = int(i)

        # Force last chunk to [-chunk_size:]
        if last_i is None or (last_i + chunk_size != total_size):
            spans.append((max(0, total_size - chunk_size), total_size))

        half_ol = overlap_lat // 2
        last = len(spans) - 1

        for ci, (start, end) in enumerate(spans):
            x = audio[..., start:end].to(device)  # [B, 1, chunk_size_samples]
            params = _encode_params(x)            # [B, params_dim, chunk_latents or edge]

            if ci == last:
                t_end = latent_size
                t_start = t_end - params.shape[-1]
            else:
                t_start = start // samples_per_latent
                t_end = t_start + chunk_latents

            cs, ce = 0, params.shape[-1]
            if ci > 0:
                t_start += half_ol
                cs += half_ol
            if ci < last:
                t_end -= half_ol
                ce -= half_ol

            posterior_params[:, :, t_start:t_end] = params[:, :, cs:ce].to(original_device)

    metadata = {
        "original_length": int(original_length),
        "latent_length": int(latent_size),
        "sample_rate": int(original_sr),
    }
    return posterior_params, metadata


@torch.no_grad()
def decode_audio_latents(
    model: torch.nn.Module,
    latents: torch.Tensor,
    metadata: dict,
    verbose: bool = False,
    resample_to_original: bool = True,
    chunked: bool = False,
    chunk_size_latents: int = 512,
    overlap_latents: int = 32,
) -> torch.Tensor:
    """
    Chunked VAE decoding.

    latents: [B, D, T_lat]
    Returns waveform tensor [B, C, T] (C is whatever model.decode returns; often 1).
    """
    model.eval()
    device = next(model.parameters()).device
    original_device = latents.device

    assert latents.dim() == 3, "expected latents [B, D, T_lat]"
    TL = int(latents.shape[-1])
    hop = int(model.hop_length)

    use_chunking = bool(chunked) and (TL > int(chunk_size_latents))

    if not use_chunking:
        recons = model.decode(latents.to(device)).to(original_device)
    else:
        chunk_lat = int(chunk_size_latents)
        overlap_lat = int(overlap_latents)

        if chunk_lat <= overlap_lat:
            raise ValueError("chunk_size_latents must be > overlap_latents")
        if overlap_lat < 2:
            raise ValueError("When chunked=True, overlap_latents must be >= 2")
        if overlap_lat % 2 != 0:
            raise ValueError("overlap_latents must be even when chunked=True")

        hop_lat = chunk_lat - overlap_lat
        assert hop_lat > 0

        spans = []
        last_i = None
        it = tqdm.trange(0, max(0, TL - chunk_lat) + 1, hop_lat) if verbose else range(
            0, max(0, TL - chunk_lat) + 1, hop_lat
        )
        for i in it:
            spans.append((int(i), int(i + chunk_lat)))
            last_i = int(i)

        # Force last chunk to [-chunk_lat:]
        if last_i is None or (last_i + chunk_lat != TL):
            spans.append((max(0, TL - chunk_lat), TL))

        half_ol = overlap_lat // 2
        last = len(spans) - 1

        y_final = None
        T_out = TL * hop

        for ci, (a, b) in enumerate(spans):
            chunk = latents[:, :, a:b].to(device)
            audio = model.decode(chunk).to(original_device)

            if audio.dim() == 2:
                audio = audio.unsqueeze(1)  # [B, 1, T]

            if y_final is None:
                y_final = torch.zeros(
                    (latents.shape[0], audio.shape[1], T_out),
                    dtype=audio.dtype,
                    device=original_device,
                )

            if ci == last:
                t_end = T_out
                t_start = t_end - audio.shape[-1]
            else:
                t_start = a * hop
                t_end = t_start + chunk_lat * hop

            cs, ce = 0, audio.shape[-1]
            if ci > 0:
                t_start += half_ol * hop
                cs += half_ol * hop
            if ci < last:
                t_end -= half_ol * hop
                ce -= half_ol * hop

            y_final[:, :, t_start:t_end] = audio[:, :, cs:ce]

        recons = y_final

    original_sr = int(metadata.get("sample_rate", model.sample_rate))
    if resample_to_original and (original_sr != int(model.sample_rate)):
        recons_signal = AudioSignal(recons, int(model.sample_rate))
        recons_signal.resample(original_sr)
        recons = recons_signal.audio_data

    if "original_length" in metadata:
        original_length = int(metadata["original_length"])
        if resample_to_original and (original_sr != int(model.sample_rate)):
            target_length = original_length
        elif original_sr != int(model.sample_rate):
            target_length = int(round(original_length * float(model.sample_rate) / float(original_sr)))
        else:
            target_length = original_length
        recons = recons[..., :target_length]

    return recons


def save_audio(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    """
    Save audio tensor to file.

    Args:
        path: Output file path
        wav: Audio tensor of shape (1, 1, T) or (1, T)
        sample_rate: Sample rate
    """
    wav = wav.detach().cpu()
    if wav.dim() == 3:
        if wav.size(0) != 1:
            raise ValueError(f"Expected batch size 1, got {wav.size(0)}")
        wav = wav[0]
    torchaudio.save(path, wav, sample_rate=sample_rate, format="mp3")
