#!/usr/bin/env python3
"""
Roundtrip WAV files through DACVAE (EqSynth latent path).

Fail-fast: any exception stops execution.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from src.data_utils.utils import decode_audio_latents, encode_audio_latents


DEFAULT_INPUT_DIR = "/share/users/student/f/friverossego/EqSynth/samples_audio/audiocaps_test_gt"
DEFAULT_DACVAE_WEIGHTS = "facebook/dacvae-watermarked"

DAC_ENCODE_CHUNKED = True
DAC_DECODE_CHUNKED = False
DAC_CHUNK_SIZE_LATENTS = 512
DAC_OVERLAP_LATENTS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roundtrip WAV files through DACVAE.")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dacvae-weights", type=str, default=DEFAULT_DACVAE_WEIGHTS)
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_input_wavs(input_dir: Path, limit: int | None) -> list[Path]:
    wavs = sorted(input_dir.glob("*.wav"), key=lambda p: p.name)
    if limit is not None:
        wavs = wavs[:limit]
    if not wavs:
        raise RuntimeError(f"No .wav files found in {input_dir}")
    return wavs


def load_mono_wav(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    wav = wav.to(torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, int(sr)


def to_saveable_audio(audio: torch.Tensor) -> torch.Tensor:
    audio = audio.detach().to(dtype=torch.float32, device="cpu")
    if audio.dim() == 3:
        audio = audio[0]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return torch.clamp(audio, -1.0, 1.0)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen_device = str(device) if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(int(seed))
    return generator


def load_dacvae(weights_path: str, device: torch.device):
    import dacvae

    return dacvae.DACVAE.load(weights_path).eval().to(device)


@torch.no_grad()
def roundtrip_dacvae(
    dac_model,
    wav_cpu: torch.Tensor,
    sample_rate: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    from audiotools import AudioSignal

    signal = AudioSignal(wav_cpu.unsqueeze(0), sample_rate)
    posterior_params, metadata = encode_audio_latents(
        dac_model,
        signal,
        chunked=DAC_ENCODE_CHUNKED,
        chunk_size_latents=DAC_CHUNK_SIZE_LATENTS,
        overlap_latents=DAC_OVERLAP_LATENTS,
    )
    posterior_params = posterior_params.to(device=device, dtype=torch.float32)

    mean, logvar = torch.chunk(posterior_params, 2, dim=1)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.shape, generator=generator, device=std.device, dtype=std.dtype)
    z = mean + std * eps

    decoded = decode_audio_latents(
        dac_model,
        z,
        metadata,
        chunked=DAC_DECODE_CHUNKED,
        chunk_size_latents=DAC_CHUNK_SIZE_LATENTS,
        overlap_latents=DAC_OVERLAP_LATENTS,
    )
    out_sr = int(metadata.get("sample_rate", sample_rate))
    return to_saveable_audio(decoded), out_sr


def run_dacvae_roundtrip(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    limit: int | None,
    seed: int,
    device_arg: str,
    dacvae_weights: str,
    progress_desc: str = "DACVAE roundtrip",
    seed_offset: int = 0,
    seed_step: int = 1,
) -> None:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_input_wavs(input_dir, limit)
    device = choose_device(device_arg)
    set_global_seed(seed)

    print(f"Found {len(files)} wav files in {input_dir}")
    print(f"Using device: {device}")
    print(f"Loading DACVAE from {dacvae_weights} ...")
    dac_model = load_dacvae(dacvae_weights, device)

    for idx, in_path in enumerate(tqdm(files, desc=progress_desc)):
        wav_cpu, sr = load_mono_wav(in_path)
        gen_seed = seed + seed_offset + (idx * seed_step)
        generator = make_generator(device, gen_seed)
        out_audio, out_sr = roundtrip_dacvae(
            dac_model,
            wav_cpu,
            sr,
            device=device,
            generator=generator,
        )
        torchaudio.save(str(output_dir / in_path.name), out_audio, sample_rate=out_sr, format="wav")

    print(f"DAC outputs: {output_dir}")


def main() -> None:
    args = parse_args()
    run_dacvae_roundtrip(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        seed=args.seed,
        device_arg=args.device,
        dacvae_weights=args.dacvae_weights,
    )


if __name__ == "__main__":
    main()
