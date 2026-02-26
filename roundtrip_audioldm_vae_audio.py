#!/usr/bin/env python3
"""
Roundtrip WAV files through AudioLDM2 VAE + vocoder (16 kHz mel path).

Fail-fast: any exception stops execution.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


DEFAULT_INPUT_DIR = "/share/users/student/f/friverossego/EqSynth/samples_audio/audiocaps_test_gt"
DEFAULT_AUDIOLDM_MODEL_ID = "cvssp/audioldm2"

AUDIO_LDM_SAMPLE_RATE = 16000
TACOTRON_STFT_CONFIG = (1024, 160, 1024, 64, 16000, 0, 8000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roundtrip WAV files through AudioLDM2 VAE + vocoder.")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--audioldm-model-id", type=str, default=DEFAULT_AUDIOLDM_MODEL_ID)
    parser.add_argument("--audioldm-local-files-only", action="store_true")
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


def _audioldm_log_compression(x: torch.Tensor) -> torch.Tensor:
    # Match AudioLDM training dynamic-range compression behavior.
    return torch.log(torch.clamp(x, min=1e-5))


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen_device = str(device) if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(int(seed))
    return generator


def module_device(module, fallback: torch.device) -> torch.device:
    for p in module.parameters():
        return p.device
    for b in module.buffers():
        return b.device
    return fallback


def align_tacotron_stft_device(stft, device: torch.device):
    if hasattr(stft, "to"):
        stft = stft.to(device)

    for attr in ("mel_basis", "window"):
        value = getattr(stft, attr, None)
        if isinstance(value, torch.Tensor):
            setattr(stft, attr, value.to(device))

    stft_fn = getattr(stft, "stft_fn", None)
    if stft_fn is not None:
        if hasattr(stft_fn, "to"):
            stft_fn = stft_fn.to(device)
        for attr in ("forward_basis", "inverse_basis", "window"):
            value = getattr(stft_fn, attr, None)
            if isinstance(value, torch.Tensor):
                setattr(stft_fn, attr, value.to(device))
    return stft


def load_audioldm_components(model_id: str, device: torch.device, local_files_only: bool):
    from diffusers import AutoencoderKL
    from transformers import SpeechT5HifiGan

    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    ).eval().to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(
        model_id,
        subfolder="vocoder",
        local_files_only=local_files_only,
    ).eval().to(device)
    return vae, vocoder


def build_tacotron_stft():
    from audioldm_eval import audio as Audio

    stft = Audio.TacotronSTFT(*TACOTRON_STFT_CONFIG)
    # audioldm_eval STFT forces CPU inside transform(), so keep it on CPU.
    return align_tacotron_stft_device(stft, torch.device("cpu"))


@torch.no_grad()
def roundtrip_audioldm_vae(
    vae,
    vocoder,
    stft,
    wav_cpu: torch.Tensor,
    sample_rate: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    vae_dev = module_device(vae, device)
    vocoder_dev = module_device(vocoder, device)
    stft = align_tacotron_stft_device(stft, torch.device("cpu"))

    wav_16k = wav_cpu
    if sample_rate != AUDIO_LDM_SAMPLE_RATE:
        wav_16k = torchaudio.functional.resample(
            wav_16k,
            orig_freq=sample_rate,
            new_freq=AUDIO_LDM_SAMPLE_RATE,
        )
    target_len = int(wav_16k.shape[-1])

    wav_16k = wav_16k - wav_16k.mean()
    peak = torch.clamp(wav_16k.abs().max(), min=1e-8)
    wav_16k = (wav_16k / peak) * 0.5
    wav_16k = wav_16k.to(device="cpu", dtype=torch.float32)

    mel, _ = stft.mel_spectrogram(wav_16k, normalize_fun=_audioldm_log_compression)
    mel_img = mel.transpose(1, 2).unsqueeze(1).to(device=vae_dev, dtype=torch.float32)

    posterior = vae.encode(mel_img).latent_dist
    mean = posterior.mean
    logvar = posterior.logvar
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.shape, generator=generator, device=std.device, dtype=std.dtype)
    z = mean + std * eps

    mel_rec = vae.decode(z).sample
    mel_for_vocoder = mel_rec.squeeze(1).to(device=vocoder_dev, dtype=torch.float32)

    wav_rec = vocoder(mel_for_vocoder)
    if wav_rec.dim() == 1:
        wav_rec = wav_rec.unsqueeze(0)
    if wav_rec.dim() == 3:
        wav_rec = wav_rec[0]

    if wav_rec.shape[-1] < target_len:
        wav_rec = F.pad(wav_rec, (0, target_len - wav_rec.shape[-1]))
    elif wav_rec.shape[-1] > target_len:
        wav_rec = wav_rec[..., :target_len]

    return to_saveable_audio(wav_rec), AUDIO_LDM_SAMPLE_RATE


def run_audioldm_vae_roundtrip(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    limit: int | None,
    seed: int,
    device_arg: str,
    audioldm_model_id: str,
    local_files_only: bool,
    progress_desc: str = "AudioLDM VAE roundtrip",
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
    print(f"Loading AudioLDM2 VAE+vocoder from {audioldm_model_id} ...")
    vae, vocoder = load_audioldm_components(
        audioldm_model_id,
        device=device,
        local_files_only=local_files_only,
    )
    stft = build_tacotron_stft()

    for idx, in_path in enumerate(tqdm(files, desc=progress_desc)):
        wav_cpu, sr = load_mono_wav(in_path)
        gen_seed = seed + seed_offset + (idx * seed_step)
        generator = make_generator(device, gen_seed)
        out_audio, out_sr = roundtrip_audioldm_vae(
            vae,
            vocoder,
            stft,
            wav_cpu,
            sr,
            device=device,
            generator=generator,
        )
        torchaudio.save(str(output_dir / in_path.name), out_audio, sample_rate=out_sr, format="wav")

    print(f"AudioLDM outputs: {output_dir}")


def main() -> None:
    args = parse_args()
    run_audioldm_vae_roundtrip(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        seed=args.seed,
        device_arg=args.device,
        audioldm_model_id=args.audioldm_model_id,
        local_files_only=args.audioldm_local_files_only,
    )


if __name__ == "__main__":
    main()
