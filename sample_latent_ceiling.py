#!/usr/bin/env python3
"""
Generate latent-ceiling audio for AudioCaps evaluation.

For each valid AudioCaps test row:
- Resolve GT audio from youtube_id in --gt-dir.
- Group rows by youtube_id.
- Encode each GT audio once in the selected VAE/VQ backend.
- For continuous VAEs, draw N posterior samples (default: 5), mapped deterministically to
  sorted audiocap_id rows.
- For deterministic VQ codecs, reuse the same reconstruction for each row in the group.
- Save outputs as {audiocap_id}.wav so evaluate.py can run unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm.auto import tqdm

from src.data_utils.utils import decode_audio_latents, encode_audio_latents


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS_CSV = PROJECT_ROOT / "audiocaps-test.csv"
DEFAULT_GT_DIR = PROJECT_ROOT / "audio_samples" / "audiocaps_test_gt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "samples_audio_vae_ceilings"

DEFAULT_DACVAE_WEIGHTS = "facebook/dacvae-watermarked"
DEFAULT_AUDIOLDM_MODEL_ID = "cvssp/audioldm2"
DEFAULT_STABLE_AUDIO_MODEL_ID = "stabilityai/stable-audio-open-1.0"
DEFAULT_AUDIOGEN_MODEL_ID = "facebook/audiogen-medium"

REQUIRED_COLUMNS = ("audiocap_id", "youtube_id", "caption")

DAC_ENCODE_CHUNKED = True
DAC_DECODE_CHUNKED = False
DAC_CHUNK_SIZE_LATENTS = 512
DAC_OVERLAP_LATENTS = 32

AUDIO_LDM_SAMPLE_RATE = 16000
TACOTRON_STFT_CONFIG = (1024, 160, 1024, 64, 16000, 0, 8000)


@dataclass(frozen=True)
class PromptRow:
    audiocap_id: str
    youtube_id: str
    caption: str
    gt_path: Path


@dataclass(frozen=True)
class PromptStats:
    total_rows: int
    valid_rows: int
    missing_ids: int
    missing_caption: int
    missing_gt: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latent-ceiling AudioCaps samples for evaluation.")
    parser.add_argument(
        "--prompts-csv",
        type=Path,
        default=DEFAULT_PROMPTS_CSV,
        help=f"AudioCaps CSV with {REQUIRED_COLUMNS} (default: {DEFAULT_PROMPTS_CSV}).",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=DEFAULT_GT_DIR,
        help=f"Directory containing GT AudioCaps wavs (default: {DEFAULT_GT_DIR}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root output dir (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="both",
        choices=("dacvae", "audioldm2", "stableaudio", "audiogen", "both"),
        help="Which backend(s) to generate. 'both' is kept for backward compatibility and runs all backends.",
    )
    parser.add_argument(
        "--samples-per-audio",
        type=int,
        default=5,
        help="Number of outputs per GT audio/youtube_id group.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto|cpu|cuda|cuda:0 ...",
    )
    parser.add_argument(
        "--dacvae-weights",
        type=str,
        default=DEFAULT_DACVAE_WEIGHTS,
        help=f"DACVAE weights ref/path (default: {DEFAULT_DACVAE_WEIGHTS}).",
    )
    parser.add_argument(
        "--audioldm-model-id",
        type=str,
        default=DEFAULT_AUDIOLDM_MODEL_ID,
        help=f"AudioLDM2 model id (default: {DEFAULT_AUDIOLDM_MODEL_ID}).",
    )
    parser.add_argument(
        "--audioldm-local-files-only",
        action="store_true",
        help="Load AudioLDM2 components with local_files_only=True.",
    )
    parser.add_argument(
        "--stable-audio-model-id",
        type=str,
        default=DEFAULT_STABLE_AUDIO_MODEL_ID,
        help=f"Stable Audio model id (default: {DEFAULT_STABLE_AUDIO_MODEL_ID}).",
    )
    parser.add_argument(
        "--stable-audio-local-files-only",
        action="store_true",
        help="Load Stable Audio components with local_files_only=True.",
    )
    parser.add_argument(
        "--audiogen-model-id",
        type=str,
        default=DEFAULT_AUDIOGEN_MODEL_ID,
        help=(
            "AudioGen model id used to load the same EnCodec/VQ compression model as sample_audiogen.py "
            f"(default: {DEFAULT_AUDIOGEN_MODEL_ID})."
        ),
    )
    parser.add_argument(
        "--audiogen-num-codebooks",
        type=int,
        nargs="+",
        default=None,
        help=(
            "AudioGen codebook counts to evaluate. Default: run all available counts for the selected model "
            "(e.g. 1 2 3 4 for facebook/audiogen-medium)."
        ),
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--limit-youtube",
        type=int,
        default=None,
        help="Optional smoke limit: number of youtube_id groups to process.",
    )
    limit_group.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help=(
            "Optional smoke limit: number of rows to process. "
            "Must be divisible by --samples-per-audio."
        ),
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"CUDA device requested ({device_arg}) but CUDA is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise SystemExit(
                f"Invalid CUDA device index {device.index}; found {torch.cuda.device_count()} CUDA device(s)."
            )
    return device


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _audiocap_sort_key(audiocap_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(audiocap_id))
    except ValueError:
        return (1, audiocap_id)


def _resolve_gt_path(gt_dir: Path, youtube_id: str) -> Path | None:
    direct = gt_dir / f"{youtube_id}.wav"
    prefixed = gt_dir / f"Y{youtube_id}.wav"
    if direct.is_file():
        return direct
    if prefixed.is_file():
        return prefixed
    return None


def _build_run_tag(samples_per_audio: int, seed: int, limit_youtube: int | None, limit_rows: int | None) -> str:
    tag = f"s{samples_per_audio}_seed{seed}"
    if limit_youtube is not None:
        tag += f"_limy{limit_youtube}"
    if limit_rows is not None:
        tag += f"_limr{limit_rows}"
    return tag


def _collect_prompt_groups(
    prompts_csv: Path,
    gt_dir: Path,
    samples_per_audio: int,
    limit_youtube: int | None,
    limit_rows: int | None,
) -> tuple[list[list[PromptRow]], PromptStats]:
    if not prompts_csv.is_file():
        raise SystemExit(f"Prompts CSV not found: {prompts_csv}")
    if not gt_dir.is_dir():
        raise SystemExit(f"GT directory not found: {gt_dir}")

    rows_by_youtube: OrderedDict[str, list[PromptRow]] = OrderedDict()
    seen_audiocap_ids: set[str] = set()
    total_rows = 0
    missing_ids = 0
    missing_caption = 0
    missing_gt = 0

    with prompts_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise SystemExit(f"Prompt CSV has no header row: {prompts_csv}")
        missing_columns = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing_columns:
            raise SystemExit(
                f"Prompt CSV missing required column(s): {missing_columns}. Expected: {list(REQUIRED_COLUMNS)}"
            )

        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            audiocap_id = (row.get("audiocap_id") or "").strip()
            youtube_id = (row.get("youtube_id") or "").strip()
            caption = (row.get("caption") or "").strip()

            if not audiocap_id or not youtube_id:
                missing_ids += 1
                continue
            if not caption:
                missing_caption += 1
                continue

            if audiocap_id in seen_audiocap_ids:
                raise SystemExit(f"Duplicate audiocap_id '{audiocap_id}' at CSV row {row_number}.")
            seen_audiocap_ids.add(audiocap_id)

            gt_path = _resolve_gt_path(gt_dir, youtube_id)
            if gt_path is None:
                missing_gt += 1
                continue

            rows_by_youtube.setdefault(youtube_id, []).append(
                PromptRow(
                    audiocap_id=audiocap_id,
                    youtube_id=youtube_id,
                    caption=caption,
                    gt_path=gt_path,
                )
            )

    grouped_rows: list[list[PromptRow]] = []
    group_size_errors: list[tuple[str, int]] = []
    for youtube_id, rows in rows_by_youtube.items():
        rows_sorted = sorted(rows, key=lambda x: _audiocap_sort_key(x.audiocap_id))
        if len(rows_sorted) != samples_per_audio:
            group_size_errors.append((youtube_id, len(rows_sorted)))
        grouped_rows.append(rows_sorted)

    if group_size_errors:
        examples = ", ".join(f"{youtube_id}:{size}" for youtube_id, size in group_size_errors[:12])
        raise SystemExit(
            f"Expected exactly {samples_per_audio} rows per youtube_id. "
            f"Found {len(group_size_errors)} mismatched group(s), e.g. {examples}"
        )

    if limit_youtube is not None:
        if limit_youtube < 1:
            raise SystemExit("--limit-youtube must be >= 1.")
        grouped_rows = grouped_rows[:limit_youtube]
    elif limit_rows is not None:
        if limit_rows < 1:
            raise SystemExit("--limit-rows must be >= 1.")
        if limit_rows % samples_per_audio != 0:
            raise SystemExit(
                "--limit-rows must be divisible by --samples-per-audio "
                f"({samples_per_audio}) so groups remain complete."
            )
        grouped_rows = grouped_rows[: (limit_rows // samples_per_audio)]

    if len(grouped_rows) == 0:
        raise SystemExit("No valid youtube groups selected for generation.")

    valid_rows = sum(len(rows) for rows in rows_by_youtube.values())
    stats = PromptStats(
        total_rows=total_rows,
        valid_rows=valid_rows,
        missing_ids=missing_ids,
        missing_caption=missing_caption,
        missing_gt=missing_gt,
    )
    return grouped_rows, stats


def _collect_target_filenames(grouped_rows: list[list[PromptRow]]) -> set[str]:
    names: set[str] = set()
    for rows in grouped_rows:
        for row in rows:
            filename = f"{row.audiocap_id}.wav"
            if filename in names:
                raise SystemExit(f"Duplicate output filename target detected: {filename}")
            names.add(filename)
    return names


def _resolve_audiogen_num_codebooks(requested: list[int] | None, total_codebooks: int) -> list[int]:
    if total_codebooks < 1:
        raise SystemExit(f"AudioGen codec reports invalid total_codebooks={total_codebooks}.")

    if requested is None or len(requested) == 0:
        return list(range(1, total_codebooks + 1))

    resolved: list[int] = []
    seen: set[int] = set()
    for n_q in requested:
        if n_q < 1 or n_q > total_codebooks:
            raise SystemExit(
                f"Invalid AudioGen codebook count {n_q}. Allowed range: [1, {total_codebooks}]"
            )
        if n_q not in seen:
            seen.add(n_q)
            resolved.append(int(n_q))
    return resolved


def _audiogen_backend_name(num_codebooks: int, total_codebooks: int) -> str:
    if num_codebooks == total_codebooks:
        return "audiogen"
    return f"audiogen_nq{num_codebooks}"


def _validate_output_dir_state(output_dir: Path, target_filenames: set[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in output_dir.glob("*.wav")}
    unexpected = existing.difference(target_filenames)
    if unexpected:
        sample_preview = ", ".join(sorted(list(unexpected))[:8])
        raise SystemExit(
            f"Output directory {output_dir} contains {len(unexpected)} unexpected wav file(s) "
            f"not part of this run (e.g. {sample_preview}). Use a clean output path."
        )


def _load_mono_wav(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    wav = wav.to(dtype=torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, int(sr)


def _to_saveable_audio(audio: torch.Tensor) -> torch.Tensor:
    audio = audio.detach().to(device="cpu", dtype=torch.float32)
    if audio.dim() == 3:
        if audio.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {tuple(audio.shape)}")
        audio = audio[0]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2:
        raise ValueError(f"Expected audio rank 2 [C, T], got shape {tuple(audio.shape)}")
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return torch.clamp(audio, -1.0, 1.0)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen_device = str(device) if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(int(seed))
    return generator


def _module_device(module, fallback: torch.device) -> torch.device:
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return fallback


def _align_tacotron_stft_device(stft, device: torch.device):
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


def _audioldm_log_compression(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=1e-5))


def _load_audioldm_components(model_id: str, device: torch.device, local_files_only: bool):
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


def _build_tacotron_stft():
    from audioldm_eval import audio as Audio

    stft = Audio.TacotronSTFT(*TACOTRON_STFT_CONFIG)
    return _align_tacotron_stft_device(stft, torch.device("cpu"))


@torch.no_grad()
def _generate_dacvae(
    grouped_rows: list[list[PromptRow]],
    output_dir: Path,
    samples_per_audio: int,
    seed: int,
    device: torch.device,
    dacvae_weights: str,
) -> int:
    import dacvae
    from audiotools import AudioSignal

    print(f"[dacvae] Loading model: {dacvae_weights}")
    model = dacvae.DACVAE.load(dacvae_weights).eval().to(device)

    generated = 0
    for group_index, rows in enumerate(tqdm(grouped_rows, desc="DACVAE latent ceiling", unit="group")):
        gt_path = rows[0].gt_path
        wav_cpu, sample_rate = _load_mono_wav(gt_path)
        signal = AudioSignal(wav_cpu.unsqueeze(0), sample_rate)

        posterior_params, metadata = encode_audio_latents(
            model,
            signal,
            chunked=DAC_ENCODE_CHUNKED,
            chunk_size_latents=DAC_CHUNK_SIZE_LATENTS,
            overlap_latents=DAC_OVERLAP_LATENTS,
        )
        posterior_params = posterior_params.to(device=device, dtype=torch.float32)
        mean, logvar = torch.chunk(posterior_params, 2, dim=1)
        std = torch.exp(0.5 * logvar)

        for sample_index, row in enumerate(rows):
            sample_seed = int(seed + group_index * samples_per_audio + sample_index)
            generator = _make_generator(device, sample_seed)
            eps = torch.randn(std.shape, generator=generator, device=std.device, dtype=std.dtype)
            z = mean + std * eps

            decoded = decode_audio_latents(
                model,
                z,
                metadata,
                chunked=DAC_DECODE_CHUNKED,
                chunk_size_latents=DAC_CHUNK_SIZE_LATENTS,
                overlap_latents=DAC_OVERLAP_LATENTS,
            )
            out_audio = _to_saveable_audio(decoded)
            out_sr = int(metadata.get("sample_rate", sample_rate))
            out_path = output_dir / f"{row.audiocap_id}.wav"
            torchaudio.save(str(out_path), out_audio, sample_rate=out_sr, format="wav")
            generated += 1

    return generated


@torch.no_grad()
def _generate_audioldm2(
    grouped_rows: list[list[PromptRow]],
    output_dir: Path,
    samples_per_audio: int,
    seed: int,
    device: torch.device,
    model_id: str,
    local_files_only: bool,
) -> int:
    print(f"[audioldm2] Loading model: {model_id}")
    vae, vocoder = _load_audioldm_components(model_id, device=device, local_files_only=local_files_only)
    stft = _build_tacotron_stft()
    stft = _align_tacotron_stft_device(stft, torch.device("cpu"))

    vae_device = _module_device(vae, device)
    vocoder_device = _module_device(vocoder, device)

    generated = 0
    for group_index, rows in enumerate(tqdm(grouped_rows, desc="AudioLDM2 latent ceiling", unit="group")):
        gt_path = rows[0].gt_path
        wav_cpu, sample_rate = _load_mono_wav(gt_path)

        wav_16k = wav_cpu
        if sample_rate != AUDIO_LDM_SAMPLE_RATE:
            wav_16k = torchaudio.functional.resample(
                wav_16k, orig_freq=sample_rate, new_freq=AUDIO_LDM_SAMPLE_RATE
            )
        target_len = int(wav_16k.shape[-1])

        wav_16k = wav_16k - wav_16k.mean()
        peak = torch.clamp(wav_16k.abs().max(), min=1e-8)
        wav_16k = (wav_16k / peak) * 0.5
        wav_16k = wav_16k.to(device="cpu", dtype=torch.float32)

        mel, _ = stft.mel_spectrogram(wav_16k, normalize_fun=_audioldm_log_compression)
        mel_img = mel.transpose(1, 2).unsqueeze(1).to(device=vae_device, dtype=torch.float32)

        posterior = vae.encode(mel_img).latent_dist
        mean = posterior.mean
        logvar = posterior.logvar
        std = torch.exp(0.5 * logvar)

        eps_list = [
            torch.randn(
                std.shape,
                generator=_make_generator(device, int(seed + group_index * samples_per_audio + i)),
                device=std.device,
                dtype=std.dtype,
            )
            for i in range(len(rows))
        ]
        z_batch = torch.cat([mean + std * eps for eps in eps_list], dim=0)  # [N, C, H, W]
        mel_rec_batch = vae.decode(z_batch).sample  # [N, 1, mel_bins, T]
        mel_for_vocoder_batch = mel_rec_batch.squeeze(1).to(device=vocoder_device, dtype=torch.float32)
        wav_rec_batch = vocoder(mel_for_vocoder_batch)  # [N, T_audio]

        for sample_index, row in enumerate(rows):
            wav_rec = wav_rec_batch[sample_index]  # [T_audio]
            if wav_rec.shape[-1] < target_len:
                wav_rec = F.pad(wav_rec, (0, target_len - wav_rec.shape[-1]))
            elif wav_rec.shape[-1] > target_len:
                wav_rec = wav_rec[..., :target_len]

            out_audio = _to_saveable_audio(wav_rec)
            out_path = output_dir / f"{row.audiocap_id}.wav"
            torchaudio.save(str(out_path), out_audio, sample_rate=AUDIO_LDM_SAMPLE_RATE, format="wav")
            generated += 1

    return generated


@torch.no_grad()
def _generate_stableaudio(
    grouped_rows: list[list[PromptRow]],
    output_dir: Path,
    samples_per_audio: int,
    seed: int,
    device: torch.device,
    model_id: str,
    local_files_only: bool,
) -> int:
    from diffusers import StableAudioPipeline

    print(f"[stableaudio] Loading model: {model_id}")
    pipe = StableAudioPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    vae = pipe.vae.eval().to(device)
    del pipe

    vae_sr: int = int(vae.sampling_rate)

    generated = 0
    for group_index, rows in enumerate(tqdm(grouped_rows, desc="StableAudio latent ceiling", unit="group")):
        gt_path = rows[0].gt_path
        wav_cpu, sample_rate = _load_mono_wav(gt_path)

        if sample_rate != vae_sr:
            wav_cpu = torchaudio.functional.resample(wav_cpu, orig_freq=sample_rate, new_freq=vae_sr)

        # AutoencoderOobleck expects [B, C, T]; duplicate mono to stereo
        wav_input = wav_cpu.unsqueeze(0)  # [1, 1, T]
        if wav_input.shape[1] == 1:
            wav_input = wav_input.expand(-1, 2, -1).contiguous()
        wav_input = wav_input.to(device=device, dtype=torch.float32)

        posterior = vae.encode(wav_input).latent_dist
        mean = posterior.mean
        std = posterior.std

        eps_list = [
            torch.randn(
                std.shape,
                generator=_make_generator(device, int(seed + group_index * samples_per_audio + i)),
                device=std.device,
                dtype=std.dtype,
            )
            for i in range(len(rows))
        ]
        z_batch = torch.cat([mean + std * eps for eps in eps_list], dim=0)  # [N, C, T_latent]
        decoded_batch = vae.decode(z_batch).sample  # [N, 2, T_audio]

        for sample_index, row in enumerate(rows):
            out_audio = _to_saveable_audio(decoded_batch[sample_index : sample_index + 1])
            out_path = output_dir / f"{row.audiocap_id}.wav"
            torchaudio.save(str(out_path), out_audio, sample_rate=vae_sr, format="wav")
            generated += 1

    return generated


@torch.no_grad()
def _generate_audiogen(
    grouped_rows: list[list[PromptRow]],
    output_dir: Path,
    device: torch.device,
    compression_model,
    num_codebooks: int,
) -> int:
    total_codebooks = int(compression_model.total_codebooks)
    codec_device = _module_device(compression_model, device)
    codec_sample_rate = int(compression_model.sample_rate)
    codec_channels = int(compression_model.channels)
    compression_model.set_num_codebooks(int(num_codebooks))

    print(
        f"[audiogen] sample_rate={codec_sample_rate} channels={codec_channels} "
        f"codebooks={num_codebooks}/{total_codebooks}"
    )

    generated = 0
    for rows in tqdm(grouped_rows, desc=f"AudioGen latent ceiling nq={num_codebooks}", unit="group"):
        gt_path = rows[0].gt_path
        wav_cpu, sample_rate = _load_mono_wav(gt_path)

        if sample_rate != codec_sample_rate:
            wav_cpu = torchaudio.functional.resample(
                wav_cpu,
                orig_freq=sample_rate,
                new_freq=codec_sample_rate,
            )

        wav_input = wav_cpu.unsqueeze(0)  # [1, 1, T]
        if codec_channels == 1:
            pass
        elif codec_channels > 1:
            wav_input = wav_input.expand(-1, codec_channels, -1).contiguous()
        else:
            raise ValueError(f"AudioGen compression model reported invalid channel count: {codec_channels}")

        target_len = int(wav_input.shape[-1])
        wav_input = wav_input.to(device=codec_device, dtype=torch.float32)

        codes, scale = compression_model.encode(wav_input)
        decoded = compression_model.decode(codes, scale)
        if decoded.shape[-1] < target_len:
            decoded = F.pad(decoded, (0, target_len - decoded.shape[-1]))
        elif decoded.shape[-1] > target_len:
            decoded = decoded[..., :target_len]

        out_audio = _to_saveable_audio(decoded)
        # AudioGen's codec path is deterministic for a fixed waveform and codebook count,
        # so each caption row for the same youtube_id gets the same reconstruction.
        for row in rows:
            out_path = output_dir / f"{row.audiocap_id}.wav"
            torchaudio.save(str(out_path), out_audio, sample_rate=codec_sample_rate, format="wav")
            generated += 1

    return generated


def _load_audiogen_compression_model(model_id: str, device: torch.device):
    from audiocraft.models.loaders import load_compression_model

    print(f"[audiogen] Loading compression model from: {model_id}")
    return load_compression_model(model_id, device=str(device))


def _write_summary_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _get_dist_context(local_rank_arg: int | None) -> tuple[int, int, int]:
    import os

    def _env_int(name: str, default: int) -> int:
        v = os.environ.get(name)
        return int(v) if v is not None else default

    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = local_rank_arg if local_rank_arg is not None else _env_int("LOCAL_RANK", 0)
    return rank, world_size, local_rank


def _resolve_backends(mode: str) -> list[str]:
    if mode == "both":
        return ["dacvae", "audioldm2", "stableaudio", "audiogen"]
    return [mode]


def main() -> None:
    args = parse_args()
    if args.samples_per_audio < 1:
        raise SystemExit("--samples-per-audio must be >= 1.")

    prompts_csv = args.prompts_csv.expanduser().resolve()
    gt_dir = args.gt_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_tag = _build_run_tag(args.samples_per_audio, args.seed, args.limit_youtube, args.limit_rows)

    grouped_rows, prompt_stats = _collect_prompt_groups(
        prompts_csv=prompts_csv,
        gt_dir=gt_dir,
        samples_per_audio=args.samples_per_audio,
        limit_youtube=args.limit_youtube,
        limit_rows=args.limit_rows,
    )

    selected_rows = len(grouped_rows) * args.samples_per_audio
    target_filenames = _collect_target_filenames(grouped_rows)
    if len(target_filenames) != selected_rows:
        raise SystemExit("Internal error: target filename count does not match selected rows.")

    rank, world_size, local_rank = _get_dist_context(args.local_rank)

    # When --device auto and CUDA is available, pick the local CUDA device for this rank.
    device_arg = args.device
    if device_arg == "auto" and torch.cuda.is_available():
        device_arg = f"cuda:{local_rank}"
    device = choose_device(device_arg)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    set_global_seed(args.seed + rank)

    # Shard groups across ranks (round-robin so each rank gets a contiguous-ish spread).
    all_grouped_rows = grouped_rows
    grouped_rows = grouped_rows[rank::world_size]
    if len(grouped_rows) == 0:
        print(f"[rank {rank}/{world_size}] No groups assigned; exiting.")
        return
    shard_target_filenames = _collect_target_filenames(grouped_rows)
    shard_selected_rows = len(grouped_rows) * args.samples_per_audio

    print(
        f"[rank {rank}/{world_size}] "
        f"groups={len(grouped_rows)} rows={len(grouped_rows) * args.samples_per_audio} "
        f"of total groups={len(all_grouped_rows)} rows={selected_rows} "
        f"(csv_total={prompt_stats.total_rows}, valid={prompt_stats.valid_rows}, "
        f"missing_ids={prompt_stats.missing_ids}, missing_caption={prompt_stats.missing_caption}, "
        f"missing_gt={prompt_stats.missing_gt})"
    )
    print(f"Device: {device}")
    print(f"Run tag: {run_tag}")

    backends = _resolve_backends(args.vae)
    for backend in backends:
        if backend == "audiogen":
            compression_model = _load_audiogen_compression_model(args.audiogen_model_id, device)
            total_codebooks = int(compression_model.total_codebooks)
            audiogen_num_codebooks = _resolve_audiogen_num_codebooks(
                args.audiogen_num_codebooks,
                total_codebooks=total_codebooks,
            )

            for num_codebooks in audiogen_num_codebooks:
                backend_name = _audiogen_backend_name(num_codebooks, total_codebooks)
                output_dir = output_root / backend_name / run_tag
                _validate_output_dir_state(output_dir, target_filenames)

                generated = _generate_audiogen(
                    grouped_rows=grouped_rows,
                    output_dir=output_dir,
                    device=device,
                    compression_model=compression_model,
                    num_codebooks=num_codebooks,
                )

                actual_files = {p.name for p in output_dir.glob("*.wav")}
                missing_files = shard_target_filenames.difference(actual_files)
                extra_files = actual_files.difference(target_filenames)  # full set: other ranks' files are fine
                if missing_files or extra_files:
                    raise SystemExit(
                        f"{backend_name}: output verification failed. "
                        f"missing={len(missing_files)} extra={len(extra_files)}"
                    )
                if generated != shard_selected_rows:
                    raise SystemExit(
                        f"{backend_name}: generated count mismatch. expected={shard_selected_rows} got={generated}"
                    )

                summary = {
                    "backend": backend_name,
                    "run_tag": run_tag,
                    "rank": rank,
                    "world_size": world_size,
                    "seed": int(args.seed),
                    "samples_per_audio": int(args.samples_per_audio),
                    "output_dir": str(output_dir),
                    "prompts_csv": str(prompts_csv),
                    "gt_dir": str(gt_dir),
                    "selected_youtube_groups": int(len(grouped_rows)),
                    "selected_rows": int(shard_selected_rows),
                    "generated_files": int(generated),
                    "limit_youtube": int(args.limit_youtube) if args.limit_youtube is not None else None,
                    "limit_rows": int(args.limit_rows) if args.limit_rows is not None else None,
                    "device": str(device),
                    "prompt_stats": asdict(prompt_stats),
                    "audiogen_model_id": args.audiogen_model_id,
                    "audiogen_num_codebooks": int(num_codebooks),
                    "audiogen_total_codebooks": int(total_codebooks),
                    "deterministic_decode_per_group": True,
                    "decode_calls": int(len(grouped_rows)),
                }

                rank_suffix = f"_rank{rank}" if world_size > 1 else ""
                summary_path = output_dir.parent / f"{output_dir.name}_generation{rank_suffix}.json"
                _write_summary_json(summary_path, summary)
                print(f"[{backend_name}] generated {generated} files at: {output_dir}")
                print(f"[{backend_name}] summary JSON: {summary_path}")

                if device.type == "cuda":
                    torch.cuda.empty_cache()
            continue

        output_dir = output_root / backend / run_tag
        _validate_output_dir_state(output_dir, target_filenames)
        summary_extra: dict[str, object] = {}

        if backend == "dacvae":
            generated = _generate_dacvae(
                grouped_rows=grouped_rows,
                output_dir=output_dir,
                samples_per_audio=args.samples_per_audio,
                seed=args.seed,
                device=device,
                dacvae_weights=args.dacvae_weights,
            )
            summary_extra["dacvae_weights"] = args.dacvae_weights
        elif backend == "audioldm2":
            generated = _generate_audioldm2(
                grouped_rows=grouped_rows,
                output_dir=output_dir,
                samples_per_audio=args.samples_per_audio,
                seed=args.seed,
                device=device,
                model_id=args.audioldm_model_id,
                local_files_only=args.audioldm_local_files_only,
            )
            summary_extra["audioldm_model_id"] = args.audioldm_model_id
            summary_extra["audioldm_local_files_only"] = bool(args.audioldm_local_files_only)
        elif backend == "stableaudio":
            generated = _generate_stableaudio(
                grouped_rows=grouped_rows,
                output_dir=output_dir,
                samples_per_audio=args.samples_per_audio,
                seed=args.seed,
                device=device,
                model_id=args.stable_audio_model_id,
                local_files_only=args.stable_audio_local_files_only,
            )
            summary_extra["stable_audio_model_id"] = args.stable_audio_model_id
            summary_extra["stable_audio_local_files_only"] = bool(args.stable_audio_local_files_only)
        else:
            raise SystemExit(f"Unsupported backend: {backend}")

        actual_files = {p.name for p in output_dir.glob("*.wav")}
        missing_files = shard_target_filenames.difference(actual_files)
        extra_files = actual_files.difference(target_filenames)  # full set: other ranks' files are fine
        if missing_files or extra_files:
            raise SystemExit(
                f"{backend}: output verification failed. "
                f"missing={len(missing_files)} extra={len(extra_files)}"
            )
        if generated != shard_selected_rows:
            raise SystemExit(
                f"{backend}: generated count mismatch. expected={shard_selected_rows} got={generated}"
            )

        summary = {
            "backend": backend,
            "run_tag": run_tag,
            "rank": rank,
            "world_size": world_size,
            "seed": int(args.seed),
            "samples_per_audio": int(args.samples_per_audio),
            "output_dir": str(output_dir),
            "prompts_csv": str(prompts_csv),
            "gt_dir": str(gt_dir),
            "selected_youtube_groups": int(len(grouped_rows)),
            "selected_rows": int(shard_selected_rows),
            "generated_files": int(generated),
            "limit_youtube": int(args.limit_youtube) if args.limit_youtube is not None else None,
            "limit_rows": int(args.limit_rows) if args.limit_rows is not None else None,
            "device": str(device),
            "prompt_stats": asdict(prompt_stats),
        }
        summary.update(summary_extra)

        rank_suffix = f"_rank{rank}" if world_size > 1 else ""
        summary_path = output_dir.parent / f"{output_dir.name}_generation{rank_suffix}.json"
        _write_summary_json(summary_path, summary)
        print(f"[{backend}] generated {generated} files at: {output_dir}")
        print(f"[{backend}] summary JSON: {summary_path}")

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
