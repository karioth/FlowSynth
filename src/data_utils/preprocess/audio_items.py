"""
Audio loading and duration filtering utilities.

Supports both file-based and HuggingFace dataset sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .common import _fast_scandir


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def list_audio_files(root: Path | str, exts: set[str] | None = None) -> list[Path]:
    """
    Recursively list audio files under a directory.

    Args:
        root: Root directory to scan
        exts: File extensions to include (default: AUDIO_EXTS)

    Returns:
        Sorted list of Path objects for audio files
    """
    exts = AUDIO_EXTS if exts is None else exts
    _, files = _fast_scandir(str(root), exts)
    files = [Path(p) for p in files]
    files.sort()
    return files


@dataclass
class AudioItem:
    """Represents a single audio item ready for encoding."""

    id: str  # Source identifier (path or HF index)
    wav: torch.Tensor  # Waveform [C, T], clamped to [-1, 1]
    sr: int  # Sample rate
    out_path: str  # Target output path


def non_silence_ratio(wav: torch.Tensor, threshold: float = 1e-4) -> float:
    """
    Calculate the ratio of non-silent samples in a waveform.

    Args:
        wav: Waveform tensor [C, T] or [T]
        threshold: Amplitude threshold for silence detection

    Returns:
        Ratio of non-silent samples (0.0 to 1.0)
    """
    non_silent = wav.abs() > threshold
    if non_silent.ndim == 2:
        non_silent = non_silent.any(dim=0)
    return non_silent.float().mean().item()


def apply_duration_filter(
    wav: torch.Tensor,
    sr: int,
    min_seconds: float,
    max_seconds: float,
    non_silence_threshold: float = 0.7,
    max_crop_attempts: int = 10,
) -> tuple[torch.Tensor | None, str | None]:
    """
    Apply min/max duration filtering with non-silence-aware cropping.

    Args:
        wav: Input waveform [C, T]
        sr: Sample rate
        min_seconds: Minimum duration (skip if shorter)
        max_seconds: Maximum duration (crop if longer, 0 = no max)
        non_silence_threshold: Required non-silence ratio for crops
        max_crop_attempts: Number of random crops to try

    Returns:
        (filtered_wav, skip_reason) - wav is None if skipped, skip_reason is None if kept
    """
    num_frames = wav.shape[-1]

    # Check minimum duration
    min_frames = int(min_seconds * float(sr))
    if num_frames < min_frames:
        return None, "too_short"

    # Check maximum duration
    max_frames = int(max_seconds * float(sr))
    if max_frames > 0 and num_frames > max_frames:
        # Try to find a non-silent crop
        max_start = num_frames - max_frames
        found = False
        seg = None
        for _ in range(max_crop_attempts):
            start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            seg = wav[:, start : start + max_frames]
            if non_silence_ratio(seg) >= non_silence_threshold:
                wav = seg
                found = True
                break
        if not found and seg is not None:
            wav = seg  # Use last segment even if too silent

    return wav, None


def load_audio_from_path(path: str | Path) -> tuple[torch.Tensor, int]:
    """
    Load audio from a file path.

    Args:
        path: Path to audio file

    Returns:
        (wav, sr) where wav is [C, T] tensor clamped to [-1, 1]
    """
    wav, sr = torchaudio.load(str(path))
    wav = wav.clamp_(-1, 1)
    return wav, int(sr)


def load_audio_from_hf_example(example: dict[str, Any]) -> tuple[torch.Tensor, int]:
    """
    Load audio from a HuggingFace dataset example.

    Expects example["audio"] with keys: "array", "sampling_rate"

    Args:
        example: HuggingFace dataset row

    Returns:
        (wav, sr) where wav is [C, T] tensor clamped to [-1, 1]
    """
    a = example["audio"]
    arr = a["array"]
    sr = int(a["sampling_rate"])

    x = torch.from_numpy(arr).to(torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # (T,) -> (1, T)
    else:
        x = x.transpose(0, 1)  # (T, C) -> (C, T)

    x = x.clamp_(-1, 1)
    return x, sr


def resolve_hf_uid(idx: int, name_value: str | None) -> tuple[str, str]:
    """
    Resolve source ID and output filename from HF index and optional name column.

    Args:
        idx: Dataset index
        name_value: Optional value from name column

    Returns:
        (source_id, out_filename) - out_filename always ends with .pt
    """
    if name_value is None:
        source_id = f"{int(idx):09d}"
    else:
        source_id = str(name_value)

    if source_id.endswith(".pt"):
        out_name = source_id
    else:
        out_name = f"{source_id}.pt"

    return source_id, out_name
