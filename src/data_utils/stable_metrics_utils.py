"""
AudioCaps-specific CLAP score and PaSST KLD utilities.

These metrics intentionally reuse models from EqSynth's existing evaluation
registry (src.data_utils.evaluation_utils.build_model_registry).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import torch
import torch.nn.functional as F
import torchaudio
from tqdm.auto import tqdm

from kadtk.model_loader import CLAPLaionModel, CLAPModel, ModelLoader, PaSSTModel

from src.data_utils.evaluation_utils import (
    ModelRegistry,
    configure_numba_cache_dir,
    load_audio_with_torchaudio,
    prepare_model_for_inference,
)


AUDIOCAPS_REQUIRED_COLUMNS = ("audiocap_id", "youtube_id", "caption")
CLAP_PRE_RESAMPLE_SR = 16000

CLAP_SCORE_REGISTRY_MODELS = (
    "clap-2023",
    "clap-laion-audio",
    "clap-laion-music",
)
DEFAULT_CLAP_SCORE_MODEL = "clap-2023"

KLD_REGISTRY_MODELS = (
    "passt-base-10s",
    "passt-base-20s",
    "passt-base-30s",
    "passt-openmic",
    "passt-fsd50k",
)
DEFAULT_KLD_MODEL = "passt-base-10s"


@dataclass(frozen=True)
class AudioCapsPair:
    audiocap_id: str
    youtube_id: str
    caption: str
    gen_path: Path
    gt_path: Path


@dataclass(frozen=True)
class PairingStats:
    total_rows: int
    matched_rows: int
    used_rows: int
    missing_gen: int
    missing_gt: int
    missing_ids: int
    missing_caption: int


def _to_clean_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return str(value).strip()
    return str(value).strip()


def resolve_audiocaps_pairs(
    prompts_csv: Path,
    gen_dir: Path,
    gt_dir: Path,
    limit: int,
) -> tuple[list[AudioCapsPair], PairingStats]:
    if not prompts_csv.is_file():
        raise SystemExit(f"Prompts CSV not found: {prompts_csv}")
    if not gen_dir.is_dir():
        raise SystemExit(f"Generated directory not found: {gen_dir}")
    if not gt_dir.is_dir():
        raise SystemExit(f"Ground-truth directory not found: {gt_dir}")

    df = pd.read_csv(prompts_csv)
    missing_columns = [c for c in AUDIOCAPS_REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise SystemExit(
            f"Prompts CSV is missing required column(s): {missing_columns}. "
            f"Expected: {list(AUDIOCAPS_REQUIRED_COLUMNS)}"
        )

    total_rows = int(len(df))
    missing_gen = 0
    missing_gt = 0
    missing_ids = 0
    missing_caption = 0
    matched: list[AudioCapsPair] = []

    for row in df[list(AUDIOCAPS_REQUIRED_COLUMNS)].itertuples(index=False):
        audiocap_id = _to_clean_id(row.audiocap_id)
        youtube_id = _to_clean_id(row.youtube_id)
        caption = "" if pd.isna(row.caption) else str(row.caption).strip()

        if not audiocap_id or not youtube_id:
            missing_ids += 1
            continue
        if not caption:
            missing_caption += 1
            continue

        gen_path = gen_dir / f"{audiocap_id}.wav"
        gt_candidates = (
            gt_dir / f"{youtube_id}.wav",
            gt_dir / f"Y{youtube_id}.wav",
        )
        gt_path = gt_candidates[0]
        has_gen = gen_path.is_file()
        has_gt = False
        for candidate in gt_candidates:
            if candidate.is_file():
                gt_path = candidate
                has_gt = True
                break
        if not has_gen:
            missing_gen += 1
        if not has_gt:
            missing_gt += 1
        if has_gen and has_gt:
            matched.append(
                AudioCapsPair(
                    audiocap_id=audiocap_id,
                    youtube_id=youtube_id,
                    caption=caption,
                    gen_path=gen_path,
                    gt_path=gt_path,
                )
            )

    used = matched[:limit] if limit > 0 else matched
    stats = PairingStats(
        total_rows=total_rows,
        matched_rows=len(matched),
        used_rows=len(used),
        missing_gen=missing_gen,
        missing_gt=missing_gt,
        missing_ids=missing_ids,
        missing_caption=missing_caption,
    )
    return used, stats


def _build_metric_model(
    registry: ModelRegistry,
    model_name: str,
    device: torch.device,
) -> ModelLoader:
    if model_name not in registry:
        raise SystemExit(f"Model '{model_name}' is not present in evaluation_utils registry.")

    if model_name.startswith("clap-laion-"):
        configure_numba_cache_dir()

    model = registry[model_name](None)

    if model_name.startswith("clap-laion-"):
        # laion_clap import path may rewrite NUMBA cache to /tmp.
        configure_numba_cache_dir()

    prepare_model_for_inference(model, device)
    return model


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise TypeError(f"Unsupported embedding type: {type(x)}")


def _extract_text_embeddings(model: ModelLoader, texts: list[str]) -> torch.Tensor:
    with torch.no_grad():
        if isinstance(model, CLAPLaionModel):
            text_emb = model.model.get_text_embedding(texts, use_tensor=True)
        elif isinstance(model, CLAPModel):
            text_emb = model.model.get_text_embeddings(texts)
        else:
            raise TypeError(f"Unsupported CLAP model type: {type(model)}")
    return _as_tensor(text_emb).detach().float().cpu()


def _load_audio_for_clap_score(path: Path, target_sr: int) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav.mean(dim=0, keepdim=True)

    if sr != CLAP_PRE_RESAMPLE_SR:
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=sr,
            new_freq=CLAP_PRE_RESAMPLE_SR,
        )
        sr = CLAP_PRE_RESAMPLE_SR

    if sr != target_sr:
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=sr,
            new_freq=target_sr,
        )

    return wav.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def compute_clap_score(
    pairs: Sequence[AudioCapsPair],
    registry: ModelRegistry,
    model_name: str,
    device: torch.device,
) -> float:
    if len(pairs) == 0:
        raise SystemExit("No matched AudioCaps pairs available for CLAP score.")
    if model_name not in CLAP_SCORE_REGISTRY_MODELS:
        raise SystemExit(
            f"Invalid CLAP score model '{model_name}'. Allowed: {list(CLAP_SCORE_REGISTRY_MODELS)}"
        )
    if device.type != "cuda":
        raise SystemExit("CLAP score requires a CUDA device.")

    model = _build_metric_model(registry=registry, model_name=model_name, device=device)
    if not isinstance(model, (CLAPModel, CLAPLaionModel)):
        raise SystemExit(f"Model '{model_name}' is not a CLAP model.")

    id_to_caption: dict[str, str] = {}
    for pair in pairs:
        if pair.audiocap_id not in id_to_caption:
            id_to_caption[pair.audiocap_id] = pair.caption

    text_embeddings: dict[str, torch.Tensor] = {}
    text_ids = list(id_to_caption.keys())
    batch_size = 64
    for i in tqdm(range(0, len(text_ids), batch_size), desc="CLAP text embeddings", unit="batch"):
        batch_ids = text_ids[i : i + batch_size]
        batch_texts = [id_to_caption[x] for x in batch_ids]
        batch_emb = _extract_text_embeddings(model, batch_texts)
        for sample_id, emb in zip(batch_ids, batch_emb):
            text_embeddings[sample_id] = emb

    score_sum = 0.0
    count = 0
    for pair in tqdm(pairs, desc="CLAP score", unit="file"):
        audio = _load_audio_for_clap_score(pair.gen_path, target_sr=model.sr)
        with torch.no_grad():
            audio_emb = model.get_embedding(audio)
        audio_emb = np.asarray(audio_emb, dtype=np.float32)
        if audio_emb.ndim == 1:
            audio_emb = np.expand_dims(audio_emb, axis=0)
        audio_vec = torch.from_numpy(audio_emb).mean(dim=0, keepdim=True)
        text_vec = text_embeddings[pair.audiocap_id].unsqueeze(0)
        cosine = F.cosine_similarity(audio_vec, text_vec, dim=1, eps=1e-8)[0]
        score_sum += float(cosine.item())
        count += 1

    if count == 0:
        raise RuntimeError("No CLAP score samples were evaluated.")
    return score_sum / count


class _PatchPasstStft:
    """
    Patch torch.stft call signature for older PaSST wrappers.
    """

    def __init__(self) -> None:
        self._old_stft = torch.stft

    def __enter__(self) -> None:
        torch.stft = partial(torch.stft, return_complex=False)

    def __exit__(self, *exc) -> None:
        torch.stft = self._old_stft


def _kld_num_classes(model_name: str) -> int:
    mapping = {
        "passt-base-10s": 527,
        "passt-base-20s": 527,
        "passt-base-30s": 527,
        "passt-openmic": 20,
        "passt-fsd50k": 200,
    }
    return mapping[model_name]


def _collect_passt_probabilities(
    model: PaSSTModel,
    audio_path: Path,
    num_classes: int,
    collect: str = "mean",
) -> torch.Tensor:
    audio = load_audio_with_torchaudio(audio_path, target_sr=model.sr, audio_len=None)
    audio = pyln.normalize.peak(audio, -1.0)

    window_seconds = int(round(model.limit / model.sr))
    overlap_seconds = max(1, window_seconds // 2)
    window_samples = window_seconds * model.sr
    step_size = (window_seconds - overlap_seconds) * model.sr

    logits_per_window: list[torch.Tensor] = []
    for i in range(0, max(step_size, len(audio) - step_size), step_size):
        window = audio[i : i + window_samples]
        if len(window) < window_samples:
            if len(window) > int(window_samples * 0.15):
                tmp = np.zeros(window_samples)
                tmp[: len(window)] = window
                window = tmp
            else:
                continue

        audio_wave = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(model.device)
        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
            with torch.no_grad(), _PatchPasstStft():
                out = model.model(audio_wave)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        logits = out[:, :num_classes].squeeze(0).detach().cpu()
        logits_per_window.append(logits)

    if len(logits_per_window) == 0:
        raise RuntimeError(f"No PaSST windows produced for '{audio_path}'.")

    logits_stack = torch.stack(logits_per_window)
    if collect == "mean":
        pooled = torch.mean(logits_stack, dim=0)
    elif collect == "max":
        pooled, _ = torch.max(logits_stack, dim=0)
    else:
        raise ValueError(f"Unsupported collect='{collect}'. Expected 'mean' or 'max'.")
    return F.softmax(pooled, dim=0).squeeze().cpu()


def compute_passt_kld(
    pairs: Sequence[AudioCapsPair],
    registry: ModelRegistry,
    model_name: str,
    device: torch.device,
    collect: str = "mean",
) -> float:
    if len(pairs) == 0:
        raise SystemExit("No matched AudioCaps pairs available for PaSST KLD.")
    if model_name not in KLD_REGISTRY_MODELS:
        raise SystemExit(f"Invalid KLD model '{model_name}'. Allowed: {list(KLD_REGISTRY_MODELS)}")
    if device.type != "cuda":
        raise SystemExit("PaSST KLD requires a CUDA device.")

    model = _build_metric_model(registry=registry, model_name=model_name, device=device)
    if not isinstance(model, PaSSTModel):
        raise SystemExit(f"Model '{model_name}' is not a PaSST classification model.")

    num_classes = _kld_num_classes(model_name)
    kl_total = 0.0
    count = 0
    for pair in tqdm(pairs, desc="PaSST KLD", unit="file"):
        ref_p = _collect_passt_probabilities(model, pair.gt_path, num_classes=num_classes, collect=collect)
        eval_p = _collect_passt_probabilities(model, pair.gen_path, num_classes=num_classes, collect=collect)
        kl = F.kl_div((ref_p + 1e-6).log(), eval_p, reduction="sum", log_target=False)
        kl_total += float(kl.item())
        count += 1

    if count == 0:
        raise RuntimeError("No PaSST KLD samples were evaluated.")
    return kl_total / count
