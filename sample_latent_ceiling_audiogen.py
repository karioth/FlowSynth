#!/usr/bin/env python3
"""
Generate AudioGen codec latent-ceiling audio for AudioCaps evaluation.

For each valid AudioCaps test row:
- Resolve GT audio from youtube_id in --gt-dir.
- Group rows by youtube_id.
- Encode/decode each GT audio once with AudioGen's compression model.
- Reuse the deterministic reconstruction for all rows in the group.
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


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS_CSV = PROJECT_ROOT / "audiocaps-test.csv"
DEFAULT_GT_DIR = PROJECT_ROOT / "samples_audiocaps_test"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "samples_audio_vae_ceilings"

DEFAULT_AUDIOGEN_MODEL_ID = "facebook/audiogen-medium"

REQUIRED_COLUMNS = ("audiocap_id", "youtube_id", "caption")


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
    parser = argparse.ArgumentParser(
        description="Generate AudioGen codec latent-ceiling AudioCaps samples for evaluation."
    )
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


def _module_device(module, fallback: torch.device) -> torch.device:
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return fallback


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
        value = os.environ.get(name)
        return int(value) if value is not None else default

    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = local_rank_arg if local_rank_arg is not None else _env_int("LOCAL_RANK", 0)
    return rank, world_size, local_rank


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

    device_arg = args.device
    if device_arg == "auto" and torch.cuda.is_available():
        device_arg = f"cuda:{local_rank}"
    device = choose_device(device_arg)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    set_global_seed(args.seed + rank)

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
        extra_files = actual_files.difference(target_filenames)
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


if __name__ == "__main__":
    main()
