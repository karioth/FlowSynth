#!/usr/bin/env python3
"""
Compute only the AudioCaps CLAP score for a generated directory.

This intentionally reuses the exact pairing and CLAP-scoring path from
evaluate.py so debugging stays apples-to-apples.

Important:
- The CLAP score compares generated audio against the caption text.
- The GT directory is only used to keep the same matched-row filtering as
  evaluate.py; GT audio is not embedded for the score itself.
- Before CLAP embedding, audio is always resampled to 16 kHz and then to the
  CLAP model's native sample rate.
- By default, stdout contains only the scalar CLAP score.
- Optional sample-rate diagnostics are printed to stderr.
"""

from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

import torch
import torchaudio

from src.data_utils.evaluation_utils import build_model_registry, resolve_device
from src.data_utils.stable_metrics_utils import (
    CLAP_SCORE_REGISTRY_MODELS,
    DEFAULT_CLAP_SCORE_MODEL,
    compute_clap_score,
    resolve_audiocaps_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS_CSV = PROJECT_ROOT / "audiocaps-test.csv"
DEFAULT_GT_DIR = PROJECT_ROOT / "audio_samples" / "audiocaps_test_gt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute only CLAP score for an AudioCaps-style generated directory."
    )
    parser.add_argument("--gen", type=Path, required=True, help="Directory with generated .wav files.")
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT_DIR,
        help=f"Directory with ground-truth .wav files (default: {DEFAULT_GT_DIR}).",
    )
    parser.add_argument(
        "--prompts-csv",
        type=Path,
        default=DEFAULT_PROMPTS_CSV,
        help=f"AudioCaps CSV with audiocap_id,youtube_id,caption (default: {DEFAULT_PROMPTS_CSV}).",
    )
    parser.add_argument(
        "--clap-model",
        type=str,
        default=DEFAULT_CLAP_SCORE_MODEL,
        choices=sorted(CLAP_SCORE_REGISTRY_MODELS),
        help="Registry CLAP model used by CLAP score metric.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g., cuda, cuda:0).")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, limit CLAP scoring to the first N matched CSV rows.",
    )
    parser.add_argument(
        "--debug-sample-rates",
        action="store_true",
        help="Print sample-rate summaries for matched generated/GT files to stderr.",
    )
    return parser.parse_args()


def _read_sample_rate(path: Path) -> int | None:
    try:
        return int(torchaudio.info(str(path)).sample_rate)
    except Exception:
        return None


def _emit_sample_rate_debug(pairs) -> None:
    gen_counts: Counter[int | None] = Counter()
    gt_counts: Counter[int | None] = Counter()
    mismatch_examples: list[str] = []

    for pair in pairs:
        gen_sr = _read_sample_rate(pair.gen_path)
        gt_sr = _read_sample_rate(pair.gt_path)
        gen_counts[gen_sr] += 1
        gt_counts[gt_sr] += 1
        if gen_sr != gt_sr and len(mismatch_examples) < 12:
            mismatch_examples.append(
                f"{pair.audiocap_id}: gen_sr={gen_sr} gt_sr={gt_sr} "
                f"gen={pair.gen_path.name} gt={pair.gt_path.name}"
            )

    print(f"[debug] matched_pairs={len(pairs)}", file=sys.stderr)
    print(f"[debug] gen sample rates: {dict(sorted(gen_counts.items(), key=lambda x: str(x[0])))}", file=sys.stderr)
    print(f"[debug] gt sample rates: {dict(sorted(gt_counts.items(), key=lambda x: str(x[0])))}", file=sys.stderr)
    if mismatch_examples:
        print("[debug] first sample-rate mismatches:", file=sys.stderr)
        for item in mismatch_examples:
            print(f"[debug]   {item}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0.")

    torch.set_grad_enabled(False)

    prompts_csv = args.prompts_csv.resolve()
    gen_dir = args.gen.resolve()
    gt_dir = args.gt.resolve()

    registry = build_model_registry()
    device = resolve_device(args.device)

    pairs, pairing_stats = resolve_audiocaps_pairs(
        prompts_csv=prompts_csv,
        gen_dir=gen_dir,
        gt_dir=gt_dir,
        limit=args.limit,
    )
    if pairing_stats.used_rows == 0:
        raise SystemExit(
            "No matched AudioCaps rows found for CLAP metric computation. "
            f"CSV rows={pairing_stats.total_rows}, matched={pairing_stats.matched_rows}, "
            f"missing_gen={pairing_stats.missing_gen}, missing_gt={pairing_stats.missing_gt}, "
            f"missing_ids={pairing_stats.missing_ids}, missing_caption={pairing_stats.missing_caption}."
        )

    if args.debug_sample_rates:
        _emit_sample_rate_debug(pairs)

    clap_score = compute_clap_score(
        pairs,
        registry=registry,
        model_name=args.clap_model,
        device=device,
    )
    print(f"{clap_score:.6f}")


if __name__ == "__main__":
    main()
