#!/usr/bin/env python3
"""
GPU-optimized KAD/FAD evaluation with kadtk models, without embedding/stat caching.

This script is intentionally separate from kadtk's CLI pipeline:
- No embedding cache files.
- No stats cache files.
- Single model instance in one process for deterministic GPU use.
- Optional CPU thread prefetch for decode/resample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from src.data_utils.evaluation_utils import (
    build_model_registry,
    collect_wavs,
    compute_embeddings,
    compute_scores,
    configure_numba_cache_dir,
    prepare_model_for_inference,
    resolve_device,
)
from src.data_utils.stable_metrics_utils import (
    CLAP_SCORE_REGISTRY_MODELS,
    DEFAULT_CLAP_SCORE_MODEL,
    DEFAULT_KLD_MODEL,
    KLD_REGISTRY_MODELS,
    compute_clap_score,
    compute_passt_kld,
    resolve_audiocaps_pairs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS_CSV = PROJECT_ROOT / "audiocaps-test.csv"
DEFAULT_GT_DIR = PROJECT_ROOT / "audio_samples" / "audiocaps_test_gt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FAD/KAD/CLAP score/PaSST KLD for AudioCaps-style runs."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="panns-wavegram-logmel",
        help="kadtk model name (PyTorch models only).",
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
    parser.add_argument(
        "--kld-model",
        type=str,
        default=DEFAULT_KLD_MODEL,
        choices=sorted(KLD_REGISTRY_MODELS),
        help="Registry PaSST classification model used by KLD metric.",
    )
    parser.add_argument("--workers", type=int, default=4, help="CPU worker threads for decode/resample prefetch.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g., cuda, cuda:0, cpu).")
    parser.add_argument("--audio-len", type=float, default=None, help="Expected clip length in seconds.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, limit FAD/KAD to first N sorted WAV files per folder and CLAP/KLD to first N matched CSV rows.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write metrics JSON (default: <parent of --gen>/<gen_dir_name>.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1.")
    torch.set_grad_enabled(False)

    prompts_csv = args.prompts_csv.resolve()
    gen_dir = args.gen.resolve()
    gt_dir = args.gt.resolve()
    registry = build_model_registry()
    available_models = sorted(registry.keys())

    if args.model.startswith("openl3-"):
        raise SystemExit(
            f"Model '{args.model}' is not supported by this script: TF/Keras-based OpenL3 models are excluded."
        )
    if args.model not in registry:
        raise SystemExit(
            f"Unknown model '{args.model}'.\nAvailable PyTorch models:\n" + "\n".join(available_models)
        )

    if args.model.startswith("clap-laion-"):
        configure_numba_cache_dir()

    device = resolve_device(args.device)
    model = registry[args.model](args.audio_len)

    if args.model.startswith("clap-laion-"):
        # CLAP import path in laion_clap may overwrite NUMBA_CACHE_DIR to /tmp;
        # enforce our user-writable cache target right before model load.
        configure_numba_cache_dir()

    # Keep explicit device control in this process (no model duplication across workers).
    prepare_model_for_inference(model, device)

    # Keep FAD/KAD behavior folder-based (independent from CSV pairing).
    gt_files = collect_wavs(gt_dir, args.limit)
    gen_files = collect_wavs(gen_dir, args.limit)

    with torch.no_grad():
        gt_emb = compute_embeddings(
            gt_files,
            model=model,
            workers=args.workers,
            label=f"Embedding GT ({len(gt_files)} files)",
        )
        gen_emb = compute_embeddings(
            gen_files,
            model=model,
            workers=args.workers,
            label=f"Embedding GEN ({len(gen_files)} files)",
        )
        fad_score, kad_score = compute_scores(gt_emb=gt_emb, gen_emb=gen_emb, device=device)
    
    print(
        f'"{model.name}" fad: {fad_score:.2f} kad: {kad_score:.2f} '
    )
    pairs, pairing_stats = resolve_audiocaps_pairs(
        prompts_csv=prompts_csv,
        gen_dir=gen_dir,
        gt_dir=gt_dir,
        limit=args.limit,
    )
    if pairing_stats.used_rows == 0:
        raise SystemExit(
            "No matched AudioCaps rows found for CLAP/KLD metric computation. "
            f"CSV rows={pairing_stats.total_rows}, matched={pairing_stats.matched_rows}, "
            f"missing_gen={pairing_stats.missing_gen}, missing_gt={pairing_stats.missing_gt}, "
            f"missing_ids={pairing_stats.missing_ids}, missing_caption={pairing_stats.missing_caption}."
        )

    clap_score = compute_clap_score(
        pairs,
        registry=registry,
        model_name=args.clap_model,
        device=device,
    )
    passt_kld = compute_passt_kld(
        pairs,
        registry=registry,
        model_name=args.kld_model,
        device=device,
        collect="mean",
    )

    output_json_path = args.output_json.resolve() if args.output_json else gen_dir.parent / f"{gen_dir.name}.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "generated_path": str(gen_dir),
        "fad_model": model.name,
        "clap_model": args.clap_model,
        "kld_model": args.kld_model,
        "num_pairs": pairing_stats.used_rows,
        "fad": float(fad_score),
        "kad": float(kad_score),
        "clap_score": float(clap_score),
        "passt_kld": float(passt_kld),
    }
    with output_json_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print(
        f'"{model.name}" fad: {fad_score:.2f} kad: {kad_score:.2f} '
        f'clap_score: {clap_score:.4f} passt_kld: {passt_kld:.4f}'
    )
    print(f"Saved metrics JSON: {output_json_path}")


if __name__ == "__main__":
    main()
