#!/usr/bin/env python3
"""
Unified preprocessing for AudioCaps + WavCaps.

Builds per-split/subset manifests, encodes audio+text latents into a single
payload per item, and optionally writes a merged training manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data_utils.preprocess import utils as prep


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Unified preprocessing (audio + captions)", add_help=True)
    parser.add_argument("--dataset", choices=["wavcaps", "audiocaps", "all"], default="all")
    parser.add_argument("--data-root", type=str, required=True)

    parser.add_argument("--wavcaps-json-root", type=str, default=None)
    parser.add_argument("--wavcaps-audio-root", type=str, default=None)
    parser.add_argument(
        "--wavcaps-subsets",
        type=str,
        default=None,
        help="Comma-separated subset names (default: all)",
    )

    parser.add_argument("--audiocaps-hf-dataset", type=str, default=prep.DEFAULT_AUDIOCAPS_DATASET)
    parser.add_argument("--audiocaps-splits", type=str, default=None)
    parser.add_argument("--audiocaps-hf-data-dir", type=str, default=None)
    parser.add_argument("--audiocaps-hf-cache-dir", type=str, default=None)
    parser.add_argument("--audiocaps-key-column", type=str, default="audiocap_id")
    parser.add_argument("--audiocaps-caption-column", type=str, default="caption")
    parser.add_argument("--audiocaps-audio-column", type=str, default="audio")
    parser.add_argument("--audiocaps-audio-length-column", type=str, default="audio_length")
    parser.add_argument("--audiocaps-dedupe", choices=["error", "first"], default="error")

    parser.add_argument("--dacvae-weights", type=str, default="facebook/dacvae-watermarked")
    parser.add_argument("--clap-model", type=str, default="laion/larger_clap_music")
    parser.add_argument("--t5-model", type=str, default="google/flan-t5-large")
    parser.add_argument("--chunk-size-latents", type=int, default=1024)
    parser.add_argument("--overlap-latents", type=int, default=12)
    parser.add_argument("--min-duration-seconds", type=float, default=0.05)
    parser.add_argument("--max-duration-seconds", type=float, default=600.0)
    parser.add_argument("--hash-prefix-len", type=int, default=2)

    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--threads-per-process", type=int, default=1)
    parser.add_argument(
        "--mp-start-method",
        choices=["fork", "spawn", "forkserver"],
        default="fork",
    )
    parser.add_argument("--node-rank", type=int, default=None)
    parser.add_argument("--node-world-size", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic")
    parser.set_defaults(deterministic=True)

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for merged training manifest",
    )
    parser.add_argument("--overwrite-manifests", action="store_true")
    parser.add_argument(
        "--merge-manifests",
        action="store_true",
        help="Merge-only: skip encoding and write merged manifest",
    )
    return parser


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if args.hash_prefix_len < 0 or args.hash_prefix_len > 32:
        raise ValueError("--hash-prefix-len must be between 0 and 32")

    data_root = Path(args.data_root).expanduser().resolve()
    if args.merge_manifests and not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else data_root / prep.DEFAULT_MERGED_MANIFEST
    )

    wavcaps_subsets_specified = args.wavcaps_subsets is not None
    audiocaps_splits_specified = args.audiocaps_splits is not None

    wavcaps_subsets = (
        prep.parse_csv(args.wavcaps_subsets)
        if args.wavcaps_subsets is not None
        else prep.parse_csv(prep.DEFAULT_WAVCAPS_SUBSETS)
    )
    audiocaps_splits = (
        prep.parse_csv(args.audiocaps_splits)
        if args.audiocaps_splits is not None
        else prep.parse_csv(prep.DEFAULT_AUDIOCAPS_SPLITS)
    )

    if args.merge_manifests:
        prep.merge_manifests(
            output_path=output_path,
            data_root=data_root,
            hash_prefix_len=args.hash_prefix_len,
            wavcaps_subsets=wavcaps_subsets,
            audiocaps_splits=audiocaps_splits,
            merge_all=True,
        )
        return

    node_rank, node_world_size = prep.resolve_node_settings(args.node_rank, args.node_world_size)
    weights_path = prep.resolve_dacvae_weights(args.dacvae_weights)

    wavcaps_json_root = Path(args.wavcaps_json_root).expanduser().resolve() if args.wavcaps_json_root else None
    wavcaps_audio_root = Path(args.wavcaps_audio_root).expanduser().resolve() if args.wavcaps_audio_root else None

    audiocaps_data_dir = prep.resolve_data_dir(args.audiocaps_hf_data_dir)
    audiocaps_cache_dir = prep.resolve_data_dir(args.audiocaps_hf_cache_dir)

    if args.dataset in {"wavcaps", "all"}:
        if wavcaps_audio_root is None:
            raise ValueError("--wavcaps-audio-root is required for WavCaps encoding")
        if not wavcaps_audio_root.exists():
            raise FileNotFoundError(f"Missing WavCaps audio root: {wavcaps_audio_root}")

        prep.build_wavcaps_manifests(
            data_root=data_root,
            json_root=wavcaps_json_root,
            subsets=wavcaps_subsets,
            overwrite=args.overwrite_manifests,
        )

        for subset in wavcaps_subsets:
            manifest_path = data_root / "WavCaps" / subset / "manifest.jsonl"
            latents_root = data_root / "WavCaps" / subset / "latents"
            prep.encode_wavcaps_subset(
                subset=subset,
                manifest_path=manifest_path,
                wavcaps_audio_root=wavcaps_audio_root,
                latents_root=latents_root,
                weights_path=weights_path,
                clap_model=args.clap_model,
                t5_model=args.t5_model,
                min_duration_seconds=args.min_duration_seconds,
                max_duration_seconds=args.max_duration_seconds,
                chunk_size_latents=args.chunk_size_latents,
                overlap_latents=args.overlap_latents,
                hash_prefix_len=args.hash_prefix_len,
                device=args.device,
                processes=args.processes,
                threads_per_process=args.threads_per_process,
                mp_start_method=args.mp_start_method,
                node_rank=node_rank,
                node_world_size=node_world_size,
                deterministic=args.deterministic,
            )

    if args.dataset in {"audiocaps", "all"}:
        prep.build_audiocaps_manifests(
            data_root=data_root,
            splits=audiocaps_splits,
            hf_dataset=args.audiocaps_hf_dataset,
            data_dir=audiocaps_data_dir,
            cache_dir=audiocaps_cache_dir,
            key_column=args.audiocaps_key_column,
            caption_column=args.audiocaps_caption_column,
            audio_length_column=args.audiocaps_audio_length_column,
            audio_column=args.audiocaps_audio_column,
            dedupe=args.audiocaps_dedupe,
            overwrite=args.overwrite_manifests,
        )

        for split in audiocaps_splits:
            manifest_path = data_root / "AudioCaps" / split / "manifest.jsonl"
            latents_root = data_root / "AudioCaps" / split / "latents"
            prep.encode_audiocaps_split(
                split=split,
                manifest_path=manifest_path,
                latents_root=latents_root,
                weights_path=weights_path,
                clap_model=args.clap_model,
                t5_model=args.t5_model,
                min_duration_seconds=args.min_duration_seconds,
                max_duration_seconds=args.max_duration_seconds,
                chunk_size_latents=args.chunk_size_latents,
                overlap_latents=args.overlap_latents,
                hash_prefix_len=args.hash_prefix_len,
                device=args.device,
                processes=args.processes,
                threads_per_process=args.threads_per_process,
                mp_start_method=args.mp_start_method,
                node_rank=node_rank,
                node_world_size=node_world_size,
                deterministic=args.deterministic,
                hf_dataset=args.audiocaps_hf_dataset,
                data_dir=audiocaps_data_dir,
                cache_dir=audiocaps_cache_dir,
                key_column=args.audiocaps_key_column,
                caption_column=args.audiocaps_caption_column,
                audio_column=args.audiocaps_audio_column,
            )

    should_merge = (
        args.dataset == "all"
        and not wavcaps_subsets_specified
        and not audiocaps_splits_specified
    )
    if should_merge:
        prep.merge_manifests(
            output_path=output_path,
            data_root=data_root,
            hash_prefix_len=args.hash_prefix_len,
            wavcaps_subsets=wavcaps_subsets,
            audiocaps_splits=audiocaps_splits,
            merge_all=False,
        )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
