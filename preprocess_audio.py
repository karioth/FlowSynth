#!/usr/bin/env python3
"""
Preprocess audio files to DACVAE latents.

Supports:
- File-based and HuggingFace dataset sources
- CPU pool (device=cpu) or GPU pool (device=cuda)

Usage:
    # CPU pool with file source
    python preprocess_audio.py --source files --data_dir /path/to/audio --device cpu --processes 8

    # GPU pool with HF source
    python preprocess_audio.py --source hf --hf_dataset OpenSound/AudioCaps --device cuda --processes 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from src.data_utils.preprocess import (
    apply_duration_filter,
    atomic_write_pt,
    belongs_to_node,
    list_audio_files,
    load_audio_from_hf_example,
    load_audio_from_path,
    resolve_hf_uid,
    resolve_node_setting,
    run_gpu_pool,
    run_pool,
    scan_cached_outputs,
    SkipLogger,
)

# Global state for workers
_MODEL = None
_WORKER_STATE = {}
_SKIP_LOGGER = None


def _init_worker_files(
    weights_path: str,
    device: str,
    input_root: str,
    out_root: str,
    min_duration_seconds: float,
    max_duration_seconds: float,
    chunk_size_latents: int,
    overlap_latents: int,
    node_rank: int,
):
    """Initialize worker for file-based audio caching."""
    global _MODEL, _WORKER_STATE, _SKIP_LOGGER

    import dacvae

    _MODEL = dacvae.DACVAE.load(weights_path).eval().to(device)

    out_root_path = Path(out_root)
    log_dir = out_root_path.parent
    _SKIP_LOGGER = SkipLogger(log_dir, node_rank)

    _WORKER_STATE = {
        "input_root": Path(input_root),
        "out_root": out_root_path,
        "min_duration_seconds": float(min_duration_seconds),
        "max_duration_seconds": float(max_duration_seconds),
        "chunk_size_latents": int(chunk_size_latents),
        "overlap_latents": int(overlap_latents),
    }


def _init_worker_hf(
    weights_path: str,
    device: str,
    hf_name: str,
    split: str,
    data_dir: str | None,
    cache_dir: str | None,
    out_root: str,
    name_column: str | None,
    min_duration_seconds: float,
    max_duration_seconds: float,
    chunk_size_latents: int,
    overlap_latents: int,
    node_rank: int,
):
    """Initialize worker for HuggingFace dataset audio caching."""
    global _MODEL, _WORKER_STATE, _SKIP_LOGGER

    import dacvae
    from datasets import load_dataset

    _MODEL = dacvae.DACVAE.load(weights_path).eval().to(device)

    # Load dataset in worker (required for spawn/fork isolation)
    ds = load_dataset(hf_name, data_dir=data_dir, split=split, cache_dir=cache_dir)

    out_root_path = Path(out_root)
    log_dir = out_root_path.parent
    _SKIP_LOGGER = SkipLogger(log_dir, node_rank)

    _WORKER_STATE = {
        "dataset": ds,
        "out_root": out_root_path,
        "name_column": name_column,
        "min_duration_seconds": float(min_duration_seconds),
        "max_duration_seconds": float(max_duration_seconds),
        "chunk_size_latents": int(chunk_size_latents),
        "overlap_latents": int(overlap_latents),
    }


def _process_file(path_str: str):
    """Process a single audio file."""
    from audiotools import AudioSignal

    from src.data_utils.utils import encode_audio_latents

    path = Path(path_str)
    input_root = _WORKER_STATE["input_root"]
    out_root = _WORKER_STATE["out_root"]

    rel = path.relative_to(input_root)
    out_rel = rel.with_suffix(".pt")
    out_path = out_root / out_rel

    if out_path.exists():
        return

    try:
        wav, sr = load_audio_from_path(path)

        # Apply duration filtering
        wav, skip_reason = apply_duration_filter(
            wav,
            sr,
            _WORKER_STATE["min_duration_seconds"],
            _WORKER_STATE["max_duration_seconds"],
        )

        if wav is None:
            _SKIP_LOGGER.log_skip(str(path), skip_reason)
            _SKIP_LOGGER.log_event(f"{skip_reason}, skipping: {path}")
            return

        # Encode
        audio = AudioSignal(wav.unsqueeze(0), sr)
        posterior_params, metadata = encode_audio_latents(
            _MODEL,
            audio,
            chunked=True,
            chunk_size_latents=_WORKER_STATE["chunk_size_latents"],
            overlap_latents=_WORKER_STATE["overlap_latents"],
        )

        tensor = posterior_params[0].to(torch.float32).cpu()
        latent_length = int(metadata.get("latent_length", tensor.shape[-1]))
        payload = {
            "posterior_params": tensor,
            "latent_length": latent_length,
            "source": str(path),
        }

        atomic_write_pt(out_path, payload)

    except Exception as exc:
        _SKIP_LOGGER.log_skip(str(path), f"{type(exc).__name__}: {exc}")


def _process_hf_index(idx: int):
    """Process a single HuggingFace dataset index."""
    from audiotools import AudioSignal

    from src.data_utils.utils import encode_audio_latents

    out_root = _WORKER_STATE["out_root"]
    ds = _WORKER_STATE["dataset"]
    source_id = f"{int(idx):09d}"

    try:
        ex = ds[int(idx)]
        name_column = _WORKER_STATE["name_column"]
        name_value = ex.get(name_column) if name_column else None
        source_id, out_name = resolve_hf_uid(idx, name_value)
        out_path = out_root / out_name

        if out_path.exists():
            return

        wav, sr = load_audio_from_hf_example(ex)

        # Apply duration filtering
        wav, skip_reason = apply_duration_filter(
            wav,
            sr,
            _WORKER_STATE["min_duration_seconds"],
            _WORKER_STATE["max_duration_seconds"],
        )

        if wav is None:
            _SKIP_LOGGER.log_skip(source_id, skip_reason)
            _SKIP_LOGGER.log_event(f"{skip_reason}, skipping: {source_id}")
            return

        # Encode
        audio = AudioSignal(wav.unsqueeze(0), sr)
        posterior_params, metadata = encode_audio_latents(
            _MODEL,
            audio,
            chunked=True,
            chunk_size_latents=_WORKER_STATE["chunk_size_latents"],
            overlap_latents=_WORKER_STATE["overlap_latents"],
        )

        tensor = posterior_params[0].to(torch.float32).cpu()
        latent_length = int(metadata.get("latent_length", tensor.shape[-1]))
        payload = {
            "posterior_params": tensor,
            "latent_length": latent_length,
            "source": source_id,
        }

        atomic_write_pt(out_path, payload)

    except Exception as exc:
        _SKIP_LOGGER.log_skip(source_id, f"{type(exc).__name__}: {exc}")


def get_args_parser():
    parser = argparse.ArgumentParser("Preprocess audio to DACVAE latents", add_help=True)

    # Source selection
    parser.add_argument(
        "--source",
        required=True,
        choices=["files", "hf"],
        help="Input source type",
    )

    # File source args
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Root directory for audio files (source=files)",
    )

    # HuggingFace source args
    parser.add_argument(
        "--hf_dataset",
        type=str,
        help="HuggingFace dataset name (source=hf)",
    )
    parser.add_argument("--hf_split", default="train", type=str)
    parser.add_argument("--hf_data_dir", default=None, type=str)
    parser.add_argument("--hf_cache_dir", default=None, type=str)
    parser.add_argument(
        "--name_column",
        nargs="?",
        const="audiocap_id",
        default=None,
        type=str,
        help="Column for output filenames (source=hf); omit to use row index",
    )

    # Output
    parser.add_argument(
        "--cached_path",
        default=None,
        type=str,
        help="Output directory (default: {input}_cached)",
    )

    # Model
    parser.add_argument(
        "--weights",
        default="facebook/dacvae-watermarked",
        type=str,
    )

    # Processing
    parser.add_argument("--chunk_size_latents", default=1024, type=int)
    parser.add_argument("--overlap_latents", default=12, type=int)
    parser.add_argument("--min_duration_seconds", default=0.05, type=float)
    parser.add_argument("--max_duration_seconds", default=600.0, type=float)

    # Device selection
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--processes",
        default=None,
        type=int,
        help="Number of worker processes (default: CPU cores or visible GPUs)",
    )
    parser.add_argument("--threads_per_process", default=1, type=int)
    parser.add_argument(
        "--mp_start_method",
        default="fork",
        choices=["fork", "spawn", "forkserver"],
    )

    # Node sharding
    parser.add_argument("--node_rank", default=None, type=int)
    parser.add_argument("--node_world_size", default=None, type=int)

    # Determinism
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.set_defaults(deterministic=True)

    return parser


@torch.no_grad()
def main(args):
    # Validate args based on source
    if args.source == "files" and not args.data_dir:
        raise ValueError("--data_dir required when source=files")
    if args.source == "hf" and not args.hf_dataset:
        raise ValueError("--hf_dataset required when source=hf")

    # Resolve weights path
    weights_ref = args.weights
    if os.path.exists(weights_ref):
        weights_path = weights_ref
    elif weights_ref.startswith("facebook/"):
        weights_path = weights_ref
    else:
        raise FileNotFoundError(f"Missing weights: {weights_ref}")

    # Resolve node settings
    node_rank = resolve_node_setting(
        args.node_rank, ["NODE_RANK", "SLURM_NODEID"], default=None
    )
    node_world_size = resolve_node_setting(
        args.node_world_size, ["NODE_WORLD_SIZE", "SLURM_JOB_NUM_NODES"], default=1
    )
    if node_world_size > 1 and node_rank is None:
        raise ValueError("node_rank is required when node_world_size > 1")
    if node_rank is None:
        node_rank = 0
    if node_rank < 0 or node_rank >= node_world_size:
        raise ValueError("node_rank must be in [0, node_world_size)")

    # Resolve output path
    if args.source == "files":
        data_dir = Path(os.path.normpath(args.data_dir)).resolve()
        cached_path = Path(
            os.path.normpath(args.cached_path or f"{data_dir}_cached")
        ).resolve()
    else:
        hf_data_dir = args.hf_data_dir
        if isinstance(hf_data_dir, str):
            if hf_data_dir.strip() == "" or hf_data_dir.strip().lower() in {"none", "null"}:
                hf_data_dir = None
        hf_cache_dir = args.hf_cache_dir
        if isinstance(hf_cache_dir, str) and hf_cache_dir.strip() == "":
            hf_cache_dir = None

        if args.cached_path:
            cached_path = Path(os.path.normpath(args.cached_path)).resolve()
        else:
            dataset_tag = args.hf_dataset.replace("/", "_")
            cached_path = Path(
                os.path.normpath(f"{dataset_tag}_{args.hf_split}_cached")
            ).resolve()

    cached_path.mkdir(parents=True, exist_ok=True)

    # Scan existing cache
    if node_rank == 0:
        print(f"Scanning already cached .pt files under: {cached_path}")
    done_cache = scan_cached_outputs(cached_path)
    if node_rank == 0:
        print(f"Found {len(done_cache)} cached files.")

    # Build task list based on source
    if args.source == "files":
        files = list_audio_files(data_dir)
        if node_rank == 0:
            print(f"Found {len(files)} files under: {data_dir}")

        tasks = []
        for path in files:
            rel = path.relative_to(data_dir)
            out_rel = rel.with_suffix(".pt").as_posix()
            if out_rel in done_cache:
                continue
            if not belongs_to_node(out_rel, node_rank, node_world_size):
                continue
            tasks.append(str(path))

        init_fn = _init_worker_files
        process_fn = _process_file
        init_args = (
            weights_path,
            args.device,
            str(data_dir),
            str(cached_path),
            args.min_duration_seconds,
            args.max_duration_seconds,
            args.chunk_size_latents,
            args.overlap_latents,
            node_rank,
        )

        def init_args_fn(gpu_id: int) -> tuple:
            device = f"cuda:{gpu_id}"
            return (
                weights_path,
                device,
                str(data_dir),
                str(cached_path),
                args.min_duration_seconds,
                args.max_duration_seconds,
                args.chunk_size_latents,
                args.overlap_latents,
                node_rank,
            )

    else:  # hf source
        from datasets import load_dataset

        ds = load_dataset(
            args.hf_dataset,
            data_dir=hf_data_dir,
            split=args.hf_split,
            cache_dir=hf_cache_dir,
        )
        total = len(ds)
        if node_rank == 0:
            print(f"HF dataset: {args.hf_dataset} ({args.hf_split}), n={total}.")

        tasks = []
        name_column = args.name_column
        name_ds = None
        if name_column is not None:
            if name_column not in ds.column_names:
                raise ValueError(
                    f"name_column {name_column!r} not found in dataset columns: {ds.column_names}"
                )
            name_ds = ds.select_columns([name_column])

        for idx in range(total):
            name_value = name_ds[idx][name_column] if name_ds is not None else None
            _, out_name = resolve_hf_uid(idx, name_value)
            if out_name in done_cache:
                continue
            if not belongs_to_node(out_name, node_rank, node_world_size):
                continue
            tasks.append(idx)

        init_fn = _init_worker_hf
        process_fn = _process_hf_index
        init_args = (
            weights_path,
            args.device,
            args.hf_dataset,
            args.hf_split,
            hf_data_dir,
            hf_cache_dir,
            str(cached_path),
            args.name_column,
            args.min_duration_seconds,
            args.max_duration_seconds,
            args.chunk_size_latents,
            args.overlap_latents,
            node_rank,
        )

        def init_args_fn(gpu_id: int) -> tuple:
            device = f"cuda:{gpu_id}"
            return (
                weights_path,
                device,
                args.hf_dataset,
                args.hf_split,
                hf_data_dir,
                hf_cache_dir,
                str(cached_path),
                args.name_column,
                args.min_duration_seconds,
                args.max_duration_seconds,
                args.chunk_size_latents,
                args.overlap_latents,
                node_rank,
            )

    print(
        f"Node {node_rank}/{node_world_size}: {len(tasks)} items to process. "
        f"Cache output: {cached_path}"
    )

    if not tasks:
        return

    # Resolve process count and run
    device = args.device
    if device == "cpu":
        processes = args.processes or (os.cpu_count() or 1)
        run_pool(
            tasks,
            process_fn,
            init_fn,
            init_args,
            num_workers=processes,
            threads_per_worker=args.threads_per_process,
            mp_start_method=args.mp_start_method,
            desc="Preprocessing",
        )
    else:  # device == "cuda"
        processes = args.processes or torch.cuda.device_count()
        run_gpu_pool(
            tasks,
            process_fn,
            init_fn,
            init_args_fn,
            num_gpus=processes,
            threads_per_gpu=args.threads_per_process,
            deterministic=args.deterministic,
            desc="Preprocessing",
        )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
