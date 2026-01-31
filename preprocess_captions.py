#!/usr/bin/env python3
"""
Preprocess captions to CLAP + T5 embeddings.

Supports:
- CPU pool (device=cpu) or GPU pool (device=cuda)
- Standardized output keys: t5_last_hidden, t5_len

Usage:
    # CPU pool
    python preprocess_captions.py \
        --metadata_path /path/to/manifest.jsonl \
        --output_dir /path/to/output \
        --device cpu --processes 28 --threads_per_process 4

    # GPU pool
    CUDA_VISIBLE_DEVICES=0,1,2,3 python preprocess_captions.py \
        --metadata_path /path/to/manifest.jsonl \
        --output_dir /path/to/output \
        --device cuda --processes 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.data_utils.preprocess import (
    TextEncoder,
    atomic_write_pt,
    belongs_to_node,
    resolve_node_setting,
    run_gpu_pool,
    run_pool,
    scan_cached_outputs,
)

# Global state for workers
_ENCODER = None
_WORKER_STATE = {}


def _init_worker(
    model_name: str,
    t5_model_name: str,
    device: str,
    output_dir: str,
):
    """Initialize worker with text encoder."""
    global _ENCODER, _WORKER_STATE

    _ENCODER = TextEncoder(
        clap_model_name=model_name,
        t5_model_name=t5_model_name,
        device=device,
    )

    _WORKER_STATE = {
        "output_dir": Path(output_dir),
    }


def _process_item(item: tuple[str, str]):
    """Process a single (key, caption) pair."""
    key, caption = item
    out_path = _WORKER_STATE["output_dir"] / f"{key}.pt"

    if out_path.exists():
        return

    payload = _ENCODER.encode(caption)
    atomic_write_pt(out_path, payload)


def get_args_parser():
    parser = argparse.ArgumentParser("Preprocess captions to CLAP + T5 embeddings", add_help=True)

    # Input
    parser.add_argument(
        "--metadata_path",
        required=True,
        type=str,
        help="Path to manifest.jsonl",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Output directory for embeddings",
    )

    # Models
    parser.add_argument(
        "--model_name",
        default="laion/larger_clap_music",
        type=str,
    )
    parser.add_argument(
        "--t5_model_name",
        default="google/flan-t5-large",
        type=str,
    )

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
    metadata_path = Path(args.metadata_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Scan existing cache
    if node_rank == 0:
        print(f"Scanning already cached .pt files under: {output_dir}")
    done_cache = scan_cached_outputs(output_dir)
    if node_rank == 0:
        print(f"Found {len(done_cache)} cached files.")

    # Build task list from manifest
    tasks = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            key = entry["key"]
            caption = entry["caption"]
            rel = f"{key}.pt"
            if rel in done_cache:
                continue
            if not belongs_to_node(key, node_rank, node_world_size):
                continue
            tasks.append((key, caption))

    print(
        f"Node {node_rank}/{node_world_size}: {len(tasks)} captions to process. "
        f"Cache output: {output_dir}"
    )

    if not tasks:
        return

    # Define init args
    init_args = (
        args.model_name,
        args.t5_model_name,
        args.device,
        str(output_dir),
    )

    def init_args_fn(gpu_id: int) -> tuple:
        device = f"cuda:{gpu_id}"
        return (
            args.model_name,
            args.t5_model_name,
            device,
            str(output_dir),
        )

    # Resolve process count and run
    device = args.device
    if device == "cpu":
        processes = args.processes or (os.cpu_count() or 1)
        run_pool(
            tasks,
            _process_item,
            _init_worker,
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
            _process_item,
            _init_worker,
            init_args_fn,
            num_gpus=processes,
            threads_per_gpu=args.threads_per_process,
            deterministic=args.deterministic,
            desc="Preprocessing",
        )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
