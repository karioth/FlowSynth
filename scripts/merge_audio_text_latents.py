#!/usr/bin/env python3
"""
Merge per-sample audio latents and text embeddings into a single .pt file.

Reads JSONL manifest(s) with audio_path and text_path fields and writes merged
outputs to a sibling directory (default: replaces `audio_latents` with `latents`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.data_utils.preprocess.common import (
    atomic_write_pt,
    belongs_to_node,
    resolve_node_setting,
    SkipLogger,
)
from src.data_utils.preprocess.runners import run_pool

_WORKER_STATE: dict[str, object] = {}
_SKIP_LOGGER: SkipLogger | None = None


def _init_worker(
    skip_missing: bool,
    skip_errors: bool,
    overwrite: bool,
    log_dir: str,
    node_rank: int,
) -> None:
    global _WORKER_STATE, _SKIP_LOGGER

    _WORKER_STATE = {
        "skip_missing": bool(skip_missing),
        "skip_errors": bool(skip_errors),
        "overwrite": bool(overwrite),
    }
    _SKIP_LOGGER = SkipLogger(log_dir, node_rank)


def _load_audio_payload(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, torch.Tensor):
        return {"posterior_params": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported audio payload type: {type(payload)}")
    if "posterior_params" not in payload:
        raise ValueError(f"Missing posterior_params in audio payload: {path}")
    return payload


def _load_text_payload(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported text payload type: {type(payload)}")
    if "clap_embedding" not in payload or "t5_last_hidden" not in payload:
        raise ValueError(f"Missing text keys in payload: {path}")
    return payload


def _process_item(item: tuple[str, str, str]) -> None:
    audio_path, text_path, out_path = item

    if not _WORKER_STATE.get("overwrite", False) and os.path.exists(out_path):
        return

    try:
        audio_payload = _load_audio_payload(audio_path)
        text_payload = _load_text_payload(text_path)
    except FileNotFoundError as exc:
        if _WORKER_STATE.get("skip_missing", False):
            if _SKIP_LOGGER is not None:
                _SKIP_LOGGER.log_skip(audio_path, f"missing_file: {exc}")
            return
        raise
    except Exception as exc:
        if _WORKER_STATE.get("skip_errors", False):
            if _SKIP_LOGGER is not None:
                _SKIP_LOGGER.log_skip(audio_path, f"error: {type(exc).__name__}: {exc}")
            return
        raise

    merged = dict(audio_payload)
    overlap = set(merged.keys()) & set(text_payload.keys())
    if overlap:
        raise ValueError(f"Key collision while merging {audio_path} and {text_path}: {sorted(overlap)}")
    merged.update(text_payload)

    atomic_write_pt(out_path, merged)


def _parse_manifest_paths(values: list[str] | None) -> list[str]:
    if not values:
        raise ValueError("No manifest paths provided")
    manifest_paths: list[str] = []
    for value in values:
        if value is None:
            continue
        for item in value.split(","):
            item = item.strip()
            if item:
                manifest_paths.append(item)
    if not manifest_paths:
        raise ValueError("No manifest paths provided")
    for path in manifest_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing manifest: {path}")
    return manifest_paths


def _resolve_path(path_str: str, data_root: Path | None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if data_root is None:
        raise ValueError(f"Relative path without data_root: {path_str}")
    return (data_root / path).absolute()


def _default_out_path(audio_path: Path, out_dir_name: str) -> Path:
    parts = audio_path.parts
    if "audio_latents" not in parts:
        raise ValueError(f"Expected 'audio_latents' in audio path: {audio_path}")
    idx = parts.index("audio_latents")
    return Path(*parts[:idx], out_dir_name, *parts[idx + 1 :])


def main() -> None:
    parser = argparse.ArgumentParser(
        "Merge audio latents and text embeddings into single .pt files",
        add_help=True,
    )
    parser.add_argument(
        "--manifest-paths",
        action="append",
        default=["/share/users/student/f/friverossego/datasets/audio_manifest_train.jsonl"],
        help="JSONL manifest paths (repeatable or comma-separated)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/share/users/student/f/friverossego/datasets",
        help="Base directory for relative manifest entries",
    )
    parser.add_argument(
        "--out-dir-name",
        type=str,
        default="latents",
        help="Output directory name to replace 'audio_latents'",
    )
    parser.add_argument(
        "--processes",
        default=None,
        type=int,
        help="Number of worker processes (default: CPU cores)",
    )
    parser.add_argument("--threads-per-process", default=1, type=int)
    parser.add_argument(
        "--mp-start-method",
        default="fork",
        choices=["fork", "spawn", "forkserver"],
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged files",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip entries with missing audio/text files",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip entries that raise errors during merge",
    )
    parser.add_argument("--log-dir", type=str, default=None)

    # Node sharding (optional)
    parser.add_argument("--node-rank", default=None, type=int)
    parser.add_argument("--node-world-size", default=None, type=int)
    args = parser.parse_args()

    manifest_paths = _parse_manifest_paths(args.manifest_paths)

    data_root = None
    if args.data_root is not None:
        data_root = Path(args.data_root).expanduser().absolute()

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

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = str(data_root or Path.cwd())

    tasks: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {manifest_path} at line {line_num}") from exc

                audio_rel = entry.get("audio_path")
                text_rel = entry.get("text_path")
                if not audio_rel or not text_rel:
                    raise ValueError(
                        f"Missing audio_path/text_path in {manifest_path} at line {line_num}"
                    )

                audio_abs = _resolve_path(str(audio_rel), data_root)
                text_abs = _resolve_path(str(text_rel), data_root)
                out_path = _default_out_path(audio_abs, args.out_dir_name)

                key = out_path.as_posix()
                if key in seen:
                    continue
                seen.add(key)

                shard_key = key
                if data_root is not None:
                    try:
                        shard_key = out_path.relative_to(data_root).as_posix()
                    except ValueError:
                        shard_key = key
                if not belongs_to_node(shard_key, node_rank, node_world_size):
                    continue

                tasks.append((str(audio_abs), str(text_abs), str(out_path)))

    if node_rank == 0:
        print(f"Manifests: {len(manifest_paths)}")
        print(f"Unique outputs: {len(seen)}")
    print(
        f"Node {node_rank}/{node_world_size}: {len(tasks)} items to process. "
        f"Output dir name: {args.out_dir_name}"
    )

    if not tasks:
        return

    processes = args.processes or (os.cpu_count() or 1)
    run_pool(
        tasks,
        _process_item,
        _init_worker,
        (
            args.skip_missing,
            args.skip_errors,
            args.overwrite,
            log_dir,
            node_rank,
        ),
        num_workers=processes,
        threads_per_worker=args.threads_per_process,
        mp_start_method=args.mp_start_method,
        desc="Merging",
    )


if __name__ == "__main__":
    main()
