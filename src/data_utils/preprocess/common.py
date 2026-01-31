"""
Shared utilities for preprocessing scripts.

Includes:
- Node sharding (resolve_node_setting, belongs_to_node)
- Cache scanning (scan_cached_outputs)
- Atomic file writing (atomic_write_pt)
- Worker thread configuration (configure_worker_threads)
- Skip logging (SkipLogger)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch


def resolve_node_setting(
    value: int | None,
    env_keys: list[str],
    default: int | None = None,
) -> int | None:
    """
    Resolve node setting from explicit value, environment variables, or default.

    Args:
        value: Explicit value if provided via CLI
        env_keys: Environment variable names to check (e.g., ["NODE_RANK", "SLURM_NODEID"])
        default: Default value if not found

    Returns:
        Resolved integer value or default
    """
    if value is not None:
        return value
    for key in env_keys:
        env_val = os.environ.get(key)
        if env_val is None:
            continue
        try:
            return int(env_val)
        except ValueError:
            continue
    return default


def belongs_to_node(key: str, node_rank: int, node_world_size: int) -> bool:
    """
    Determine if a key belongs to this node using MD5-based consistent hashing.

    Args:
        key: String key (typically relative output path)
        node_rank: Current node's rank
        node_world_size: Total number of nodes

    Returns:
        True if this node should process this key
    """
    if node_world_size <= 1:
        return True
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % int(node_world_size)
    return bucket == int(node_rank)


def _fast_scandir(root: str, exts: set[str]) -> tuple[list[str], list[str]]:
    """
    Recursively scan directory for files with given extensions.

    Args:
        root: Root directory to scan
        exts: Set of extensions (with leading dot)

    Returns:
        (subdirs, files) tuple
    """
    exts = {e if e.startswith(".") else f".{e}" for e in exts}
    subdirs, files = [], []
    try:
        for entry in os.scandir(root):
            try:
                if entry.is_dir():
                    subdirs.append(entry.path)
                elif entry.is_file():
                    name = entry.name
                    if not name.startswith(".") and Path(name).suffix.lower() in exts:
                        files.append(entry.path)
            except (OSError, PermissionError):
                pass
    except (OSError, PermissionError):
        pass
    for d in list(subdirs):
        sd, f = _fast_scandir(d, exts)
        subdirs.extend(sd)
        files.extend(f)
    return subdirs, files


def scan_cached_outputs(out_root: Path | str, extension: str = ".pt") -> set[str]:
    """
    Scan output directory for already-completed cache files.

    Args:
        out_root: Root directory to scan
        extension: File extension to look for (default: ".pt")

    Returns:
        Set of relative POSIX paths for completed files
    """
    out_root = Path(out_root).resolve()
    _, files = _fast_scandir(str(out_root), {extension})
    done = set()
    for path in files:
        try:
            rel = Path(path).resolve().relative_to(out_root)
        except ValueError:
            continue
        done.add(rel.as_posix())
    return done


def atomic_write_pt(
    out_path: Path | str,
    payload: dict[str, Any],
    tmp_suffix: str | None = None,
) -> None:
    """
    Atomically write a PyTorch payload to disk using tmp file + rename.

    Args:
        out_path: Final output path
        payload: Dictionary to save via torch.save
        tmp_suffix: Optional suffix for temp file (default: uses pid)
    """
    out_path = Path(out_path)
    if tmp_suffix is None:
        tmp_suffix = f".{os.getpid()}.tmp"
    tmp_path = str(out_path) + tmp_suffix

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def configure_worker_threads(num_threads: int) -> None:
    """
    Configure CPU thread settings for a worker process.

    Sets:
        - torch.set_num_threads()
        - torch.set_num_interop_threads(1)

    Args:
        num_threads: Number of threads for intra-op parallelism
    """
    torch.set_num_threads(int(num_threads))
    torch.set_num_interop_threads(1)


class SkipLogger:
    """
    Thread-safe logger for skipped files and processing events.

    Writes to:
        - {log_dir}/skipped_files.node{node_rank}.log
        - {log_dir}/processing_events.node{node_rank}.log
    """

    def __init__(self, log_dir: Path | str, node_rank: int = 0):
        """
        Initialize skip logger.

        Args:
            log_dir: Directory to write log files
            node_rank: Node rank for log file naming
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.skip_log_path = log_dir / f"skipped_files.node{int(node_rank)}.log"
        self.event_log_path = log_dir / f"processing_events.node{int(node_rank)}.log"

    def log_skip(self, item_id: str, reason: str) -> None:
        """
        Log a skipped item.

        Args:
            item_id: Identifier of the skipped item
            reason: Reason for skipping
        """
        msg = f"{item_id}\t{reason}\n"
        try:
            with open(self.skip_log_path, "a", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass

    def log_event(self, message: str) -> None:
        """
        Log a processing event.

        Args:
            message: Event message
        """
        try:
            with open(self.event_log_path, "a", encoding="utf-8") as f:
                f.write(f"{message}\n")
        except Exception:
            pass
