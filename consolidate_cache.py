#!/usr/bin/env python3
"""
Consolidate one subset's latents_bf16/*.pt files into a single ragged cache .pt.

Expected subset layout:
  <subset_path>/
    manifest.jsonl
    latents_bf16/**/*.pt

The consolidated file stores variable-length tensors via concatenation + offsets.
Use --confirm to re-load source files and verify exact tensor equality by slice.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import torch
from tqdm import tqdm


POSTERIOR_DIM = 256
CLAP_DIM = 512
T5_DIM = 1024

REQUIRED_KEYS = {
    "posterior_params",
    "latent_length",
    "clap_embedding",
    "t5_last_hidden",
    "t5_len",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Consolidate one subset's bf16 latents into a single .pt cache")
    parser.add_argument(
        "--subset-path",
        type=str,
        required=True,
        help="Subset root (e.g., /.../datasets/WavCaps/FreeSound)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output cache path (default: <subset-path>/consolidated_latents_bf16.pt)",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.jsonl",
        help="Manifest filename inside subset path",
    )
    parser.add_argument(
        "--latents-dir-name",
        type=str,
        default="latents_bf16",
        help="Latents directory inside subset path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(128, os.cpu_count() or 1),
        help="Thread pool size",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Max queued futures (default: workers * 4)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="After save, confirm each slice equals original source tensors",
    )
    return parser.parse_args()


def _iter_parallel(
    tasks: list[tuple[int, Path]],
    worker_fn,
    *,
    workers: int,
    max_in_flight: int,
    desc: str,
):
    total = len(tasks)
    if total == 0:
        return

    if workers <= 1:
        for task in tqdm(tasks, total=total, desc=desc):
            yield worker_fn(task)
        return

    if max_in_flight <= 0:
        max_in_flight = max(1, workers * 4)

    task_iter = iter(tasks)
    futures = set()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in range(max_in_flight):
            try:
                task = next(task_iter)
            except StopIteration:
                break
            futures.add(ex.submit(worker_fn, task))

        with tqdm(total=total, desc=desc) as pbar:
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    yield fut.result()
                    pbar.update(1)
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        continue
                    futures.add(ex.submit(worker_fn, task))


def _load_manifest_keys(manifest_path: Path) -> list[str]:
    keys: list[str] = []
    seen = set()

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                entry = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path} line {line_num}") from exc
            key = entry.get("key")
            if key is None:
                raise ValueError(f"Missing 'key' in {manifest_path} line {line_num}")
            key = str(key)
            if key in seen:
                raise ValueError(f"Duplicate key '{key}' in {manifest_path}")
            seen.add(key)
            keys.append(key)

    if not keys:
        raise ValueError(f"No entries found in {manifest_path}")
    return keys


def _index_latent_files(latents_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for file_path in latents_root.rglob("*.pt"):
        key = file_path.stem
        if key in index:
            raise ValueError(
                f"Duplicate latent filename stem '{key}' found:\n"
                f"  {index[key]}\n"
                f"  {file_path}"
            )
        index[key] = file_path
    return index


def _resolve_paths(keys: list[str], file_index: dict[str, Path]) -> list[Path]:
    paths: list[Path] = []
    missing = []
    for key in keys:
        path = file_index.get(key)
        if path is None:
            missing.append(key)
            continue
        paths.append(path)

    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... ({len(missing)} missing total)"
        raise FileNotFoundError(f"Missing bf16 latent files for keys: {preview}{suffix}")
    return paths


def _validate_payload(data: dict, path: Path) -> tuple[int, int]:
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise KeyError(f"{path} missing required keys: {sorted(missing)}")

    posterior = data["posterior_params"]
    clap = data["clap_embedding"]
    t5 = data["t5_last_hidden"]
    latent_length = int(data["latent_length"])
    t5_len = int(data["t5_len"])

    if posterior.ndim != 2:
        raise ValueError(f"{path}: posterior_params must be 2D, got {tuple(posterior.shape)}")
    if posterior.shape[0] != POSTERIOR_DIM:
        raise ValueError(f"{path}: posterior_params[0] expected {POSTERIOR_DIM}, got {posterior.shape[0]}")
    if latent_length < 0 or latent_length > posterior.shape[1]:
        raise ValueError(
            f"{path}: latent_length={latent_length} out of bounds for posterior T={posterior.shape[1]}"
        )

    if clap.ndim != 1 or clap.shape[0] != CLAP_DIM:
        raise ValueError(f"{path}: clap_embedding expected [{CLAP_DIM}], got {tuple(clap.shape)}")

    if t5.ndim != 2:
        raise ValueError(f"{path}: t5_last_hidden must be 2D, got {tuple(t5.shape)}")
    if t5.shape[1] != T5_DIM:
        raise ValueError(f"{path}: t5_last_hidden[1] expected {T5_DIM}, got {t5.shape[1]}")
    if t5_len < 0 or t5_len > t5.shape[0]:
        raise ValueError(f"{path}: t5_len={t5_len} out of bounds for t5 rows={t5.shape[0]}")

    return latent_length, t5_len


def _meta_worker(task: tuple[int, Path]) -> tuple[int, int, int]:
    idx, path = task
    data = torch.load(path, map_location="cpu", weights_only=True)
    latent_length, t5_len = _validate_payload(data, path)
    return idx, latent_length, t5_len


def _sample_worker(task: tuple[int, Path]) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx, path = task
    data = torch.load(path, map_location="cpu", weights_only=True)
    latent_length, t5_len = _validate_payload(data, path)

    posterior = data["posterior_params"][:, :latent_length].transpose(0, 1).contiguous().to(torch.bfloat16)
    clap = data["clap_embedding"].contiguous().to(torch.bfloat16)
    t5 = data["t5_last_hidden"][:t5_len].contiguous().to(torch.bfloat16)
    return idx, posterior, clap, t5


def _build_offsets(lengths: list[int]) -> list[int]:
    offsets = [0]
    running = 0
    for value in lengths:
        running += int(value)
        offsets.append(running)
    return offsets


def _consolidate(
    *,
    keys: list[str],
    paths: list[Path],
    workers: int,
    max_in_flight: int,
) -> dict:
    num_items = len(keys)
    tasks = list(enumerate(paths))

    latent_lengths = [-1] * num_items
    t5_lens = [-1] * num_items
    for idx, latent_length, t5_len in _iter_parallel(
        tasks,
        _meta_worker,
        workers=workers,
        max_in_flight=max_in_flight,
        desc="Metadata pass",
    ):
        latent_lengths[idx] = latent_length
        t5_lens[idx] = t5_len

    if any(v < 0 for v in latent_lengths):
        raise RuntimeError("Metadata pass incomplete: latent_lengths not fully populated")
    if any(v < 0 for v in t5_lens):
        raise RuntimeError("Metadata pass incomplete: t5_lens not fully populated")

    posterior_offsets = _build_offsets(latent_lengths)
    t5_offsets = _build_offsets(t5_lens)

    posterior_cat = torch.empty((posterior_offsets[-1], POSTERIOR_DIM), dtype=torch.bfloat16)
    clap = torch.empty((num_items, CLAP_DIM), dtype=torch.bfloat16)
    t5_cat = torch.empty((t5_offsets[-1], T5_DIM), dtype=torch.bfloat16)

    for idx, posterior, clap_i, t5_i in _iter_parallel(
        tasks,
        _sample_worker,
        workers=workers,
        max_in_flight=max_in_flight,
        desc="Fill pass",
    ):
        p_start = posterior_offsets[idx]
        p_end = posterior_offsets[idx + 1]
        t_start = t5_offsets[idx]
        t_end = t5_offsets[idx + 1]

        if posterior.shape != (p_end - p_start, POSTERIOR_DIM):
            raise ValueError(f"Posterior shape mismatch at idx={idx}: got {tuple(posterior.shape)}")
        if clap_i.shape != (CLAP_DIM,):
            raise ValueError(f"CLAP shape mismatch at idx={idx}: got {tuple(clap_i.shape)}")
        if t5_i.shape != (t_end - t_start, T5_DIM):
            raise ValueError(f"T5 shape mismatch at idx={idx}: got {tuple(t5_i.shape)}")

        posterior_cat[p_start:p_end] = posterior
        clap[idx] = clap_i
        t5_cat[t_start:t_end] = t5_i

    payload = {
        "version": 1,
        "num_items": num_items,
        "keys": keys,
        "posterior_cat": posterior_cat,
        "posterior_offsets": torch.tensor(posterior_offsets, dtype=torch.int64),
        "latent_lengths": torch.tensor(latent_lengths, dtype=torch.int32),
        "clap": clap,
        "t5_cat": t5_cat,
        "t5_offsets": torch.tensor(t5_offsets, dtype=torch.int64),
        "t5_lens": torch.tensor(t5_lens, dtype=torch.int32),
    }
    return payload


def _confirm_payload(
    *,
    payload: dict,
    paths: list[Path],
    workers: int,
    max_in_flight: int,
) -> None:
    posterior_cat = payload["posterior_cat"]
    posterior_offsets = payload["posterior_offsets"].tolist()
    clap = payload["clap"]
    t5_cat = payload["t5_cat"]
    t5_offsets = payload["t5_offsets"].tolist()
    num_items = int(payload["num_items"])

    if len(paths) != num_items:
        raise ValueError("Confirm error: number of source paths does not match payload num_items")

    tasks = list(enumerate(paths))
    mismatch_count = 0

    for idx, posterior, clap_i, t5_i in _iter_parallel(
        tasks,
        _sample_worker,
        workers=workers,
        max_in_flight=max_in_flight,
        desc="Confirm pass",
    ):
        p_start = posterior_offsets[idx]
        p_end = posterior_offsets[idx + 1]
        t_start = t5_offsets[idx]
        t_end = t5_offsets[idx + 1]

        if not torch.equal(posterior, posterior_cat[p_start:p_end]):
            mismatch_count += 1
            if mismatch_count <= 3:
                print(f"[mismatch] posterior idx={idx}")
        if not torch.equal(clap_i, clap[idx]):
            mismatch_count += 1
            if mismatch_count <= 3:
                print(f"[mismatch] clap idx={idx}")
        if not torch.equal(t5_i, t5_cat[t_start:t_end]):
            mismatch_count += 1
            if mismatch_count <= 3:
                print(f"[mismatch] t5 idx={idx}")

    if mismatch_count > 0:
        raise RuntimeError(f"Confirm failed with {mismatch_count} mismatched tensor comparisons")
    print("PASS: consolidated slices are identical to source bf16 tensors.")


def main() -> None:
    args = parse_args()

    subset_path = Path(args.subset_path).expanduser().resolve()
    if not subset_path.exists():
        raise FileNotFoundError(f"Missing subset path: {subset_path}")

    manifest_path = subset_path / args.manifest_name
    latents_root = subset_path / args.latents_dir_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not latents_root.exists():
        raise FileNotFoundError(f"Missing latents root: {latents_root}")

    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else subset_path / "consolidated_latents_bf16.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers))
    max_in_flight = int(args.max_in_flight)

    print(f"[subset]   {subset_path}")
    print(f"[manifest] {manifest_path}")
    print(f"[latents]  {latents_root}")
    print(f"[output]   {output_path}")
    print(f"[workers]  {workers}")

    keys = _load_manifest_keys(manifest_path)
    file_index = _index_latent_files(latents_root)
    paths = _resolve_paths(keys, file_index)
    print(f"[items]    {len(keys):,}")

    payload = _consolidate(
        keys=keys,
        paths=paths,
        workers=workers,
        max_in_flight=max_in_flight,
    )
    payload["subset_path"] = subset_path.as_posix()
    payload["manifest_path"] = manifest_path.as_posix()
    payload["latents_root"] = latents_root.as_posix()

    print("Saving consolidated cache...")
    torch.save(payload, output_path)
    print("Saved.")

    if args.confirm:
        print("Reloading saved cache for confirmation...")
        reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
        _confirm_payload(
            payload=reloaded,
            paths=paths,
            workers=workers,
            max_in_flight=max_in_flight,
        )

    print("Done.")


if __name__ == "__main__":
    main()
