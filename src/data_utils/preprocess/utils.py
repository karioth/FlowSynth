"""
Utilities for unified preprocessing (AudioCaps + WavCaps).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from src.data_utils.preprocess import (
    AUDIO_EXTS,
    SkipLogger,
    TextEncoder,
    apply_duration_filter,
    atomic_write_pt,
    belongs_to_node,
    load_audio_from_hf_example,
    load_audio_from_path,
    resolve_hf_uid,
    resolve_node_setting,
    run_gpu_pool,
    run_pool,
    scan_cached_outputs,
)
from src.data_utils.utils import encode_audio_latents

DEFAULT_WAVCAPS_SUBSETS = "AudioSet_SL,BBC_Sound_Effects,FreeSound,SoundBible"
DEFAULT_AUDIOCAPS_SPLITS = "train"
DEFAULT_AUDIOCAPS_DATASET = "OpenSound/AudioCaps"
DEFAULT_MERGED_MANIFEST = "audio_manifest_train.jsonl"

SUBSET_JSON = {
    "AudioSet_SL": "AudioSet_SL/as_final.json",
    "BBC_Sound_Effects": "BBC_Sound_Effects/bbc_final.json",
    "FreeSound": "FreeSound/fsd_final.json",
    "SoundBible": "SoundBible/sb_final.json",
}

# Global state for workers
_MODEL = None
_ENCODER = None
_WORKER_STATE: dict[str, object] = {}
_SKIP_LOGGER = None


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    cleaned = value.strip()
    if cleaned == "" or cleaned.lower() in {"none", "null"}:
        return []
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def resolve_data_dir(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned == "" or cleaned.lower() in {"none", "null"}:
        return None
    return value


def resolve_dacvae_weights(weights_ref: str) -> str:
    if os.path.exists(weights_ref):
        return weights_ref
    if weights_ref.startswith("facebook/"):
        return weights_ref
    raise FileNotFoundError(f"Missing weights: {weights_ref}")


def resolve_node_settings(node_rank: int | None, node_world_size: int | None) -> tuple[int, int]:
    resolved_rank = resolve_node_setting(node_rank, ["NODE_RANK", "SLURM_NODEID"], default=None)
    resolved_world = resolve_node_setting(
        node_world_size, ["NODE_WORLD_SIZE", "SLURM_JOB_NUM_NODES"], default=1
    )
    if resolved_world > 1 and resolved_rank is None:
        raise ValueError("node_rank is required when node_world_size > 1")
    if resolved_rank is None:
        resolved_rank = 0
    if resolved_rank < 0 or resolved_rank >= resolved_world:
        raise ValueError("node_rank must be in [0, node_world_size)")
    return resolved_rank, resolved_world


def _ensure_pt(name: str) -> str:
    if name.endswith(".pt"):
        return name
    return f"{name}.pt"


def _hash_prefix(value: str, prefix_len: int) -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return digest[:prefix_len]


def _latent_rel_path(key: str, hash_prefix_len: int) -> Path:
    filename = _ensure_pt(key)
    if hash_prefix_len > 0:
        bucket = _hash_prefix(key, hash_prefix_len)
        return Path(bucket) / filename
    return Path(filename)


def _latent_out_path(latents_root: Path, key: str, hash_prefix_len: int) -> Path:
    return latents_root / _latent_rel_path(key, hash_prefix_len)


def _normalize_key(subset: str, source_id: str) -> str:
    if subset == "AudioSet_SL" and source_id.endswith(".wav"):
        return source_id[:-4]
    return source_id


def _iter_wavcaps_entries(subset: str, json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("data", [])
    for item in items:
        source_id = str(item["id"])
        key = _normalize_key(subset, source_id)
        caption = item["caption"]
        duration = item["duration"]
        yield {"key": key, "caption": caption, "duration": duration}


def _iter_audiocaps_entries(
    *,
    hf_dataset: str,
    split: str,
    data_dir: str | None,
    cache_dir: str | None,
    key_column: str,
    caption_column: str,
    audio_length_column: str,
    audio_column: str,
    dedupe: str,
):
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, data_dir=data_dir, split=split, cache_dir=cache_dir)

    if key_column not in ds.column_names:
        raise ValueError(f"Missing key column '{key_column}'. Have: {ds.column_names}")
    if caption_column not in ds.column_names:
        raise ValueError(f"Missing caption column '{caption_column}'. Have: {ds.column_names}")

    sr = None
    if audio_column in ds.features and hasattr(ds.features[audio_column], "sampling_rate"):
        sr = ds.features[audio_column].sampling_rate
    have_duration = (audio_length_column in ds.column_names) and (sr is not None)

    keys = ds[key_column]
    caps = ds[caption_column]
    lens = ds[audio_length_column] if have_duration else None

    seen = set()
    for i in range(len(ds)):
        k_raw = keys[i]
        k = str(k_raw)
        if k in seen:
            if dedupe == "first":
                continue
            raise ValueError(f"Duplicate key '{k}' (idx={i}) in split={split}.")
        seen.add(k)

        entry = {"key": k, "caption": caps[i]}
        if have_duration:
            entry["duration"] = float(lens[i]) / float(sr)
        yield entry


def _write_manifest(out_path: Path, entries, *, overwrite: bool) -> bool:
    if out_path.exists() and not overwrite:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=True))
            f.write("\n")
    return True


def _iter_manifest_entries(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path} at line {line_num}") from exc
            key = entry.get("key")
            if key is None:
                raise ValueError(f"Missing 'key' in {manifest_path} at line {line_num}")
            yield str(key), entry


def _resolve_wavcaps_audio_path(
    audio_root: Path,
    subset: str,
    key: str,
) -> tuple[Path | None, str | None]:
    subset_root = audio_root / subset
    if Path(key).suffix:
        candidate = subset_root / key
        if candidate.is_file():
            return candidate, None
        return None, "missing_audio"

    matches = []
    for ext in AUDIO_EXTS:
        candidate = subset_root / f"{key}{ext}"
        if candidate.is_file():
            matches.append(candidate)

    if len(matches) == 1:
        return matches[0], None
    if len(matches) == 0:
        return None, "missing_audio"
    return None, f"ambiguous_audio({len(matches)})"


def _init_worker_wavcaps(
    weights_path: str,
    device: str,
    wavcaps_audio_root: str,
    subset: str,
    latents_root: str,
    min_duration_seconds: float,
    max_duration_seconds: float,
    chunk_size_latents: int,
    overlap_latents: int,
    hash_prefix_len: int,
    clap_model: str,
    t5_model: str,
    node_rank: int,
):
    global _MODEL, _ENCODER, _WORKER_STATE, _SKIP_LOGGER

    import dacvae

    _MODEL = dacvae.DACVAE.load(weights_path).eval().to(device)
    _ENCODER = TextEncoder(
        clap_model_name=clap_model,
        t5_model_name=t5_model,
        device=device,
    )

    latents_root_path = Path(latents_root)
    log_dir = latents_root_path.parent
    _SKIP_LOGGER = SkipLogger(log_dir, node_rank)

    _WORKER_STATE = {
        "wavcaps_audio_root": Path(wavcaps_audio_root),
        "subset": subset,
        "latents_root": latents_root_path,
        "min_duration_seconds": float(min_duration_seconds),
        "max_duration_seconds": float(max_duration_seconds),
        "chunk_size_latents": int(chunk_size_latents),
        "overlap_latents": int(overlap_latents),
        "hash_prefix_len": int(hash_prefix_len),
    }


def _init_worker_audiocaps(
    weights_path: str,
    device: str,
    hf_dataset: str,
    split: str,
    data_dir: str | None,
    cache_dir: str | None,
    latents_root: str,
    key_column: str,
    caption_column: str,
    audio_column: str,
    min_duration_seconds: float,
    max_duration_seconds: float,
    chunk_size_latents: int,
    overlap_latents: int,
    hash_prefix_len: int,
    clap_model: str,
    t5_model: str,
    node_rank: int,
):
    global _MODEL, _ENCODER, _WORKER_STATE, _SKIP_LOGGER

    import dacvae
    from datasets import load_dataset

    _MODEL = dacvae.DACVAE.load(weights_path).eval().to(device)
    _ENCODER = TextEncoder(
        clap_model_name=clap_model,
        t5_model_name=t5_model,
        device=device,
    )

    ds = load_dataset(hf_dataset, data_dir=data_dir, split=split, cache_dir=cache_dir)

    if key_column not in ds.column_names:
        raise ValueError(f"Missing key column '{key_column}'. Have: {ds.column_names}")
    if caption_column not in ds.column_names:
        raise ValueError(f"Missing caption column '{caption_column}'. Have: {ds.column_names}")
    if audio_column not in ds.column_names:
        raise ValueError(f"Missing audio column '{audio_column}'. Have: {ds.column_names}")

    latents_root_path = Path(latents_root)
    log_dir = latents_root_path.parent
    _SKIP_LOGGER = SkipLogger(log_dir, node_rank)

    _WORKER_STATE = {
        "dataset": ds,
        "latents_root": latents_root_path,
        "key_column": key_column,
        "caption_column": caption_column,
        "audio_column": audio_column,
        "min_duration_seconds": float(min_duration_seconds),
        "max_duration_seconds": float(max_duration_seconds),
        "chunk_size_latents": int(chunk_size_latents),
        "overlap_latents": int(overlap_latents),
        "hash_prefix_len": int(hash_prefix_len),
    }


def _process_wavcaps_item(item: tuple[str, str]) -> None:
    from audiotools import AudioSignal

    key, caption = item
    out_path = _latent_out_path(
        _WORKER_STATE["latents_root"],
        key,
        _WORKER_STATE["hash_prefix_len"],
    )
    if out_path.exists():
        return

    audio_path, reason = _resolve_wavcaps_audio_path(
        _WORKER_STATE["wavcaps_audio_root"],
        _WORKER_STATE["subset"],
        key,
    )
    if audio_path is None:
        _SKIP_LOGGER.log_skip(key, reason or "missing_audio")
        return

    try:
        wav, sr = load_audio_from_path(audio_path)
        wav, skip_reason = apply_duration_filter(
            wav,
            sr,
            _WORKER_STATE["min_duration_seconds"],
            _WORKER_STATE["max_duration_seconds"],
        )
        if wav is None:
            _SKIP_LOGGER.log_skip(key, skip_reason or "filtered")
            return

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

        text_payload = _ENCODER.encode(caption)
        payload = {
            "posterior_params": tensor,
            "latent_length": latent_length,
            "clap_embedding": text_payload["clap_embedding"],
            "t5_last_hidden": text_payload["t5_last_hidden"],
            "t5_len": int(text_payload["t5_len"]),
        }
        atomic_write_pt(out_path, payload)
    except Exception as exc:
        _SKIP_LOGGER.log_skip(key, f"{type(exc).__name__}: {exc}")


def _process_audiocaps_item(item: tuple[int, str, str]) -> None:
    from audiotools import AudioSignal

    ds = _WORKER_STATE["dataset"]
    audio_column = _WORKER_STATE["audio_column"]

    try:
        idx, key, caption = item
        ex = ds[int(idx)]
        out_path = _latent_out_path(
            _WORKER_STATE["latents_root"],
            key,
            _WORKER_STATE["hash_prefix_len"],
        )
        if out_path.exists():
            return

        if audio_column not in ex:
            raise ValueError(f"Missing audio column '{audio_column}' in index {idx}")
        wav, sr = load_audio_from_hf_example({"audio": ex[audio_column]})

        wav, skip_reason = apply_duration_filter(
            wav,
            sr,
            _WORKER_STATE["min_duration_seconds"],
            _WORKER_STATE["max_duration_seconds"],
        )
        if wav is None:
            _SKIP_LOGGER.log_skip(key, skip_reason or "filtered")
            return

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

        text_payload = _ENCODER.encode(str(caption))
        payload = {
            "posterior_params": tensor,
            "latent_length": latent_length,
            "clap_embedding": text_payload["clap_embedding"],
            "t5_last_hidden": text_payload["t5_last_hidden"],
            "t5_len": int(text_payload["t5_len"]),
        }
        atomic_write_pt(out_path, payload)
    except Exception as exc:
        _SKIP_LOGGER.log_skip(str(item[1]), f"{type(exc).__name__}: {exc}")


def build_wavcaps_manifests(
    *,
    data_root: Path,
    json_root: Path | None,
    subsets: list[str],
    overwrite: bool,
) -> None:
    for subset in subsets:
        if subset not in SUBSET_JSON:
            raise ValueError(f"Unknown subset: {subset}. Expected one of {sorted(SUBSET_JSON)}")
        out_path = data_root / "WavCaps" / subset / "manifest.jsonl"
        if out_path.exists() and not overwrite:
            print(f"[skip] WavCaps/{subset}: manifest exists")
            continue

        if json_root is None:
            raise ValueError("--wavcaps-json-root is required to build WavCaps manifests")

        json_path = json_root / SUBSET_JSON[subset]
        if not json_path.exists():
            raise FileNotFoundError(f"Missing JSON for {subset}: {json_path}")

        entries = list(_iter_wavcaps_entries(subset, json_path))
        wrote = _write_manifest(out_path, entries, overwrite=True)
        if wrote:
            print(f"[wrote] WavCaps/{subset}: {len(entries):,} entries -> {out_path}")


def build_audiocaps_manifests(
    *,
    data_root: Path,
    splits: list[str],
    hf_dataset: str,
    data_dir: str | None,
    cache_dir: str | None,
    key_column: str,
    caption_column: str,
    audio_length_column: str,
    audio_column: str,
    dedupe: str,
    overwrite: bool,
) -> None:
    for split in splits:
        out_path = data_root / "AudioCaps" / split / "manifest.jsonl"
        if out_path.exists() and not overwrite:
            print(f"[skip] AudioCaps/{split}: manifest exists")
            continue

        entries = list(
            _iter_audiocaps_entries(
                hf_dataset=hf_dataset,
                split=split,
                data_dir=data_dir,
                cache_dir=cache_dir,
                key_column=key_column,
                caption_column=caption_column,
                audio_length_column=audio_length_column,
                audio_column=audio_column,
                dedupe=dedupe,
            )
        )
        wrote = _write_manifest(out_path, entries, overwrite=True)
        if wrote:
            print(f"[wrote] AudioCaps/{split}: {len(entries):,} entries -> {out_path}")


def encode_wavcaps_subset(
    *,
    subset: str,
    manifest_path: Path,
    wavcaps_audio_root: Path,
    latents_root: Path,
    weights_path: str,
    clap_model: str,
    t5_model: str,
    min_duration_seconds: float,
    max_duration_seconds: float,
    chunk_size_latents: int,
    overlap_latents: int,
    hash_prefix_len: int,
    device: str,
    processes: int | None,
    threads_per_process: int,
    mp_start_method: str,
    node_rank: int,
    node_world_size: int,
    deterministic: bool,
) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    latents_root.mkdir(parents=True, exist_ok=True)
    done_cache = scan_cached_outputs(latents_root)
    tasks: list[tuple[str, str]] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            key = entry.get("key")
            caption = entry.get("caption")
            if key is None or caption is None:
                raise ValueError(f"Missing key/caption in manifest: {manifest_path}")
            key = str(key)

            out_rel = _latent_rel_path(key, hash_prefix_len).as_posix()
            if out_rel in done_cache:
                continue
            if not belongs_to_node(out_rel, node_rank, node_world_size):
                continue
            tasks.append((key, str(caption)))

    print(
        f"WavCaps/{subset}: {len(tasks)} items to process. "
        f"Cache output: {latents_root}"
    )
    if not tasks:
        return

    init_args = (
        weights_path,
        device,
        str(wavcaps_audio_root),
        subset,
        str(latents_root),
        min_duration_seconds,
        max_duration_seconds,
        chunk_size_latents,
        overlap_latents,
        hash_prefix_len,
        clap_model,
        t5_model,
        node_rank,
    )

    def init_args_fn(gpu_id: int) -> tuple:
        return (
            weights_path,
            f"cuda:{gpu_id}",
            str(wavcaps_audio_root),
            subset,
            str(latents_root),
            min_duration_seconds,
            max_duration_seconds,
            chunk_size_latents,
            overlap_latents,
            hash_prefix_len,
            clap_model,
            t5_model,
            node_rank,
        )

    if device == "cpu":
        worker_count = processes or (os.cpu_count() or 1)
        run_pool(
            tasks,
            _process_wavcaps_item,
            _init_worker_wavcaps,
            init_args,
            num_workers=worker_count,
            threads_per_worker=threads_per_process,
            mp_start_method=mp_start_method,
            desc=f"Preprocessing WavCaps/{subset}",
        )
    else:
        worker_count = processes or torch.cuda.device_count()
        run_gpu_pool(
            tasks,
            _process_wavcaps_item,
            _init_worker_wavcaps,
            init_args_fn,
            num_gpus=worker_count,
            threads_per_gpu=threads_per_process,
            deterministic=deterministic,
            desc=f"Preprocessing WavCaps/{subset}",
        )


def encode_audiocaps_split(
    *,
    split: str,
    manifest_path: Path,
    latents_root: Path,
    weights_path: str,
    clap_model: str,
    t5_model: str,
    min_duration_seconds: float,
    max_duration_seconds: float,
    chunk_size_latents: int,
    overlap_latents: int,
    hash_prefix_len: int,
    device: str,
    processes: int | None,
    threads_per_process: int,
    mp_start_method: str,
    node_rank: int,
    node_world_size: int,
    deterministic: bool,
    hf_dataset: str,
    data_dir: str | None,
    cache_dir: str | None,
    key_column: str,
    caption_column: str,
    audio_column: str,
) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    from datasets import load_dataset

    latents_root.mkdir(parents=True, exist_ok=True)
    done_cache = scan_cached_outputs(latents_root)

    ds = load_dataset(hf_dataset, data_dir=data_dir, split=split, cache_dir=cache_dir)
    if key_column not in ds.column_names:
        raise ValueError(f"Missing key column '{key_column}'. Have: {ds.column_names}")
    if caption_column not in ds.column_names:
        raise ValueError(f"Missing caption column '{caption_column}'. Have: {ds.column_names}")
    if audio_column not in ds.column_names:
        raise ValueError(f"Missing audio column '{audio_column}'. Have: {ds.column_names}")

    total = len(ds)
    key_ds = ds.select_columns([key_column])
    key_to_idx: dict[str, int] = {}
    for idx in range(total):
        key_value = key_ds[idx][key_column]
        if key_value is None:
            raise ValueError(f"Missing key column '{key_column}' at index {idx}")
        key, _ = resolve_hf_uid(idx, str(key_value))
        if key not in key_to_idx:
            key_to_idx[key] = idx

    tasks: list[tuple[int, str, str]] = []
    seen_keys: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {manifest_path} at line {line_num}"
                ) from exc
            key = entry.get("key")
            caption = entry.get("caption")
            if key is None or caption is None:
                raise ValueError(f"Missing key/caption in manifest: {manifest_path}")
            key = str(key)
            if key in seen_keys:
                raise ValueError(f"Duplicate key '{key}' in manifest: {manifest_path}")
            seen_keys.add(key)
            if key not in key_to_idx:
                raise ValueError(
                    f"Key '{key}' not found in {hf_dataset} ({split})"
                )

            out_rel = _latent_rel_path(key, hash_prefix_len).as_posix()
            if out_rel in done_cache:
                continue
            if not belongs_to_node(out_rel, node_rank, node_world_size):
                continue
            tasks.append((key_to_idx[key], key, str(caption)))

    print(
        f"AudioCaps/{split}: {len(tasks)} items to process. "
        f"Cache output: {latents_root}"
    )
    if not tasks:
        return

    init_args = (
        weights_path,
        device,
        hf_dataset,
        split,
        data_dir,
        cache_dir,
        str(latents_root),
        key_column,
        caption_column,
        audio_column,
        min_duration_seconds,
        max_duration_seconds,
        chunk_size_latents,
        overlap_latents,
        hash_prefix_len,
        clap_model,
        t5_model,
        node_rank,
    )

    def init_args_fn(gpu_id: int) -> tuple:
        return (
            weights_path,
            f"cuda:{gpu_id}",
            hf_dataset,
            split,
            data_dir,
            cache_dir,
            str(latents_root),
            key_column,
            caption_column,
            audio_column,
            min_duration_seconds,
            max_duration_seconds,
            chunk_size_latents,
            overlap_latents,
            hash_prefix_len,
            clap_model,
            t5_model,
            node_rank,
        )

    if device == "cpu":
        worker_count = processes or (os.cpu_count() or 1)
        run_pool(
            tasks,
            _process_audiocaps_item,
            _init_worker_audiocaps,
            init_args,
            num_workers=worker_count,
            threads_per_worker=threads_per_process,
            mp_start_method=mp_start_method,
            desc=f"Preprocessing AudioCaps/{split}",
        )
    else:
        worker_count = processes or torch.cuda.device_count()
        run_gpu_pool(
            tasks,
            _process_audiocaps_item,
            _init_worker_audiocaps,
            init_args_fn,
            num_gpus=worker_count,
            threads_per_gpu=threads_per_process,
            deterministic=deterministic,
            desc=f"Preprocessing AudioCaps/{split}",
        )


def _write_source(
    *,
    out_file,
    rel_root: Path,
    manifest_path: Path,
    hash_prefix_len: int,
    source_label: str,
):
    total = 0
    for key, entry in _iter_manifest_entries(manifest_path):
        filename = _ensure_pt(key)
        if hash_prefix_len > 0:
            bucket = _hash_prefix(key, hash_prefix_len)
            latent_rel = rel_root / "latents" / bucket / filename
        else:
            latent_rel = rel_root / "latents" / filename

        out_entry = {
            "path": latent_rel.as_posix(),
            "source": source_label,
            "key": key,
        }
        if "caption" in entry:
            out_entry["caption"] = entry["caption"]
        if "duration" in entry:
            out_entry["duration"] = entry["duration"]

        out_file.write(json.dumps(out_entry, ensure_ascii=True))
        out_file.write("\n")
        total += 1

    return total


def _discover_manifest_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    dirs = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.jsonl"
        if manifest_path.exists():
            dirs.append(entry)
    dirs.sort(key=lambda p: p.name)
    return dirs


def merge_manifests(
    *,
    output_path: Path,
    data_root: Path,
    hash_prefix_len: int,
    wavcaps_subsets: list[str],
    audiocaps_splits: list[str],
    merge_all: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if merge_all:
        wavcaps_dirs = _discover_manifest_dirs(data_root / "WavCaps")
        audiocaps_dirs = _discover_manifest_dirs(data_root / "AudioCaps")
        wavcaps_subsets = [d.name for d in wavcaps_dirs]
        audiocaps_splits = [d.name for d in audiocaps_dirs]

    if not wavcaps_subsets and not audiocaps_splits:
        raise ValueError("No manifests found to merge")

    total_entries = 0
    with output_path.open("w", encoding="utf-8") as out_file:
        for subset in wavcaps_subsets:
            rel_root = Path("WavCaps") / subset
            manifest_path = data_root / rel_root / "manifest.jsonl"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing manifest: {manifest_path}")
            source_label = f"WavCaps/{subset}"
            count = _write_source(
                out_file=out_file,
                rel_root=rel_root,
                manifest_path=manifest_path,
                hash_prefix_len=hash_prefix_len,
                source_label=source_label,
            )
            print(f"[wrote] {source_label}: {count:,} entries")
            total_entries += count

        for split in audiocaps_splits:
            rel_root = Path("AudioCaps") / split
            manifest_path = data_root / rel_root / "manifest.jsonl"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing manifest: {manifest_path}")
            source_label = f"AudioCaps/{split}"
            count = _write_source(
                out_file=out_file,
                rel_root=rel_root,
                manifest_path=manifest_path,
                hash_prefix_len=hash_prefix_len,
                source_label=source_label,
            )
            print(f"[wrote] {source_label}: {count:,} entries")
            total_entries += count

    print(f"[done] {total_entries:,} entries written -> {output_path}")
