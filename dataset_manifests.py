#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


DEFAULT_WAVCAPS_OUT_ROOT = "/share/users/student/f/friverossego/datasets/WavCaps"
DEFAULT_AUDIOCAPS_OUT_ROOT = "/share/users/student/f/friverossego/datasets/AudioCaps"

SUBSET_JSON = {
    "AudioSet_SL": "AudioSet_SL/as_final.json",
    "BBC_Sound_Effects": "BBC_Sound_Effects/bbc_final.json",
    "FreeSound": "FreeSound/fsd_final.json",
    "SoundBible": "SoundBible/sb_final.json",
}


def _normalize_key(subset: str, source_id: str) -> str:
    if subset == "AudioSet_SL" and source_id.endswith(".wav"):
        return source_id[:-4]
    return source_id


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

    # duration is optional: audio_length / sampling_rate
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


def _iter_entries(subset: str, json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("data", [])
    for item in items:
        source_id = str(item["id"])
        key = _normalize_key(subset, source_id)
        caption = item["caption"]
        duration = item["duration"]
        yield {"key": key, "caption": caption, "duration": duration}


def _write_manifest(out_path: Path, entries, *, overwrite: bool):
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=True))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser("Build manifest.jsonl files (WavCaps + AudioCaps)", add_help=True)
    parser.add_argument("--dataset", choices=["wavcaps", "audiocaps"], default="wavcaps")
    parser.add_argument(
        "--json_root",
        type=str,
        default="/share/users/student/f/friverossego/raw/wavcaps_hf/json_files",
        help="Root folder containing subset JSONs",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=DEFAULT_WAVCAPS_OUT_ROOT,
        help="Root output folder for per-subset manifests",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Optional subset names to process (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing manifest.jsonl files",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report counts; do not write files",
    )

    # AudioCaps (HF) options
    parser.add_argument("--hf_dataset", type=str, default="OpenSound/AudioCaps")
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--hf_data_dir", type=str, default="data")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--key_column", type=str, default="audiocap_id")
    parser.add_argument("--caption_column", type=str, default="caption")
    parser.add_argument("--audio_length_column", type=str, default="audio_length")
    parser.add_argument("--audio_column", type=str, default="audio")
    parser.add_argument("--dedupe", choices=["error", "first"], default="error")
    args = parser.parse_args()

    json_root = Path(args.json_root).resolve()

    if args.dataset == "audiocaps":
        out_root = Path(args.out_root).resolve()
        if out_root == Path(DEFAULT_WAVCAPS_OUT_ROOT).resolve():
            out_root = Path(DEFAULT_AUDIOCAPS_OUT_ROOT).resolve()
        data_dir = args.hf_data_dir
        if isinstance(data_dir, str) and data_dir.strip().lower() in {"", "none", "null"}:
            data_dir = None
        cache_dir = args.hf_cache_dir
        if isinstance(cache_dir, str) and cache_dir.strip() == "":
            cache_dir = None

        out_path = out_root / args.hf_split / "manifest.jsonl"
        entries = list(
            _iter_audiocaps_entries(
                hf_dataset=args.hf_dataset,
                split=args.hf_split,
                data_dir=data_dir,
                cache_dir=cache_dir,
                key_column=args.key_column,
                caption_column=args.caption_column,
                audio_length_column=args.audio_length_column,
                audio_column=args.audio_column,
                dedupe=("first" if args.dedupe == "first" else "error"),
            )
        )
        if args.dry_run:
            print(f"[dry_run] AudioCaps/{args.hf_split}: {len(entries):,} entries -> {out_path}")
            return
        _write_manifest(out_path, entries, overwrite=args.overwrite)
        print(f"[wrote] AudioCaps/{args.hf_split}: {len(entries):,} entries -> {out_path}")
        return

    out_root = Path(args.out_root).resolve()

    if args.subsets:
        subsets = args.subsets
    else:
        subsets = list(SUBSET_JSON.keys())

    for subset in subsets:
        if subset not in SUBSET_JSON:
            raise ValueError(f"Unknown subset: {subset}. Expected one of {sorted(SUBSET_JSON)}")
        json_path = json_root / SUBSET_JSON[subset]
        if not json_path.exists():
            raise FileNotFoundError(f"Missing JSON for {subset}: {json_path}")
        out_path = out_root / subset / "manifest.jsonl"

        entries = list(_iter_entries(subset, json_path))
        if args.dry_run:
            print(f"[dry_run] {subset}: {len(entries):,} entries -> {out_path}")
            continue

        _write_manifest(out_path, entries, overwrite=args.overwrite)
        print(f"[wrote] {subset}: {len(entries):,} entries -> {out_path}")


if __name__ == "__main__":
    main()
