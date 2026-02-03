#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_DATA_ROOT = "/share/users/student/f/friverossego/datasets"
DEFAULT_WAVCAPS_SUBSETS = "AudioSet_SL,BBC_Sound_Effects,FreeSound,SoundBible"
DEFAULT_AUDIOCAPS_SPLITS = "train"


def _parse_csv(value: str) -> list[str]:
    if value is None:
        return []
    cleaned = value.strip()
    if cleaned == "" or cleaned.lower() in {"none", "null"}:
        return []
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def _ensure_pt(name: str) -> str:
    if name.endswith(".pt"):
        return name
    return f"{name}.pt"


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


def _write_source(
    *,
    out_file,
    data_root: Path,
    rel_root: Path,
    manifest_path: Path,
    latents_dir: str,
    verify_exists: bool,
    skip_missing: bool,
    source_label: str,
):
    total = 0
    missing = 0

    for key, entry in _iter_manifest_entries(manifest_path):
        filename = _ensure_pt(key)
        latent_rel = rel_root / latents_dir / filename

        if verify_exists:
            latent_abs = data_root / latent_rel
            if not latent_abs.exists():
                missing += 1
                if skip_missing:
                    continue
                raise FileNotFoundError(
                    f"Missing file for key '{key}' in {source_label}: {latent_abs}"
                )

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

    return total, missing


def main() -> None:
    parser = argparse.ArgumentParser("Build a global audio manifest from per-split manifests")
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Base dataset directory (contains AudioCaps/ and WavCaps/)",
    )
    parser.add_argument(
        "--wavcaps-subsets",
        type=str,
        default=DEFAULT_WAVCAPS_SUBSETS,
        help="Comma-separated WavCaps subsets to include",
    )
    parser.add_argument(
        "--audiocaps-splits",
        type=str,
        default=DEFAULT_AUDIOCAPS_SPLITS,
        help="Comma-separated AudioCaps splits to include",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL manifest path",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.jsonl",
        help="Manifest filename inside each subset/split directory",
    )
    parser.add_argument(
        "--latents-dir",
        type=str,
        default="latents",
        help="Merged latents directory name",
    )
    parser.add_argument(
        "--verify-exists",
        action="store_true",
        help="Check that each audio/text file exists",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip entries with missing files (requires --verify-exists)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")

    wavcaps_subsets = _parse_csv(args.wavcaps_subsets)
    audiocaps_splits = _parse_csv(args.audiocaps_splits)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_entries = 0
    total_missing = 0

    with output_path.open("w", encoding="utf-8") as out_file:
        for subset in wavcaps_subsets:
            rel_root = Path("WavCaps") / subset
            manifest_path = data_root / rel_root / args.manifest_name
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing manifest: {manifest_path}")
            source_label = f"WavCaps/{subset}"
            count, missing = _write_source(
                out_file=out_file,
                data_root=data_root,
                rel_root=rel_root,
                manifest_path=manifest_path,
                latents_dir=args.latents_dir,
                verify_exists=args.verify_exists,
                skip_missing=args.skip_missing,
                source_label=source_label,
            )
            print(f"[wrote] {source_label}: {count:,} entries")
            total_entries += count
            total_missing += missing

        for split in audiocaps_splits:
            rel_root = Path("AudioCaps") / split
            manifest_path = data_root / rel_root / args.manifest_name
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing manifest: {manifest_path}")
            source_label = f"AudioCaps/{split}"
            count, missing = _write_source(
                out_file=out_file,
                data_root=data_root,
                rel_root=rel_root,
                manifest_path=manifest_path,
                latents_dir=args.latents_dir,
                verify_exists=args.verify_exists,
                skip_missing=args.skip_missing,
                source_label=source_label,
            )
            print(f"[wrote] {source_label}: {count:,} entries")
            total_entries += count
            total_missing += missing

    if args.verify_exists:
        if args.skip_missing:
            print(f"[done] {total_entries:,} entries written ({total_missing:,} skipped)")
        else:
            print(f"[done] {total_entries:,} entries written")
    else:
        print(f"[done] {total_entries:,} entries written")


if __name__ == "__main__":
    main()
