# Preprocess Unification Plan (Audio + Captions + Manifests)

## Goal
Create a single `preprocess.py` entry point that:
- Builds WavCaps/AudioCaps per-subset/split manifests (today in `dataset_manifests.py`).
- Encodes **combined** audio + caption latents into one `.pt` file per item (so `src/data_utils/datamodule.py` remains unchanged).
- Builds merged training manifests (today in `scripts/build_manifest.py`).
- Writes latents into hashed directories (`latents/<md5[:2]>/<key>.pt`).

## Non-Goals
- No changes to `src/data_utils/datamodule.py` or model code.
- No changes to training/sampling logic.
- No format change to the dataloader payload keys it expects.

## Current State (for reference)
- `preprocess_audio.py` encodes DACVAE latents only.
- `preprocess_captions.py` encodes CLAP/T5 only.
- `dataset_manifests.py` writes per-subset/split `manifest.jsonl` with `key`, `caption`, `duration`.
- `scripts/build_manifest.py` merges those manifests into a training JSONL with `path` (and optional `caption`/`duration`).
- `src/data_utils/datamodule.py` expects **single** `.pt` containing:
  - `posterior_params`, `latent_length`, `clap_embedding`, `t5_last_hidden`, `t5_len`.

## Proposed Output Layout (unchanged for dataloader)
```
<data_root>/WavCaps/<subset>/
  manifest.jsonl
  latents/<md5[:2]>/<key>.pt

<data_root>/AudioCaps/<split>/
  manifest.jsonl
  latents/<md5[:2]>/<key>.pt
```

## Payload Schema (per latent file)
Only the fields required by `src/data_utils/datamodule.py` are stored in the `.pt` file.
```
{
  "posterior_params": Tensor[2C, T],
  "latent_length": int,
  "clap_embedding": Tensor[512],
  "t5_last_hidden": Tensor[L, 1024],
  "t5_len": int,
}
```
Any optional metadata (key/source/caption/duration) stays in the manifests only.
CLAP last hidden states (`clap_last_hidden`, `clap_len`) are not stored.

## New CLI (single entry point)
`preprocess.py` (root) with a default full run that processes both datasets and writes the merged manifest.
If you pass dataset/subset filters, it only encodes that subset/dataset and does not merge.

### Required dataset options
- `--dataset` (`wavcaps`, `audiocaps`, or `all`; default `all`)
- `--data-root` (base directory for outputs)

### WavCaps inputs
- `--wavcaps-json-root` (same as `dataset_manifests.py --json_root`)
- `--wavcaps-audio-root` (root of audio files)
- `--wavcaps-subsets` (comma-separated, default like today; if omitted, process all)

### AudioCaps inputs (HF)
- `--audiocaps-hf-dataset` (default `OpenSound/AudioCaps`)
- `--audiocaps-splits` (comma-separated; default `train` only)
- `--audiocaps-hf-data-dir`, `--audiocaps-hf-cache-dir`
- `--audiocaps-key-column`, `--audiocaps-caption-column`, `--audiocaps-audio-column`, `--audiocaps-audio-length-column`
- `--audiocaps-dedupe` (`error` or `first`)

### Encoding options
- `--dacvae-weights`
- `--clap-model`, `--t5-model`
- `--chunk-size-latents`, `--overlap-latents`
- `--min-duration-seconds`, `--max-duration-seconds`
- `--hash-prefix-len` (default `2`)

### Execution controls
- `--device` (`cpu` or `cuda`)
- `--processes`, `--threads-per-process`, `--mp-start-method`
- `--node-rank`, `--node-world-size`
- `--deterministic/--no-deterministic`
- `--output` (optional: path for merged training manifest; default `<data-root>/audio_manifest_train.jsonl`)
- `--overwrite-manifests` (optional: rebuild per-split/subset manifests)
- `--merge-manifests` (merge-only; skip encoding and write the merged manifest from existing per-split/subset manifests)
Note: merge-only is intended for when datasets were encoded separately; it reads per-split/subset manifests under `data-root`.

## Stages and Exact Code Changes

### Stage 1: Add unified `preprocess.py`
**Files**: add `preprocess.py`

**Key modules reused** (no changes to these):
- `src/data_utils/preprocess/common.py`
- `src/data_utils/preprocess/items.py`
- `src/data_utils/preprocess/text_encoder.py`
- `src/data_utils/preprocess/runners.py`
- `src/data_utils/utils.py` (for `encode_audio_latents`)

**Core logic to implement** (reuse current scripts; avoid new abstractions):
1) **Manifest builders**
   - Port `_iter_entries` and `_iter_audiocaps_entries` from `dataset_manifests.py`.
   - Keep key normalization for `AudioSet_SL` (`.wav` stripped).
   - Write per-subset/split `manifest.jsonl` with `key`, `caption`, `duration`.

2) **Audio path resolution (WavCaps)**
   - Given a `key` and subset, locate audio file under `--wavcaps-audio-root/<subset>/`.
   - If key has a suffix, use it as-is; otherwise, search for `key + ext` where ext in `AUDIO_EXTS`.
   - Consistency: audio lookup is driven strictly by the manifest key; no directory scans for discovery.
   - If missing or ambiguous (multiple matches), log via `SkipLogger` and continue (mirrors `preprocess_audio.py`).

3) **AudioCaps audio loading (HF)**
   - Use `datasets.load_dataset` per split and `load_audio_from_hf_example`.
   - Use `resolve_hf_uid`-style key mapping to keep compatibility with existing keys.

4) **Combined encoding**
   - Lift worker init and per-item processing directly from `preprocess_audio.py` and `preprocess_captions.py`.
   - Encode DACVAE latents first, then text embeddings back-to-back in the same worker.
   - Keep only `clap_embedding`, `t5_last_hidden`, `t5_len` from `TextEncoder` output (drop CLAP last hidden/len).
   - Merge into single payload and `atomic_write_pt` to hashed output path.

5) **Hashed output paths**
   - Compute `bucket = md5(key)[:hash_prefix_len]`.
   - Write to `latents/<bucket>/<key>.pt` (ensure `.pt` suffix).

6) **Resume + sharding**
   - Use `scan_cached_outputs(latents_root)` to skip existing files.
   - Use `belongs_to_node(relative_out_path)` for node sharding.

7) **Merged training manifest**
   - Reuse logic from `scripts/build_manifest.py` to build a global JSONL with:
     - `path` (relative to `--data-root`, pointing to `latents/...`), `source`, `key`, plus `caption`/`duration`.
   - Honor `--hash-prefix-len` for path generation.

**Files touched**: `preprocess.py` (new only).

### Stage 2: Remove legacy manifest scripts
**Files**: delete
- `dataset_manifests.py`
- `scripts/build_manifest.py`

**Change**:
- Remove these files entirely after `preprocess.py` is in place.
- Update documentation to reference only `preprocess.py`.

### Stage 3: Update docs and examples
**Files**: modify
- `README.md`
- `bash_scripts/preencode_wavcaps.sh`
- `bash_scripts/preencode_caps.sh`

**Change**:
- Replace old multi-script instructions with `preprocess.py` examples.
- Ensure examples show hashed latents and merged manifest build.

## Data Flow Summary
Default run (`--dataset all`, no subset filters):
1) Build per-split/subset `manifest.jsonl` if missing (or when `--overwrite-manifests`).
   - WavCaps: iterate all selected subsets from JSON metadata under `--wavcaps-json-root`.
   - AudioCaps: train split only (no val/test).
2) Encode audio + text and write combined latents to hashed `latents/` (skips cached files).
3) Write the merged training manifest at `<data-root>/audio_manifest_train.jsonl` (or `--output`).

Filtered runs (`--dataset wavcaps|audiocaps` or `--wavcaps-subsets`):
- Same as above, but skip the merged manifest step.
 - AudioCaps still means train split only (no val/test).

Merge-only (`--merge-manifests`):
- Skip encoding and only write the merged manifest from existing per-split/subset manifests.

Consistency: encoding is manifest-driven (keys/captions/durations from the manifests are the source of truth); missing/invalid audio is logged and skipped, not silently substituted.

## Validation Steps (run on HPC)
1) **Full run (both datasets + merged manifest)**
```
python preprocess.py --data-root /path/to/datasets \
  --wavcaps-json-root /path/to/jsons --wavcaps-audio-root /path/to/audio \
  --device cpu --processes 8 --threads-per-process 4
```

2) **WavCaps only (CPU)**
```
python preprocess.py --dataset wavcaps --data-root /path/to/datasets \
  --wavcaps-json-root /path/to/jsons --wavcaps-audio-root /path/to/audio \
  --device cpu --processes 8 --threads-per-process 4
```

3) **AudioCaps train (GPU)**
```
python preprocess.py --dataset audiocaps --data-root /path/to/datasets \
  --device cuda --processes 4
```

4) **Merged manifest only**
```
python preprocess.py --merge-manifests --data-root /path/to/datasets \
  --output /path/to/datasets/audio_manifest_train.jsonl
```

5) **Dataloader smoke check**
```
python -m src.data_utils.datamodule \
  --manifest-paths /path/to/datasets/audio_manifest_train.jsonl \
  --data-root /path/to/datasets \
  --silence-latent-path /path/to/silence_10s_dacvae.pt
```

## Risk Notes / Edge Cases
- **Audio path resolution**: if keys do not map 1:1 to filenames, add a one-time index build of `<subset>/**/*.ext` to map `stem -> path` (fail on duplicates).
- **HF AudioCaps durations**: keep `duration` only if `audio_length` + `sampling_rate` are available (as in current script).
- **Hash collisions**: MD5 prefix bucket only partitions, full filename is `key.pt`, so collisions are acceptable.

## Files Touched (Summary)
- **Add**: `preprocess.py`
- **Delete**: `dataset_manifests.py`
- **Delete**: `scripts/build_manifest.py`
- **Modify**: `README.md`
- **Modify**: `bash_scripts/preencode_wavcaps.sh`, `bash_scripts/preencode_caps.sh`
