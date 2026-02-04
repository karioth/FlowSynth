This file guides agentic coding in this repository.

## Repository overview
- LatentLM is a PyTorch Lightning research codebase for modeling DACVAE latents as sequences.
- Top-level scripts (`train.py`, `sample.py`, `preprocess_audio.py`, `preprocess_captions.py`) are the main entry points.
- Core code lives in `src/` (models, Lightning module, data utilities).
- Data/cache outputs are stored in `logs_*`, `audio_samples`, `pretrained_models`, `silence_samples`; avoid editing or committing large binaries.
- Preprocessing utilities live in `src/data_utils/preprocess` and are used by `preprocess_audio.py` and `preprocess_captions.py`.
- Slurm/HPC helpers live in `bash_scripts/` and `train.sh`; they assume a cluster environment.

## Repo layout (selected)
- `train.py`: Lightning training entry point.
- `sample.py`: Sampling/decoding entry point.
- `preprocess_audio.py`, `preprocess_captions.py`: Caching pipelines.
- `scripts/build_manifest.py`: Manifest builder for AudioCaps/WavCaps layouts.
- `src/lightning.py`: `LitModule` and optimizer/scheduler setup.
- `src/utils.py`: Posterior sampling helper.
- `src/models/`: Transformer/DiT/AR_DiT/MaskedAR implementations.
- `src/data_utils/`: datamodule (`datamodule.py`), helpers (`utils.py`), preprocessing helpers.

## Build / lint / test
### Build/run (no build system)
- There is no build step (no `pyproject.toml`, `setup.py`, or `Makefile`); run scripts from repo root.
- Audio caching:
  `python preprocess_audio.py --source files --data_dir /path/to/audio --device cpu --processes 8`
  `python preprocess_audio.py --source hf --hf_dataset OpenSound/AudioCaps --hf_split train --device cuda --processes 4`
- Caption embeddings:
  `python preprocess_captions.py --metadata_path /path/to/manifest.jsonl --output_dir /path/to/text_embeddings --device cpu --processes 28`
- Build a training manifest (AudioCaps/WavCaps layout):
  `python scripts/build_manifest.py --data-root /path/to/datasets --output /path/to/manifest.jsonl`
- Audio training:
  `python train.py --manifest-paths /path/to/manifest.jsonl --data-root /path/to/datasets --results-dir logs/... --model MaskedAR-L --seq-len 251 --latent-size 128 ...`
- Audio sampling:
  `python sample.py --checkpoint logs/.../last.ckpt --output-dir audio_samples --text "Prompt"`

### Lint/format
- No lint/format config found (no `ruff`, `black`, `flake8`, `isort`, or `pre-commit` config).
- Keep formatting consistent with nearby files; do not introduce an auto-formatter unless the project adds one.

### Tests
- No test suite or `tests/` directory found.
- Single-test command: N/A. If you add pytest tests later, use:
  `python -m pytest tests/test_file.py -k test_name`

## Dependencies and environment
- Core runtime: `torch`, `lightning`, `torchaudio`, `numpy`, `tqdm`.
- Optional/external: `diffusers`, `flash_attn`, `dacvae`, `audiotools`, `datasets`, `transformers`.
- GPU is the default for training/sampling; CPU paths exist for preprocessing.
- Mixed precision flags are `bf16-mixed`, `16-mixed`, or `32` in training and sampling scripts.

## CLI conventions
- Scripts use `argparse` with kebab-case flags (e.g., `--manifest-paths`, `--num-workers`).
- Audio training expects `--seq-len 251` for DACVAE latents.
- Sampling scripts accept `--cfg-scale` and `--num-inference-steps`; keep these consistent with training defaults.

## Checkpoints and logging
- Training checkpoints are written under `--results-dir` in a `checkpoints/` subdir.
- `ModelCheckpoint` uses `filename="{step:07d}"` and `save_last=True` (see `train.py`).
- `--log-every` controls TQDM/Lightning log frequency; `--ckpt-every` controls step-based checkpointing.
- Sampling outputs go to `--output-dir`; avoid mixing outputs with source datasets.

## Code style guidelines
### Imports
- Group imports: standard library, third-party, then local `src` imports.
- Keep one blank line between import groups.
- Prefer explicit imports (avoid `import *`).

### Formatting
- 4-space indentation; no tabs.
- Keep lines readable; wrap long argument lists like existing code.
- Use blank lines to separate logical blocks; avoid excessive vertical spacing.
- Comments are used sparingly; add only when a block is non-obvious.

### Types and annotations
- Type hints are used in many modules (especially in `src/data_utils/preprocess` and `src/models`).
- Prefer `| None` for optionals (Python 3.10+) as seen in core modules.
- Some files include `from __future__ import annotations`; keep it at top if present.
- Annotate public functions/classes and non-trivial helpers; avoid over-annotating small local variables.

### Naming
- Classes: `PascalCase` (e.g., `MaskedARTransformer`).
- Functions/variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CLASS_LABELS`).
- Keep argument names consistent with CLI flags (e.g., `seq_len`, `latent_size`).

### Error handling
- Raise `ValueError` or `RuntimeError` with clear messages on invalid inputs.
- Avoid silent failures; preprocessing utilities log skips via `SkipLogger` and continue.
- When catching broad exceptions for batch processing, log the error and keep the loop alive.

### Torch/Lightning conventions
- Use `@torch.no_grad()` for sampling/inference utilities.
- Respect autocast handling (`torch.autocast`, `torch.is_autocast_enabled`).
- Lightning training uses `LitModule`, `self.log`, and `self.save_hyperparameters()`; keep these patterns.
- `train.py` enables TF32; keep performance flags consistent.
- When loading cached `.pt` data, prefer `torch.load(..., weights_only=True)` as used in datasets.

## Data conventions
- Audio caches: `.pt` with `posterior_params` (shape `[2C, T]`), `clap_embedding`, `t5_last_hidden`, `t5_len`, and optional `latent_length`; datasets transpose to `(T, 2C)`.
- Prompting uses a dict with `clap`, `t5`, and `t5_mask`.
- Output latents are `(B, T, C)`; decoding expects `(B, C, T)` for DACVAE.

## Pathing and imports
- Top-level scripts assume repo root is on `sys.path`; run them from repo root.
- Avoid relative imports inside `src`; use absolute `from src...` like existing code.

## Serialization and safety
- Cache files are written via `atomic_write_pt` to avoid partial writes.
- Keep payloads simple dicts of tensors/ints/strings (no custom classes).
- Prefer `weights_only=True` for `torch.load` when reading cached artifacts.

## Extending the codebase
- New models must be registered in `src/models/__init__.py` via `All_models`.
- New datamodules should follow existing `Cached*DataModule` patterns and return `(posterior_params, prompts)`.
- For new preprocess steps, reuse helpers in `src/data_utils/preprocess` (e.g., `atomic_write_pt`, `run_pool`, `run_gpu_pool`).
- Keep sample and evaluation scripts in the repo root for consistency with current entry points.

## Determinism and seeding
- Training scripts call `seed_everything` and set TF32 flags; keep this behavior unless you have a reason to change it.
- Sampling scripts seed `random`, `numpy`, and `torch`; preserve per-rank offsets for distributed runs.
- GPU preprocessing can enforce determinism via `--deterministic`; be aware of the speed tradeoff.

## Distributed and multiprocessing notes
- Distributed sampling/training relies on `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`; initialize with `torch.distributed.init_process_group`.
- Preprocessing supports node sharding via `node_rank` and `node_world_size`; see `preprocess_audio.py` and `preprocess_captions.py`.
- GPU preprocessing uses spawn and a shared queue; avoid `fork` with CUDA.

## Common pitfalls
- `preprocess_audio.py` requires `--source` and a matching set of source-specific flags.
- `preprocess_captions.py` expects a JSONL manifest with `key` and `caption` fields.

## Repo hygiene
- Do not modify or commit large binary artifacts (checkpoints, samples, logs, pretrained weights).
- Keep outputs in existing artifact directories (`logs_*`, `audio_samples`, `pretrained_models`, `silence_samples`).
- Run scripts from repo root so `src` imports resolve correctly.

## Editor/agent rules
- No Cursor rules found (`.cursor/rules/` or `.cursorrules`).
- No Copilot rules found (`.github/copilot-instructions.md`).
- When ambiguity could change the outcome, ask immediately using the question tool and proceed after resolving it.
- Avoid excessively defensive code, if we are dealing with errors potentially external to the codebase (as in dataset handling), then its fine. If not, avoid try and except and always diagnose the root cause to make sure the whole flow of the code works as expected.
