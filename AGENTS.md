# AGENTS.md
This file guides agentic coding in this repository.

## Repository overview
- LatentLM is a PyTorch Lightning research codebase for modeling VAE/DACVAE latents as sequences.
- Top-level scripts (`train_vision.py`, `train_audio.py`, `sample_vision.py`, `sample_audio.py`) are the main entry points.
- Core code lives in `src/` (models, Lightning module, data utilities, metrics).
- Data/cache outputs are stored in `logs_*`, `img_samples`, `audio_samples`, `pretrained_models`, `silence_samples`; avoid editing or committing large binaries.
- Preprocessing utilities live in `src/data_utils/preprocess` and are used by `preprocess_audio.py` and `preprocess_captions.py`.
- Slurm/HPC helpers live in `bash_scripts/` and `train_audio.sh`; they assume a cluster environment.

## Repo layout (selected)
- `train_vision.py`, `train_audio.py`: Lightning training entry points.
- `sample_vision.py`, `sample_audio.py`: Sampling/decoding entry points.
- `preprocess_vision.py`, `preprocess_audio.py`, `preprocess_captions.py`: Caching pipelines.
- `evaluate_vision.py`: FID/IS/torch_fidelity evaluation.
- `src/lightning.py`: `LitModule` and optimizer/scheduler setup.
- `src/models/`: Transformer/DiT/AR_DiT/MaskedAR implementations.
- `src/data_utils/`: datamodules, preprocessing helpers, audio/vision utilities.
- `src/metrics/`: Inception/FID/IS utilities.

## Build / lint / test
### Build/run (no build system)
- There is no build step (no `pyproject.toml`, `setup.py`, or `Makefile`); run scripts from repo root.
- Vision caching:
  `python preprocess_vision.py --vae pretrained_models/kl16.ckpt --data_dir /path/to/imagenet/train --batch_size 256 --num_workers 6`
- Vision training:
  `python train_vision.py --data-path /path/to/imagenet/train_cached --results-dir logs/... --model Transformer-Medium --input-size 16 --latent-size 16 ...`
- Vision sampling:
  `python sample_vision.py --checkpoint logs/.../last.ckpt --vae pretrained_models/kl16.ckpt --image_size 256 --output_dir visuals`
- Vision eval:
  `python evaluate_vision.py --images_path visuals --ref_stat_path /path/to/imagenet_256_val.npz --fid --is`
- Audio caching (new pipeline):
  `python preprocess_audio.py --source files --data_dir /path/to/audio --device cpu --processes 8`
  `python preprocess_audio.py --source hf --hf_dataset OpenSound/AudioCaps --hf_split train --device cuda --processes 4`
- Caption embeddings:
  `python preprocess_captions.py --metadata_path /path/to/manifest.jsonl --output_dir /path/to/text_embeddings --device cpu --processes 28`
- Audio training:
  `python train_audio.py --data-path /path/to/audioset_cached --results-dir logs/... --model Transformer-Medium --seq-len 251 --latent-size 128 --conditioning-type continuous --conditioning-dim 512 ...`
- Audio sampling:
  `python sample_audio.py --checkpoint logs/.../last.ckpt --output-dir audio_samples --text "Prompt"`

### Lint/format
- No lint/format config found (no `ruff`, `black`, `flake8`, `isort`, or `pre-commit` config).
- Keep formatting consistent with nearby files; do not introduce an auto-formatter unless the project adds one.

### Tests
- No test suite or `tests/` directory found.
- Single-test command: N/A. If you add pytest tests later, use:
  `python -m pytest tests/test_file.py -k test_name`

## Dependencies and environment
- Core runtime: `torch`, `lightning`, `torchvision`, `torchaudio`, `numpy`, `tqdm`.
- Optional/external: `diffusers`, `flash_attn`, `dacvae`, `audiotools`, `datasets`, `transformers`, `torch_fidelity`, `PIL`, `requests`.
- GPU is the default for training/sampling; CPU paths exist for preprocessing.
- Mixed precision flags are `bf16-mixed`, `16-mixed`, or `32` in training and sampling scripts.

## CLI conventions
- Scripts use `argparse` with kebab-case flags (e.g., `--data-path`, `--num-workers`).
- Vision training derives `seq_len` as `input_size * input_size` in `train_vision.py`.
- Audio training expects `--seq-len 251` for DACVAE latents and `--conditioning-type continuous` for CLAP.
- Sampling scripts accept `--cfg-scale` and `--num-inference-steps`; keep these consistent with training defaults.

## Checkpoints and logging
- Training checkpoints are written under `--results-dir` in a `checkpoints/` subdir.
- `ModelCheckpoint` uses `filename="{step:07d}"` and `save_last=True` (see `train_vision.py`, `train_audio.py`).
- `--log-every` controls TQDM/Lightning log frequency; `--ckpt-every` controls step-based checkpointing.
- Sampling outputs go to `--output_dir`; avoid mixing outputs with source datasets.

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
- Keep argument names consistent with CLI flags (e.g., `seq_len`, `latent_size`, `conditioning_type`).

### Error handling
- Raise `ValueError` or `RuntimeError` with clear messages on invalid inputs.
- Avoid silent failures; preprocessing utilities log skips via `SkipLogger` and continue.
- When catching broad exceptions for batch processing, log the error and keep the loop alive.

### Torch/Lightning conventions
- Use `@torch.no_grad()` for sampling/inference utilities.
- Respect autocast handling (`torch.autocast`, `torch.is_autocast_enabled`).
- Lightning training uses `LitModule`, `self.log`, and `self.save_hyperparameters()`; keep these patterns.
- `train_vision.py` and `train_audio.py` enable TF32; keep performance flags consistent.
- When loading cached `.pt` data, prefer `torch.load(..., weights_only=True)` as used in datasets.

## Data conventions
- Vision caches: `.npz` with `moments` and `moments_flip`; latents are sequence-first `(T, C)` with `T = H * W`.
- Audio caches: `.pt` with `posterior_params` (shape `[2C, T]`) and `text_embedding` (CLAP vector); datasets transpose to `(T, 2C)`.
- Prompting: class conditioning uses label indices; continuous conditioning uses CLAP embeddings.
- Vision normalization is fixed in `train_vision.py` (`data_scale` and `data_bias`); audio defaults to identity.
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
- New datamodules should follow existing `Cached*DataModule` patterns and return `(moments, prompts)`.
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
- `sample_vision.py` requires `--vae` unless `--speed-only` is set.
- `preprocess_audio.py` requires `--source` and a matching set of source-specific flags.
- `preprocess_captions.py` expects a JSONL manifest with `key` and `caption` fields.
- `evaluate_vision.py --fidelity` requires `torch_fidelity` and a valid `--train_data_dir`.

## Repo hygiene
- Do not modify or commit large binary artifacts (checkpoints, samples, logs, pretrained weights).
- Keep outputs in existing artifact directories (`logs_*`, `audio_samples`, `img_samples`, `pretrained_models`).
- Run scripts from repo root so `src` imports resolve correctly.

## Editor/agent rules
- No Cursor rules found (`.cursor/rules/` or `.cursorrules`).
- No Copilot rules found (`.github/copilot-instructions.md`).
- When ambiguity could change the outcome, ask immediately using the question tool and proceed after resolving it.
- Avoid excessively defensive code, if we are dealing with errors potentially external to the codebase (as in dataset handling), then its fine. If not, avoid try and except and always diagnose the root cause to make sure the whole flow of the code works as expected. 
- The README.md is a bit outdated, we will fix it later. 
