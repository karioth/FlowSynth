# FlowSynth (PyTorch)

Audio-only research codebase for modeling DACVAE latents as sequences (B, T, C).

## Data Layout

Per-item preprocessing writes `.pt` payloads containing:
- `posterior_params`: `[2C, T]` DACVAE posterior mean/logvar
- `latent_length`: valid latent length before pad/crop
- `clap_embedding`: `[512]` pooled CLAP text embedding
- `t5_last_hidden`: `[L, 1024]` T5 hidden states
- `t5_len`: valid T5 token count

Manifest mode uses JSONL entries pointing at those per-item files. The default
training path is now consolidated mode, where each subset/split has a
`consolidated_latents_bf16.pt` cache built from its `manifest.jsonl` plus
`latents/**/*.pt`.

## Default Workflow

### 1) Preprocess audio + captions

Default preprocessing is `--dataset all`, which means:
- WavCaps subsets: `AudioSet_SL,BBC_Sound_Effects,FreeSound,SoundBible`
- AudioCaps splits: `train`
- DACVAE weights: `facebook/dacvae-watermarked`
- CLAP model: `laion/larger_clap_music`
- T5 model: `google/flan-t5-large`

Typical full run:
```bash
python preprocess.py \
  --data-root /path/to/datasets \
  --wavcaps-json-root /path/to/wavcaps/jsons \
  --wavcaps-audio-root /path/to/wavcaps/audio \
  --device cpu \
  --processes 8 \
  --threads-per-process 4
```

This writes:
- `WavCaps/<subset>/manifest.jsonl`
- `WavCaps/<subset>/latents/<md5[:2]>/<key>.pt`
- `AudioCaps/<split>/manifest.jsonl`
- `AudioCaps/<split>/latents/<md5[:2]>/<key>.pt`

When you run the default `--dataset all` with the default subsets/splits,
`preprocess.py` also writes `audio_manifest_train.jsonl` under `--data-root`.
That merged manifest is mainly for manifest-mode training; the default training
path below uses consolidated caches instead.

Useful variants:

WavCaps only:
```bash
python preprocess.py \
  --dataset wavcaps \
  --data-root /path/to/datasets \
  --wavcaps-json-root /path/to/wavcaps/jsons \
  --wavcaps-audio-root /path/to/wavcaps/audio \
  --device cpu \
  --processes 8
```

AudioCaps only:
```bash
python preprocess.py \
  --dataset audiocaps \
  --data-root /path/to/datasets \
  --device cuda \
  --processes 4
```

Merge manifests only:
```bash
python preprocess.py \
  --merge-manifests \
  --data-root /path/to/datasets \
  --output /path/to/datasets/audio_manifest_train.jsonl
```

Notes:
- `--processes` defaults to all CPUs on `--device cpu`, or all visible GPUs on `--device cuda`.
- Duration filtering defaults to `--min-duration-seconds 0.05 --max-duration-seconds 600.0`.
- Chunked DACVAE encoding defaults to `--chunk-size-latents 1024 --overlap-latents 12`.

### 2) Consolidate caches

`train.py` defaults to `--data-mode consolidated`, so after preprocessing you
normally build one consolidated cache per subset/split:

```bash
python consolidate_cache.py \
  --subset-path /path/to/datasets/WavCaps/AudioSet_SL
```

That reads:
- `/path/to/datasets/WavCaps/AudioSet_SL/manifest.jsonl`
- `/path/to/datasets/WavCaps/AudioSet_SL/latents/**/*.pt`

and writes:
- `/path/to/datasets/WavCaps/AudioSet_SL/consolidated_latents_bf16.pt`

Repeat for each WavCaps subset and each AudioCaps split you want to train on.
Use `--confirm` if you want a slower integrity check after writing the cache.

### 3) Train

Default training path:
```bash
python train.py \
  --data-root /path/to/datasets \
  --results-dir logs/run_01
```

Current CLI defaults:
- `--data-mode consolidated`
- `--model MaskSynth-L`
- `--seq-len 251`
- `--latent-size 128`
- `--prompt-seq-len 69`
- `--batch-size 128`
- `--epochs 80`
- `--lr 1e-4`
- `--lr-scheduler constant_with_warmup`
- `--precision bf16-mixed`
- `--strategy ddp`

In consolidated mode, `--data-root` is searched recursively for
`consolidated_latents_bf16.pt`. The silence latent defaults to
`silence_samples/silence_10s_dacvae.pt`.

Manifest mode is still supported if you want to train directly from merged
JSONL manifests instead of consolidated caches:
```bash
python train.py \
  --data-mode manifest \
  --manifest-paths /path/to/datasets/audio_manifest_train.jsonl \
  --data-root /path/to/datasets \
  --results-dir logs/run_01
```

### 4) Sample

Run sampling from the repo root (`FlowSynth/`).

Single prompt:
```bash
python sample.py \
  --checkpoint logs/run_01/checkpoints/last.ckpt \
  --text "Birds singing in clear and loud chirps to each other" \
  --output-dir audio_samples
```

Default sampling values from the CLI:
- `--cfg-scale 3.0`
- `--num-inference-steps 100`
- `--batch-size 1`
- `--sample-rate 48000`
- `--precision bf16-mixed`
- CLAP model `laion/larger_clap_music`
- T5 model `google/flan-t5-large`

Prompt sources are mutually exclusive:
- `--text`
- `--text-file`
- `--embedding`
- `--prompt-csv`

If none are provided, `sample.py` falls back to the built-in `DEFAULT_PROMPTS`
list in the script.

Output naming:
- `--prompt-csv` writes `.wav` files named `{audiocap_id}.wav`
- all other prompt modes write `.mp3` files named `{global_idx}_{prompt}.mp3`

AudioCaps/eval-compatible sampling:
```bash
python sample.py \
  --checkpoint logs/run_01/checkpoints/last.ckpt \
  --prompt-csv audiocaps-test.csv \
  --batch-size 8 \
  --output-dir samples_audio/run_name
```

Multi-GPU with `torchrun` shards prompts by rank so work is not duplicated:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 sample.py \
  --checkpoint logs/run_01/checkpoints/last.ckpt \
  --prompt-csv audiocaps-test.csv \
  --batch-size 8 \
  --output-dir samples_audio/run_name
```

Notes:
- `--max-t5-tokens` defaults to `prompt_seq_len - 1` from the checkpoint, or `68` if unavailable.
- `--ardiff-step` is only relevant for DriftSynth checkpoints.
- Older `--prompt-json` docs are stale; the current AudioCaps/eval path is `--prompt-csv`.

### 5) Evaluate

`evaluate.py` is single-process and single-device. It computes:
- FAD
- KAD
- CLAP score
- PaSST KLD

Typical AudioCaps evaluation run:
```bash
python evaluate.py \
  --gen samples_audio/run_name \
  --device cuda
```

Current defaults:
- `--model panns-wavegram-logmel` for FAD/KAD embeddings
- `--prompts-csv audiocaps-test.csv`
- `--gt samples_audiocaps_test`
- `--clap-model clap-2023`
- `--kld-model passt-base-10s`
- `--workers 4`

Important behavior:
- FAD/KAD are computed folder-wise over `*.wav` files in `--gt` and `--gen`.
- CLAP score and PaSST KLD use AudioCaps CSV pairing, so generated files must be named `{audiocap_id}.wav`.
- Ground-truth files are matched as `{youtube_id}.wav` or `Y{youtube_id}.wav`.
- The script writes metrics JSON to `<parent of --gen>/<gen_dir_name>.json` unless `--output-json` is set.
- The script always computes all four metrics. CLAP score currently requires CUDA, so in practice evaluation should be run on a GPU device.
- Non-CSV sampling modes save `.mp3`, so they are not directly consumable by `evaluate.py` without conversion to `.wav`.

Short smoke run:
```bash
python evaluate.py \
  --gen samples_audio/run_name \
  --device cuda \
  --limit 64
```

## Notes

- The Lightning module trains on sampled latents of shape `(B, T, C)`, even though cached `posterior_params` on disk are `[2C, T]`.
- Any pad/crop logic for audio and T5 happens in `src/data_utils`.
- DACVAE decoding expects latents shaped `(B, C, T)`, so sampling transposes model output before decode.
