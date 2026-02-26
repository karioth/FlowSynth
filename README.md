# EqSynth (PyTorch)

Audio-only research codebase for modeling DACVAE latents as sequences (B, T, C).

## Audio (DACVAE + CLAP/T5)

Training expects JSONL manifests whose entries point to `.pt` files containing:
- `posterior_params`: [2C, T] mean+logvar from DACVAE
- `clap_embedding`: [D] CLAP pooled text embedding
- `t5_last_hidden`: [L, D]
- `t5_len`: int
- `latent_length`: int (optional, kept for metadata)

### 1) Preprocess audio + captions
Default run (WavCaps + AudioCaps, writes merged manifest):
```bash
python preprocess.py \
  --data-root /path/to/datasets \
  --wavcaps-json-root /path/to/jsons \
  --wavcaps-audio-root /path/to/audio \
  --device cpu --processes 8 --threads-per-process 4
```

WavCaps only:
```bash
python preprocess.py \
  --dataset wavcaps \
  --data-root /path/to/datasets \
  --wavcaps-json-root /path/to/jsons \
  --wavcaps-audio-root /path/to/audio \
  --device cpu --processes 8 --threads-per-process 4
```

AudioCaps only:
```bash
python preprocess.py \
  --dataset audiocaps \
  --data-root /path/to/datasets \
  --device cuda --processes 4
```

Merged manifest only:
```bash
python preprocess.py \
  --merge-manifests \
  --data-root /path/to/datasets \
  --output /path/to/datasets/audio_manifest_train.jsonl
```

Latents are written under `latents/<md5[:2]>/<key>.pt` for each subset/split.

### 2) Train
```bash
python train.py \
  --manifest-paths /path/to/datasets/audio_manifest_train.jsonl \
  --data-root /path/to/datasets \
  --results-dir logs/run_01 \
  --model MaskedAR-L \
  --seq-len 251 \
  --latent-size 128 \
  --batch-size 32 \
  --epochs 1 \
  --lr 1e-4 \
  --precision bf16-mixed
```

### 3) Sample
Run from the repo root (`EqSynth/`).

Single-process:
```bash
python sample.py \
  --checkpoint logs/run_01/checkpoints/last.ckpt \
  --dacvae-weights facebook/dacvae-watermarked \
  --clap-model laion/larger_clap_music \
  --cfg-scale 4.0 \
  --num-inference-steps 250 \
  --batch-size 8 \
  --output-dir audio_samples
```
Multi-GPU with torchrun (rank-sharded prompts, no duplicated work):
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 sample.py \
  --checkpoint logs/run_01/checkpoints/last.ckpt \
  --prompt-json /path/to/filename_to_caption_mapping.json \
  --cfg-scale 3.0 \
  --batch-size 8 \
  --output-dir samples_audio/run_name
```

Prompt sources are mutually exclusive (use exactly one):
- `--text`
- `--text-file`
- `--embedding`
- `--prompt-json`

Output format:
- `--prompt-json`: writes `.wav` files using JSON keys as filenames.
- other modes: writes `.mp3` files with global-index filename prefix.

Notes:
- `--output-dir` is created automatically if it does not exist.
- JSON mode expects a mapping: `output_filename.wav -> caption`.

### 4) Evaluate
`evaluate.py` is single-process/single-device only.

```bash
python evaluate.py \
  --gen /path/to/generated_wavs \
  --gt /path/to/ground_truth_wavs \
  --sr 16000 \
  --backbone cnn14
```

Optional smoke test:
```bash
python evaluate.py \
  --gt samples_audio/audiocaps_test_gt \
  --gen samples_audio/generated_wavs 
```

Important:
- `evaluate.py` compares by basename and warns when there is no overlap, but still runs in unpaired mode.
- For AudioCaps-style eval, use `--prompt-json` in sampling so generated names match GT names.

## Notes
- The Lightning module is sequence-first; any reshaping happens in `src/data_utils`.
- Output latents are `(B, T, C)`; DACVAE decoding expects `(B, C, T)`.
