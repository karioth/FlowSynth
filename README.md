# LatentLM (PyTorch)

Audio-only research codebase for modeling DACVAE latents as sequences (B, T, C).

## Audio (DACVAE + CLAP/T5)

Training expects JSONL manifests whose entries point to `.pt` files containing:
- `posterior_params`: [2C, T] mean+logvar from DACVAE
- `clap_embedding`: [D] CLAP pooled text embedding
- `t5_last_hidden`: [L, D]
- `t5_len`: int
- `latent_length`: int (optional, kept for metadata)

### 1) Cache audio latents
```bash
python preprocess_audio.py --source files --data_dir /path/to/audio --device cpu --processes 8
python preprocess_audio.py --source hf --hf_dataset OpenSound/AudioCaps --hf_split train --device cuda --processes 4
```
Outputs `<data_dir>_cached` (or `--cached_path`) with `.pt` latents.

### 2) Cache caption embeddings
```bash
python preprocess_captions.py --metadata_path /path/to/manifest.jsonl --output_dir /path/to/text_embeddings --device cpu --processes 28
```

### 3) Build a training manifest
If you are using the AudioCaps/WavCaps layout, build a merged manifest:
```bash
python scripts/build_manifest.py --data-root /path/to/datasets --output /path/to/manifest.jsonl
```
Otherwise, provide your own JSONL manifest with `path` fields pointing at cached `.pt` files.

### 4) Train
```bash
python train.py \
  --manifest-paths /path/to/manifest.jsonl \
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

### 5) Sample
```bash
python sample.py \
  --checkpoint logs/run_01/checkpoints/last.ckpt \
  --dacvae-weights facebook/dacvae-watermarked \
  --clap-model laion/larger_clap_music \
  --cfg-scale 4.0 \
  --num-inference-steps 250 \
  --output-dir audio_samples
```
You can also pass `--text`, `--text-file`, or `--embedding` to control prompts.

## Notes
- The Lightning module is sequence-first; any reshaping happens in `src/data_utils`.
- Output latents are `(B, T, C)`; DACVAE decoding expects `(B, C, T)`.
