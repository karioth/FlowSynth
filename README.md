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
