# LatentLM (PyTorch)

Unofficial research codebase. Models operate on cached VAE/DACVAE latents as
sequences (B, T, C) rather than pixels or waveforms.

## Vision (ImageNet-style latents)

### 1) Cache image latents
```bash
python preprocess_vision.py \
  --vae pretrained_models/kl16.ckpt \
  --data_dir /path/to/imagenet/train \
  --batch_size 256 \
  --num_workers 6
```
Outputs `<data_dir>_cached` with `.npz` files containing `moments` and
`moments_flip` (posterior params).

### 2) Train (vision)
```bash
python train_vision.py \
  --data-path /path/to/imagenet/train_cached \
  --results-dir logs/lightning_transformer_medium_40e \
  --model Transformer-Medium \
  --input-size 16 \
  --latent-size 16 \
  --prediction-type flow \
  --batch-size 128 \
  --epochs 40 \
  --lr 1e-4 \
  --weight-decay 0.1 \
  --lr-scheduler cosine \
  --lr-warmup-steps 100 \
  --batch-mul 2 \
  --precision bf16-mixed
```
For KL16 + 256px images, use `--input-size 16 --latent-size 16`.

### 3) Sample (vision)
```bash
python sample_vision.py \
  --checkpoint logs/transformer_medium_40e/checkpoints/last.ckpt \
  --vae pretrained_models/kl16.ckpt \
  --image_size 256 \
  --cfg-scale 3.0 \
  --num_inference_steps 20 \
  --output_dir visuals/visuals_transformer120k
```
Defaults to a fixed class list; use `--num_images` for random classes or
`--class_labels 281,282,...` to specify classes.

### 4) Evaluate (vision)
FID/IS from generated images:
```bash
python evaluate_vision.py \
  --images_path visuals \
  --ref_stat_path /path/to/imagenet_256_val.npz \
  --batch_size 64 \
  --fid \
  --is
```
Optional torch_fidelity metrics against a reference dataset:
```bash
python evaluate_vision.py \
  --images_path visuals \
  --train_data_dir /path/to/imagenet/train \
  --image_size 256 \
  --batch_size 64 \
  --fidelity
```

## Audio (DACVAE + CLAP)

The audio dataloader expects `.pt` files with:
- `posterior_params`: [2C, T] mean+logvar from DACVAE
- `text_embedding`: [D] CLAP text embedding
- `latent_length`: int (optional, kept for metadata)

### 1) Cache audio latents + CLAP text embeddings (AudioSet)
```bash
python -m src.data_utils.cache_audioset \
  --output_dir /path/to/audioset_cached \
  --hf_dataset laion/audioset-with-captions \
  --hf_split train \
  --batch_size 1 \
  --num_workers 6
```

### 2) Train (audio)
```bash
python train_audio.py \
  --data-path /path/to/audioset_cached \
  --results-dir logs/audio_transformer_medium \
  --model Transformer-Medium \
  --seq-len 251 \
  --latent-size 128 \
  --conditioning-type continuous \
  --conditioning-dim 512 \
  --batch-size 16 \
  --epochs 1 \
  --lr 1e-4 \
  --precision bf16-mixed
```

### 3) Sample (audio)
```bash
python sample_audio.py \
  --checkpoint logs/audio_transformer_medium/checkpoints/last.ckpt \
  --dacvae-weights facebook/dacvae-watermarked \
  --clap-model laion/larger_clap_music \
  --cfg-scale 4.0 \
  --num-inference-steps 250 \
  --output-dir audio_samples
```
You can also pass `--text`, `--text-file`, or `--embedding` to control prompts.

## Notes
- The Lightning module is sequence-first; any image/audio reshaping happens in
  `src/data_utils`.
- Normalization is fixed: vision uses `data_scale=0.2331244945526123` and
  `data_bias=-0.07858214527368546` in `train_vision.py`. Audio defaults to
  identity (scale=1, bias=0).
