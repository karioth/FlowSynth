# PyTorch implementation of LatentLM

Unofficial research codebase. No pretrained models are provided.

## Setup & Usage

This repo uses cached latents for training. The typical workflow is:

### 1) Cache latents
```bash
python -m src.data_utils.cache_image_latents \
  --vae pretrained_models/kl16.ckpt \
  --data_dir /path/to/imagenet/train \
  --batch_size 256 \
  --num_workers 6
```
By default this writes to `<data_dir>_cached`. Use `--cached_path` to override.

### 2) Train (Lightning, cached latents only)
```bash
python train.py \
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

### 3) Sample
```bash
python sample.py \
  --checkpoint logs/transformer_medium_40e/checkpoints/last.ckpt \
  --vae pretrained_models/kl16.ckpt \
  --image_size 256 \
  --cfg-scale 3.0 \
  --num_inference_steps 20 \
  --output_dir visuals/visuals_transformer120k
```
Defaults to a fixed class list; use `--num_images` for random classes or `--class_labels 281,282,...` to specify classes.

### 4) Evaluate
FID/IS from generated images:
```bash
python evaluate.py \
  --images_path visuals \
  --ref_stat_path /path/to/imagenet_256_val.npz \
  --batch_size 64 \
  --fid \
  --is
```
Optional torch_fidelity metrics against a reference dataset:
```bash
python evaluate.py \
  --images_path visuals \
  --train_data_dir /path/to/imagenet/train \
  --image_size 256 \
  --batch_size 64 \
  --fidelity
```
