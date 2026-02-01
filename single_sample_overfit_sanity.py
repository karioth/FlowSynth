import json
import os
import random
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torchaudio

from src.models import DiT, Transformer, AR_DiT, MaskedARTransformer
from src.flow_matching import (
    FlowMatchingSchedulerDiT,
    FlowMatchingSchedulerTransformer,
    FlowMatchingSchedulerARDiff,
    FlowMatchingSchedulerMaskedAR,
)
from src.models.modules.distributions import SequenceDiagonalGaussianDistribution
from src.data_utils.audio_utils import decode_audio_latents
from sample_audio import load_clap_model, load_t5_model, get_text_embeddings, load_dacvae


# Hardcode paths here
MANIFEST_PATHS = [
    "/share/users/student/f/friverossego/datasets/audio_manifest_train.jsonl",
]
DATA_ROOT = "/share/users/student/f/friverossego/datasets"
SILENCE_LATENT_PATH = "silence_samples/silence_10s_dacvae.pt"
CAPTION_MANIFEST_PATH = "/share/users/student/f/friverossego/datasets/WavCaps/AudioSet_SL/manifest.jsonl"

# Provide a caption directly or via file
CAPTION_TEXT = None
CAPTION_PATH = "/path/to/caption.txt"
USE_MANIFEST_CAPTION = True

# Optional filters
SOURCE_FILTER = "AudioSet_SL"
SELECT_KEY = None

# Models and dims
CLAP_MODEL_ID = "laion/larger_clap_music"
T5_MODEL_ID = "google/flan-t5-large"

SEQ_LEN = 251
LATENT_SIZE = 128
MAX_T5_TOKENS = 68
PROMPT_SEQ_LEN = MAX_T5_TOKENS + 1
CLAP_DIM = 512
T5_DIM = 1024

# DiT config
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 1024
DEPTH = 9
NUM_HEADS = 8
CLASS_DROPOUT_PROB = 0.1

# Model switch
MODEL_KIND = "MaskedAR"  # Options: DiT, Transformer, AR_DiT, MaskedAR
DIFFUSION_DEPTH = 2
BATCH_MUL = 1
MASK_PROB_MIN = 0.5
MASK_PROB_MAX = 0.5

# Training
LR = 1e-4
WEIGHT_DECAY = 0.0
MAX_STEPS = 2000
LOG_EVERY = 50
USE_POSTERIOR_MEAN = True

# Sampling
CFG_SCALES = [0.0, 4.0]
NUM_INFERENCE_STEPS = 250
SAMPLE_RATE = 48000
DACVAE_WEIGHTS = None
OUTPUT_DIR = "single_sample_overfit"
SMOKETEST_DIR = "smoketest"
SAMPLE_SEED = 123

# Conditioning comparison tolerances
CLAP_RTOL = 1e-3
CLAP_ATOL = 1e-3
T5_RTOL = 1e-3
T5_ATOL = 1e-3


def _resolve_path(path_str: str, data_root: str | None) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return path.as_posix()
    if data_root is None:
        raise ValueError(f"Relative path without data_root: {path_str}")
    return (Path(data_root) / path).as_posix()


def _iter_manifest_entries(manifest_paths: list[str]):
    for manifest_path in manifest_paths:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                yield json.loads(stripped)


def _select_sample():
    select_key_lower = SELECT_KEY.lower() if isinstance(SELECT_KEY, str) else None

    for entry in _iter_manifest_entries(MANIFEST_PATHS):
        entry_key = entry.get("key") or entry.get("id")
        entry_key_str = str(entry_key) if entry_key is not None else ""
        entry_key_base = os.path.splitext(os.path.basename(entry_key_str))[0]
        entry_key_lower = entry_key_str.lower()
        entry_base_lower = entry_key_base.lower()
        if select_key_lower is not None and select_key_lower not in {entry_key_lower, entry_base_lower}:
            continue
        if SOURCE_FILTER:
            source = entry.get("source")
            if source is not None and SOURCE_FILTER not in source:
                continue

        audio_path = _resolve_path(entry["audio_path"], DATA_ROOT)
        text_path = _resolve_path(entry["text_path"], DATA_ROOT)
        if not os.path.exists(audio_path):
            if select_key_lower is not None:
                raise FileNotFoundError(f"Missing audio for SELECT_KEY={SELECT_KEY}: {audio_path}")
            continue
        if not os.path.exists(text_path):
            if select_key_lower is not None:
                raise FileNotFoundError(f"Missing text for SELECT_KEY={SELECT_KEY}: {text_path}")
            continue

        audio_data = torch.load(audio_path, map_location="cpu", weights_only=True)
        posterior_params = audio_data["posterior_params"]
        moments = posterior_params.transpose(0, 1)
        latent_length = int(audio_data.get("latent_length", moments.shape[0]))
        if latent_length != SEQ_LEN:
            if select_key_lower is not None:
                raise RuntimeError(
                    f"SELECT_KEY={SELECT_KEY} has latent_length={latent_length}, expected {SEQ_LEN}"
                )
            continue

        sample_key = entry_key_base or os.path.splitext(os.path.basename(audio_path))[0]
        fallback_caption = entry.get("caption")
        return sample_key, audio_path, text_path, fallback_caption

    if select_key_lower is not None:
        raise RuntimeError(
            f"No matching sample found for SELECT_KEY={SELECT_KEY}. "
            "Check key, SOURCE_FILTER, and manifest path."
        )
    raise RuntimeError("No matching sample found. Adjust filters or manifest path.")


def _read_caption_from_manifest(manifest_path: str | None, target_key: str) -> tuple[str | None, str]:
    if not manifest_path or not os.path.exists(manifest_path):
        return None, "manifest_missing"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                key = entry.get("key") or entry.get("id")
                if key is None:
                    continue
                key_str = str(key)
                key_base = os.path.splitext(os.path.basename(key_str))[0]
                if key_str == target_key or key_base == target_key:
                    caption = entry.get("caption")
                    if isinstance(caption, str) and caption:
                        return caption, "found"
                    return None, "caption_missing"
    except OSError:
        return None, "manifest_unreadable"
    return None, "id_not_found"


def _load_caption(sample_key: str, fallback_caption: str | None) -> tuple[str, str]:
    if CAPTION_TEXT:
        return CAPTION_TEXT, "caption_text"
    if CAPTION_PATH and os.path.exists(CAPTION_PATH):
        with open(CAPTION_PATH, "r", encoding="utf-8") as f:
            return f.read().strip(), "caption_path"
    if USE_MANIFEST_CAPTION:
        caption, status = _read_caption_from_manifest(CAPTION_MANIFEST_PATH, sample_key)
        if caption:
            return caption, f"manifest:{status}"
        if fallback_caption:
            return fallback_caption, f"fallback:{status}"
        raise RuntimeError(f"Caption not found for key '{sample_key}': {status}")
    if fallback_caption:
        return fallback_caption, "fallback"
    raise RuntimeError("No caption available. Set CAPTION_TEXT or CAPTION_PATH.")


def _prepare_t5(t5_hidden: torch.Tensor, t5_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    if t5_len <= 0:
        t5_padded = t5_hidden.new_zeros(MAX_T5_TOKENS, t5_hidden.shape[1])
        t5_mask = torch.zeros(MAX_T5_TOKENS, dtype=torch.bool)
        return t5_padded, t5_mask
    if t5_len >= MAX_T5_TOKENS:
        t5_padded = t5_hidden[: MAX_T5_TOKENS].clone()
        t5_mask = torch.ones(MAX_T5_TOKENS, dtype=torch.bool)
        return t5_padded, t5_mask
    pad_size = MAX_T5_TOKENS - t5_len
    padding = t5_hidden.new_zeros(pad_size, t5_hidden.shape[1])
    t5_padded = torch.cat([t5_hidden[:t5_len], padding], dim=0)
    t5_mask = torch.zeros(MAX_T5_TOKENS, dtype=torch.bool)
    t5_mask[:t5_len] = True
    return t5_padded, t5_mask


def _compare_conditioning(train_prompt: dict, raw_prompt: dict) -> None:
    clap_train = train_prompt["clap"].float()
    clap_raw = raw_prompt["clap"].float()
    t5_train = train_prompt["t5"].float()
    t5_raw = raw_prompt["t5"].float()

    clap_close = torch.allclose(clap_train, clap_raw, rtol=CLAP_RTOL, atol=CLAP_ATOL)
    t5_close = torch.allclose(t5_train, t5_raw, rtol=T5_RTOL, atol=T5_ATOL)
    mask_equal = torch.equal(train_prompt["t5_mask"], raw_prompt["t5_mask"])

    clap_diff = (clap_train - clap_raw).abs()
    t5_diff = (t5_train - t5_raw).abs()

    print(
        f"CLAP diff max={clap_diff.max().item():.6f} mean={clap_diff.mean().item():.6f} close={clap_close}"
    )
    print(
        f"T5 diff max={t5_diff.max().item():.6f} mean={t5_diff.mean().item():.6f} close={t5_close}"
    )
    print(f"T5 mask match: {mask_equal}")

    if not clap_close or not t5_close or not mask_equal:
        raise RuntimeError("Prompt embeddings do not match raw caption within tolerance.")


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for flash-attn")

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    sample_key, audio_path, text_path, fallback_caption = _select_sample()
    caption, caption_source = _load_caption(sample_key, fallback_caption)
    print(f"Using audio: {audio_path}")
    print(f"Using text:  {text_path}")
    print(f"Caption:    {caption}")
    print(f"Caption src: {caption_source}")

    audio_data = torch.load(audio_path, map_location="cpu", weights_only=True)
    posterior_params = audio_data["posterior_params"].transpose(0, 1)
    latent_length = int(audio_data.get("latent_length", posterior_params.shape[0]))
    if latent_length != SEQ_LEN:
        raise RuntimeError(f"latent_length {latent_length} != SEQ_LEN {SEQ_LEN}")
    moments = posterior_params[:latent_length]

    text_data = torch.load(text_path, map_location="cpu", weights_only=True)
    clap_emb = text_data["clap_embedding"]
    t5_hidden = text_data["t5_last_hidden"]
    t5_len = int(text_data["t5_len"])
    if clap_emb.shape[-1] != CLAP_DIM:
        raise RuntimeError(f"CLAP dim mismatch: {clap_emb.shape[-1]} != {CLAP_DIM}")
    if t5_hidden.shape[-1] != T5_DIM:
        raise RuntimeError(f"T5 dim mismatch: {t5_hidden.shape[-1]} != {T5_DIM}")

    t5_padded, t5_mask = _prepare_t5(t5_hidden, t5_len)
    prompt_train = {
        "clap": clap_emb.unsqueeze(0),
        "t5": t5_padded.unsqueeze(0),
        "t5_mask": t5_mask.unsqueeze(0),
    }

    os.makedirs(SMOKETEST_DIR, exist_ok=True)
    dacvae_model = load_dacvae(DACVAE_WEIGHTS, device)
    sample_id = os.path.splitext(os.path.basename(audio_path))[0]
    x0_ref = moments.unsqueeze(0)
    posterior = SequenceDiagonalGaussianDistribution(x0_ref, deterministic=USE_POSTERIOR_MEAN)
    x0_latents = posterior.mode() if USE_POSTERIOR_MEAN else posterior.sample()
    latents = x0_latents.transpose(1, 2)
    metadata = {
        "sample_rate": int(getattr(dacvae_model, "sample_rate", SAMPLE_RATE)),
        "latent_length": latents.shape[-1],
    }
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        audio = decode_audio_latents(dacvae_model, latents.float().to(device), metadata)
    audio = audio.detach().cpu()
    if audio.dim() == 3:
        audio = audio[0]
    smoke_path = os.path.join(SMOKETEST_DIR, f"{sample_id}_input.mp3")
    torchaudio.save(smoke_path, audio, metadata["sample_rate"], format="mp3")
    print(f"Saved smoketest input: {smoke_path}")

    del dacvae_model
    torch.cuda.empty_cache()

    print("Loading CLAP/T5 models for caption embedding...")
    clap_model, clap_processor = load_clap_model(CLAP_MODEL_ID, device)
    t5_model, t5_tokenizer = load_t5_model(T5_MODEL_ID, device)

    raw_prompt = get_text_embeddings(
        clap_model,
        clap_processor,
        t5_model,
        t5_tokenizer,
        caption,
        device,
        max_t5_tokens=MAX_T5_TOKENS,
        clap_dim=CLAP_DIM,
        t5_dim=T5_DIM,
    )

    del clap_model, clap_processor, t5_model, t5_tokenizer
    torch.cuda.empty_cache()

    prompt_train_cpu = {
        "clap": prompt_train["clap"].detach().cpu(),
        "t5": prompt_train["t5"].detach().cpu(),
        "t5_mask": prompt_train["t5_mask"].detach().cpu(),
    }
    raw_prompt_cpu = {
        "clap": raw_prompt["clap"].detach().cpu(),
        "t5": raw_prompt["t5"].detach().cpu(),
        "t5_mask": raw_prompt["t5_mask"].detach().cpu(),
    }
    _compare_conditioning(prompt_train_cpu, raw_prompt_cpu)

    prompt_train = {
        "clap": prompt_train["clap"].to(device),
        "t5": prompt_train["t5"].to(device),
        "t5_mask": prompt_train["t5_mask"].to(device),
    }
    prompt_sample = {
        "clap": raw_prompt["clap"].to(device),
        "t5": raw_prompt["t5"].to(device),
        "t5_mask": raw_prompt["t5_mask"].to(device),
    }

    moments = moments.unsqueeze(0).to(device)
    posterior = SequenceDiagonalGaussianDistribution(moments, deterministic=USE_POSTERIOR_MEAN)

    if MODEL_KIND == "DiT":
        model = DiT(
            seq_len=SEQ_LEN,
            in_channels=LATENT_SIZE,
            hidden_size=HIDDEN_SIZE,
            depth=DEPTH,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE_SIZE,
            conditioning_type="continuous",
            clap_dim=CLAP_DIM,
            t5_dim=T5_DIM,
            prompt_seq_len=PROMPT_SEQ_LEN,
            class_dropout_prob=CLASS_DROPOUT_PROB,
        ).to(device)
        scheduler = FlowMatchingSchedulerDiT()
    elif MODEL_KIND == "Transformer":
        model = Transformer(
            seq_len=SEQ_LEN,
            in_channels=LATENT_SIZE,
            hidden_size=HIDDEN_SIZE,
            depth=DEPTH,
            diffusion_depth=DIFFUSION_DEPTH,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE_SIZE,
            diffusion_intermediate_size=INTERMEDIATE_SIZE,
            conditioning_type="continuous",
            clap_dim=CLAP_DIM,
            t5_dim=T5_DIM,
            prompt_seq_len=PROMPT_SEQ_LEN,
            class_dropout_prob=CLASS_DROPOUT_PROB,
        ).to(device)
        scheduler = FlowMatchingSchedulerTransformer(batch_mul=BATCH_MUL)
    elif MODEL_KIND == "AR_DiT":
        model = AR_DiT(
            seq_len=SEQ_LEN,
            in_channels=LATENT_SIZE,
            hidden_size=HIDDEN_SIZE,
            depth=DEPTH,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE_SIZE,
            conditioning_type="continuous",
            clap_dim=CLAP_DIM,
            t5_dim=T5_DIM,
            prompt_seq_len=PROMPT_SEQ_LEN,
            class_dropout_prob=CLASS_DROPOUT_PROB,
        ).to(device)
        scheduler = FlowMatchingSchedulerARDiff()
    elif MODEL_KIND == "MaskedAR":
        model = MaskedARTransformer(
            seq_len=SEQ_LEN,
            in_channels=LATENT_SIZE,
            hidden_size=HIDDEN_SIZE,
            depth=DEPTH,
            diffusion_depth=DIFFUSION_DEPTH,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_HEADS,
            intermediate_size=INTERMEDIATE_SIZE,
            diffusion_intermediate_size=INTERMEDIATE_SIZE,
            conditioning_type="continuous",
            clap_dim=CLAP_DIM,
            t5_dim=T5_DIM,
            prompt_seq_len=PROMPT_SEQ_LEN,
            class_dropout_prob=CLASS_DROPOUT_PROB,
        ).to(device)
        scheduler = FlowMatchingSchedulerMaskedAR(
            mask_prob_min=MASK_PROB_MIN,
            mask_prob_max=MASK_PROB_MAX,
            batch_mul=BATCH_MUL,
        )
    else:
        raise ValueError(f"Unknown MODEL_KIND: {MODEL_KIND}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        x0 = posterior.mode() if USE_POSTERIOR_MEAN else posterior.sample()
        with autocast:
            loss = scheduler.get_losses(model, x0, prompt_train)
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0 or step == 1 or step == MAX_STEPS:
            print(f"step {step:05d} | loss {loss.item():.6f}")

    model.eval()
    del optimizer
    torch.cuda.empty_cache()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dacvae_model = load_dacvae(DACVAE_WEIGHTS, device)

    for cfg_scale in CFG_SCALES:
        torch.manual_seed(SAMPLE_SEED)
        scheduler.configure_sampling()
        scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
        with torch.no_grad(), autocast:
            latents = model.sample_with_cfg(prompt_sample, cfg_scale=cfg_scale, sample_func=scheduler)

        latents = latents.transpose(1, 2)
        metadata = {
            "sample_rate": SAMPLE_RATE,
            "latent_length": latents.shape[-1],
        }
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            audio = decode_audio_latents(dacvae_model, latents.float(), metadata)

        audio = audio.detach().cpu()
        if audio.dim() == 3:
            audio = audio[0]

        filename = f"cfg_{cfg_scale:g}.mp3"
        filepath = os.path.join(OUTPUT_DIR, filename)
        torchaudio.save(filepath, audio, SAMPLE_RATE, format="mp3")
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    main()
