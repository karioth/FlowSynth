# sample.py
import argparse
import csv
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from src.lightning import LitModule
from src.data_utils.utils import decode_audio_latents


DEFAULT_PROMPTS = [
    "Multiple rapid bursts of sound, which sound like a series of gunshots.",
    "Birds singing in clear and loud chirps to each other",
    "Loud rock music from concert",
    "Metal bar clanking with sharp loud noises",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sample audio from a Lightning checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt.")
    parser.add_argument("--dacvae-weights", type=str, default=None,
                        help="Path to DACVAE weights (default: facebook/dacvae-watermarked)")
    parser.add_argument("--clap-model", type=str, default="laion/larger_clap_music",
                        help="HuggingFace CLAP model ID")
    parser.add_argument("--t5-model", type=str, default="google/flan-t5-large",
                        help="HuggingFace T5 model ID")
    parser.add_argument("--max-t5-tokens", type=int, default=None,
                        help="Maximum T5 tokens (defaults to prompt_seq_len - 1)")
    parser.add_argument("--clap-dim", type=int, default=512,
                        help="CLAP pooled embedding dimension")
    parser.add_argument("--t5-dim", type=int, default=1024,
                        help="T5 hidden state dimension")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["bf16-mixed", "16-mixed", "32"],
        help="Autocast precision.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="Initial paper CFG omega_0 (not a constant standard CFG scale).",
    )
    parser.add_argument(
        "--cfg-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear_decay"],
        help="CFG schedule: 'constant' (default) or 'linear_decay' (ramps from cfg_scale to 1.0).",
    )
    parser.add_argument(
        "--cfg-mask-prob",
        type=float,
        default=0.0,
        help="Probability of masking unconditional branch tokens (autoguidance). 0.0=off, 1.0=always mask.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument(
        "--ardiff-step",
        type=int,
        default=None,
        help=(
            "Override AR-Diff sampling lag for AR_DiT checkpoints. "
            "When omitted, uses the scheduler default."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text prompt for audio generation. If not provided, uses defaults.",
    )
    prompt_source.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="File with text prompts (one per line)",
    )
    prompt_source.add_argument(
        "--embedding",
        type=str,
        default=None,
        help="Path to pre-computed prompt embeddings (.pt dict with clap/t5/t5_mask)",
    )
    prompt_source.add_argument(
        "--prompt-csv",
        type=str,
        default=None,
        help=(
            "Path to AudioCaps CSV with columns audiocap_id,youtube_id,caption. "
            "Outputs are saved as {audiocap_id}.wav."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="audio_samples")
    parser.add_argument("--sample-rate", type=int, default=48000,
                        help="Output sample rate (DACVAE default: 48000)")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _parse_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def get_dist_context(local_rank_arg: int | None) -> tuple[int, int, int]:
    rank = _parse_env_int("RANK", 0)
    world_size = _parse_env_int("WORLD_SIZE", 1)
    local_rank = local_rank_arg if local_rank_arg is not None else _parse_env_int("LOCAL_RANK", 0)

    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, {world_size - 1}], got {rank}")
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {local_rank}")

    return rank, world_size, local_rank


def shard_indices(total: int, rank: int, world_size: int) -> list[int]:
    return list(range(rank, total, world_size))


def iter_chunks(items: list[int], chunk_size: int):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def get_autocast_dtype(precision: str, device: torch.device):
    if device.type != "cuda":
        return None
    if precision == "bf16-mixed":
        return torch.bfloat16
    if precision == "16-mixed":
        return torch.float16
    return None


def load_clap_model(model_id: str, device: torch.device):
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained(model_id, use_safetensors=True).eval().to(device)
    processor = ClapProcessor.from_pretrained(model_id)
    return model, processor


def load_t5_model(model_id: str, device: torch.device):
    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = T5EncoderModel.from_pretrained(model_id).eval().to(device)
    return model, tokenizer


def get_text_embeddings_batch(
    clap_model,
    clap_processor,
    t5_model,
    t5_tokenizer,
    texts: list[str],
    device: torch.device,
    max_t5_tokens: int,
    clap_dim: int,
    t5_dim: int,
):
    if not texts:
        raise ValueError("texts must be non-empty for batched embedding")

    clap_inputs = clap_processor(
        text=texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        clap_emb = clap_model.get_text_features(**clap_inputs)

    if clap_emb.dim() != 2:
        raise ValueError(f"Unexpected CLAP rank: {clap_emb.dim()} (expected 2)")
    if clap_emb.shape[0] != len(texts):
        raise ValueError(f"Unexpected CLAP batch: {clap_emb.shape[0]} (expected {len(texts)})")
    if clap_emb.shape[1] != clap_dim:
        raise ValueError(f"Unexpected CLAP dim: {clap_emb.shape[-1]} (expected {clap_dim})")

    t5_inputs = t5_tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_t5_tokens,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        t5_output = t5_model(
            input_ids=t5_inputs["input_ids"],
            attention_mask=t5_inputs.get("attention_mask"),
            return_dict=True,
        )
    t5_hidden = t5_output.last_hidden_state
    if t5_hidden.dim() != 3:
        raise ValueError(f"Unexpected T5 rank: {t5_hidden.dim()} (expected 3)")
    if t5_hidden.shape[0] != len(texts):
        raise ValueError(f"Unexpected T5 batch: {t5_hidden.shape[0]} (expected {len(texts)})")
    if t5_hidden.shape[1] != max_t5_tokens:
        raise ValueError(
            f"Unexpected T5 sequence length: {t5_hidden.shape[1]} (expected {max_t5_tokens})"
        )
    if t5_hidden.shape[2] != t5_dim:
        raise ValueError(f"Unexpected T5 dim: {t5_hidden.shape[2]} (expected {t5_dim})")

    attention_mask = t5_inputs.get("attention_mask")
    if attention_mask is None:
        raise ValueError("T5 tokenizer did not return attention_mask")
    if attention_mask.shape != (len(texts), max_t5_tokens):
        raise ValueError(
            f"Unexpected attention_mask shape: {tuple(attention_mask.shape)} "
            f"(expected {(len(texts), max_t5_tokens)})"
        )
    t5_mask = attention_mask.to(device=device, dtype=torch.bool)

    return {
        "clap": clap_emb,
        "t5": t5_hidden,
        "t5_mask": t5_mask,
    }


def load_dacvae(weights_path: str, device: torch.device):
    import dacvae
    if weights_path is None:
        weights_path = "facebook/dacvae-watermarked"
    model = dacvae.DACVAE.load(weights_path).eval().to(device)
    return model


def load_audiocaps_prompt_csv(path: str) -> tuple[list[str], list[str]]:
    required_columns = {"audiocap_id", "youtube_id", "caption"}
    output_filenames: list[str] = []
    prompts: list[str] = []
    seen_ids: set[int] = set()

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Prompt CSV has no header row: {path}")
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Prompt CSV missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            raw_id = (row.get("audiocap_id") or "").strip()
            caption = (row.get("caption") or "").strip()
            if not raw_id:
                raise ValueError(f"Row {row_number}: audiocap_id is empty.")
            if not caption:
                raise ValueError(f"Row {row_number}: caption is empty.")

            try:
                audiocap_id = int(raw_id)
            except ValueError as exc:
                raise ValueError(f"Row {row_number}: invalid audiocap_id '{raw_id}'.") from exc
            if audiocap_id in seen_ids:
                raise ValueError(f"Row {row_number}: duplicate audiocap_id {audiocap_id}.")

            seen_ids.add(audiocap_id)
            output_filenames.append(f"{audiocap_id}.wav")
            prompts.append(caption)

    if not prompts:
        raise ValueError("Prompt CSV is empty.")

    return output_filenames, prompts


@torch.no_grad()
def main():
    args = parse_args()
    rank, world_size, local_rank = get_dist_context(args.local_rank)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        if local_rank >= cuda_count:
            raise ValueError(
                f"LOCAL_RANK {local_rank} is out of range for {cuda_count} visible CUDA device(s)."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    amp_dtype = get_autocast_dtype(args.precision, device)
    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )

    # Load model
    print(f"Loading model from {args.checkpoint}")
    lit = LitModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    lit.to(device=device)
    lit.eval()

    prompt_seq_len = getattr(lit.hparams, "prompt_seq_len", None)
    clap_dim = getattr(lit.hparams, "clap_dim", args.clap_dim)
    t5_dim = getattr(lit.hparams, "t5_dim", args.t5_dim)
    if args.max_t5_tokens is not None:
        max_t5_tokens = args.max_t5_tokens
    elif prompt_seq_len is not None:
        max_t5_tokens = int(prompt_seq_len) - 1
    else:
        max_t5_tokens = 68

    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")

    global_total = 0
    global_indices: list[int] = []
    prompts: list[str] = []
    local_output_filenames: list[str | None] = []

    # Get prompts
    if args.embedding is not None:
        # Use pre-computed embeddings
        print(f"Loading embedding from {args.embedding}")
        embeddings = torch.load(args.embedding, map_location=device)
        if not isinstance(embeddings, dict):
            raise ValueError("Expected embedding file to be a dict with keys: clap, t5, t5_mask")
        if not {"clap", "t5", "t5_mask"}.issubset(embeddings):
            raise ValueError("Embedding dict missing required keys: clap, t5, t5_mask")

        clap = embeddings["clap"].to(device)
        t5 = embeddings["t5"].to(device)
        t5_mask = embeddings["t5_mask"].to(device)

        if clap.dim() == 1:
            clap = clap.unsqueeze(0)
        if t5.dim() == 2:
            t5 = t5.unsqueeze(0)
        if t5_mask.dim() == 1:
            t5_mask = t5_mask.unsqueeze(0)

        if clap.shape[-1] != clap_dim:
            raise ValueError(f"Unexpected CLAP dim: {clap.shape[-1]} (expected {clap_dim})")
        if t5.shape[-1] != t5_dim:
            raise ValueError(f"Unexpected T5 dim: {t5.shape[-1]} (expected {t5_dim})")
        if t5.shape[1] != max_t5_tokens or t5_mask.shape[1] != max_t5_tokens:
            raise ValueError("T5 sequence length does not match max_t5_tokens")

        global_total = clap.shape[0]
        global_indices = shard_indices(global_total, rank, world_size)
        prompts = [f"embedding_{global_idx}" for global_idx in global_indices]
        local_output_filenames = [None] * len(global_indices)

        if global_indices:
            gather_idx = torch.tensor(global_indices, device=clap.device, dtype=torch.long)
            local_clap = clap.index_select(0, gather_idx)
            local_t5 = t5.index_select(0, gather_idx)
            local_t5_mask = t5_mask.index_select(0, gather_idx)
        else:
            local_clap = clap[:0]
            local_t5 = t5[:0]
            local_t5_mask = t5_mask[:0]
    else:
        # Get text prompts
        if args.prompt_csv is not None:
            print(f"Loading AudioCaps prompts from {args.prompt_csv}")
            global_output_filenames, global_prompts = load_audiocaps_prompt_csv(args.prompt_csv)
            print(f"Loaded {len(global_prompts)} AudioCaps row(s)")
        elif args.text is not None:
            global_prompts = [args.text]
            global_output_filenames = None
        elif args.text_file is not None:
            with open(args.text_file, "r") as f:
                global_prompts = [line.strip() for line in f if line.strip()]
            global_output_filenames = None
        else:
            global_prompts = DEFAULT_PROMPTS
            global_output_filenames = None

        global_total = len(global_prompts)
        global_indices = shard_indices(global_total, rank, world_size)
        prompts = [global_prompts[global_idx] for global_idx in global_indices]
        if global_output_filenames is not None:
            local_output_filenames = [global_output_filenames[global_idx] for global_idx in global_indices]
        else:
            local_output_filenames = [None] * len(global_indices)

        local_count = len(global_indices)
        local_clap = torch.empty((local_count, clap_dim), dtype=torch.float32, device="cpu")
        local_t5 = torch.empty((local_count, max_t5_tokens, t5_dim), dtype=torch.float32, device="cpu")
        local_t5_mask = torch.empty((local_count, max_t5_tokens), dtype=torch.bool, device="cpu")

        if local_count:
            # Load CLAP model
            print(f"Loading CLAP model: {args.clap_model}")
            clap_model, clap_processor = load_clap_model(args.clap_model, device)

            # Load T5 model
            print(f"Loading T5 model: {args.t5_model}")
            t5_model, t5_tokenizer = load_t5_model(args.t5_model, device)

            # Generate embeddings for local shard prompts in batches
            print("Generating prompt embeddings in batches...")
            local_positions = list(range(local_count))
            embedding_batches = iter_chunks(local_positions, args.batch_size)
            num_embedding_batches = (local_count + args.batch_size - 1) // args.batch_size
            for batch_positions in tqdm(
                embedding_batches,
                total=num_embedding_batches,
                disable=rank != 0,
            ):
                batch_texts = [prompts[pos] for pos in batch_positions]
                batch_prompt_data = get_text_embeddings_batch(
                    clap_model,
                    clap_processor,
                    t5_model,
                    t5_tokenizer,
                    batch_texts,
                    device,
                    max_t5_tokens=max_t5_tokens,
                    clap_dim=clap_dim,
                    t5_dim=t5_dim,
                )
                local_clap[batch_positions] = batch_prompt_data["clap"].detach().to(dtype=torch.float32).cpu()
                local_t5[batch_positions] = batch_prompt_data["t5"].detach().to(dtype=torch.float32).cpu()
                local_t5_mask[batch_positions] = batch_prompt_data["t5_mask"].detach().cpu()

            # Free CLAP/T5 model memory
            del clap_model, clap_processor, t5_model, t5_tokenizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"[rank {rank}/{world_size}] handling {len(global_indices)} of {global_total} prompts")
    if not global_indices:
        print(f"[rank {rank}/{world_size}] No prompts assigned; exiting.")
        return

    # Load DACVAE for decoding
    print("Loading DACVAE...")
    dacvae_model = load_dacvae(args.dacvae_weights, device)
    os.makedirs(args.output_dir, exist_ok=True)
    local_positions = list(range(len(global_indices)))
    num_batches = (len(local_positions) + args.batch_size - 1) // args.batch_size

    # Generate samples
    print(
        f"[rank {rank}/{world_size}] Generating {len(global_indices)} audio samples "
        f"in {num_batches} batches..."
    )
    generation_batches = iter_chunks(local_positions, args.batch_size)
    for batch_positions in tqdm(
        generation_batches,
        total=num_batches,
        disable=rank != 0,
    ):
        batch_global_indices = [global_indices[pos] for pos in batch_positions]
        batch_prompts = [prompts[pos] for pos in batch_positions]
        batch_output_filenames = [local_output_filenames[pos] for pos in batch_positions]
        batch_prompt_data = {
            "clap": local_clap[batch_positions].to(device=device),
            "t5": local_t5[batch_positions].to(device=device),
            "t5_mask": local_t5_mask[batch_positions].to(device=device, dtype=torch.bool),
        }

        with autocast:
            latents = lit.sample_latents(
                batch_prompt_data,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                ardiff_step=args.ardiff_step,
                cfg_schedule=args.cfg_schedule,
                cfg_mask_prob=args.cfg_mask_prob,
            )

        # latents shape: [B, T, D] -> need [B, D, T] for DACVAE
        latents = latents.transpose(1, 2)

        # Decode with DACVAE
        metadata = {
            "sample_rate": args.sample_rate,
            "latent_length": latents.shape[-1],
        }
        with torch.autocast(device_type=device.type, enabled=False):
            audio = decode_audio_latents(dacvae_model, latents.float(), metadata)

        # Save audio
        audio = audio.detach().cpu()
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if audio.dim() != 3:
            raise ValueError(f"Expected decoded audio rank 3 [B, C, T], got shape {tuple(audio.shape)}")
        if audio.shape[0] != len(batch_positions):
            raise ValueError(
                f"Decoded batch size {audio.shape[0]} does not match prompt batch size {len(batch_positions)}"
            )

        for batch_idx, (global_idx, prompt, output_filename) in enumerate(
            zip(batch_global_indices, batch_prompts, batch_output_filenames)
        ):
            sample_audio = audio[batch_idx]
            if output_filename is not None:
                filename = output_filename
                filepath = os.path.join(args.output_dir, filename)
                torchaudio.save(filepath, sample_audio, args.sample_rate, format="wav")
            else:
                # Create safe filename from prompt
                if isinstance(prompt, str) and not prompt.startswith("embedding_"):
                    safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)[:50]
                else:
                    safe_name = prompt
                filename = f"{global_idx:04d}_{safe_name}.mp3"
                filepath = os.path.join(args.output_dir, filename)
                torchaudio.save(filepath, sample_audio, args.sample_rate, format="mp3")
    print(f"\n[rank {rank}/{world_size}] Generated {len(prompts)} audio samples in {args.output_dir}")


if __name__ == "__main__":
    main()
