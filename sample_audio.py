# sample_audio.py
import argparse
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from src.lightning import LitModule
from src.data_utils.audio_utils import decode_audio_latents


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
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-inference-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text", type=str, default=None,
                        help="Text prompt for audio generation. If not provided, uses defaults.")
    parser.add_argument("--text-file", type=str, default=None,
                        help="File with text prompts (one per line)")
    parser.add_argument("--embedding", type=str, default=None,
                        help="Path to pre-computed prompt embeddings (.pt dict with clap/t5/t5_mask)")
    parser.add_argument("--output-dir", type=str, default="audio_samples")
    parser.add_argument("--sample-rate", type=int, default=48000,
                        help="Output sample rate (DACVAE default: 48000)")
    return parser.parse_args()


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


def get_text_embeddings(
    clap_model,
    clap_processor,
    t5_model,
    t5_tokenizer,
    text: str,
    device: torch.device,
    max_t5_tokens: int,
    clap_dim: int,
    t5_dim: int,
):
    clap_inputs = clap_processor(
        text=[text],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        clap_emb = clap_model.get_text_features(**clap_inputs)

    if clap_emb.shape[-1] != clap_dim:
        raise ValueError(f"Unexpected CLAP dim: {clap_emb.shape[-1]} (expected {clap_dim})")

    t5_inputs = t5_tokenizer(
        [text],
        padding=False,
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
    t5_hidden = t5_output.last_hidden_state[0]
    if t5_hidden.shape[-1] != t5_dim:
        raise ValueError(f"Unexpected T5 dim: {t5_hidden.shape[-1]} (expected {t5_dim})")

    actual_len = t5_hidden.shape[0]
    if actual_len > max_t5_tokens:
        t5_hidden = t5_hidden[:max_t5_tokens]
        actual_len = max_t5_tokens

    if actual_len < max_t5_tokens:
        pad_size = max_t5_tokens - actual_len
        padding = torch.zeros(pad_size, t5_hidden.shape[1], device=device, dtype=t5_hidden.dtype)
        t5_hidden = torch.cat([t5_hidden, padding], dim=0)

    t5_mask = torch.zeros(max_t5_tokens, device=device, dtype=torch.bool)
    t5_mask[:actual_len] = True

    return {
        "clap": clap_emb,
        "t5": t5_hidden.unsqueeze(0),
        "t5_mask": t5_mask.unsqueeze(0),
    }


def load_dacvae(weights_path: str, device: torch.device):
    import dacvae
    if weights_path is None:
        weights_path = "facebook/dacvae-watermarked"
    model = dacvae.DACVAE.load(weights_path).eval().to(device)
    return model


@torch.no_grad()
def main():
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        prompts = [f"embedding_{i}" for i in range(clap.shape[0])]
        all_prompt_data = [
            {
                "clap": clap[i : i + 1],
                "t5": t5[i : i + 1],
                "t5_mask": t5_mask[i : i + 1],
            }
            for i in range(clap.shape[0])
        ]
    else:
        # Get text prompts
        if args.text is not None:
            prompts = [args.text]
        elif args.text_file is not None:
            with open(args.text_file, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            prompts = DEFAULT_PROMPTS

        # Load CLAP model
        print(f"Loading CLAP model: {args.clap_model}")
        clap_model, clap_processor = load_clap_model(args.clap_model, device)

        # Load T5 model
        print(f"Loading T5 model: {args.t5_model}")
        t5_model, t5_tokenizer = load_t5_model(args.t5_model, device)

        # Generate embeddings for all prompts
        print("Generating prompt embeddings...")
        all_prompt_data = []
        for prompt in prompts:
            prompt_data = get_text_embeddings(
                clap_model,
                clap_processor,
                t5_model,
                t5_tokenizer,
                prompt,
                device,
                max_t5_tokens=max_t5_tokens,
                clap_dim=clap_dim,
                t5_dim=t5_dim,
            )
            all_prompt_data.append(prompt_data)

        # Free CLAP/T5 model memory
        del clap_model, clap_processor, t5_model, t5_tokenizer
        torch.cuda.empty_cache()

    # Load DACVAE for decoding
    print("Loading DACVAE...")
    dacvae_model = load_dacvae(args.dacvae_weights, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate samples
    print(f"Generating {len(prompts)} audio samples...")
    for i, (prompt, prompt_data) in enumerate(tqdm(zip(prompts, all_prompt_data), total=len(prompts))):
        with autocast:
            latents = lit.sample_latents(
                prompt_data,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
            )

        # latents shape: [1, T, D] -> need [1, D, T] for DACVAE
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
        if audio.dim() == 3:
            audio = audio[0]  # Remove batch dim

        # Create safe filename from prompt
        if isinstance(prompt, str) and not prompt.startswith("embedding_"):
            safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)[:50]
        else:
            safe_name = prompt
        filename = f"{i:04d}_{safe_name}.mp3"
        filepath = os.path.join(args.output_dir, filename)

        torchaudio.save(filepath, audio, args.sample_rate, format="mp3")
        print(f"Saved: {filepath}")

    print(f"\nGenerated {len(prompts)} audio samples in {args.output_dir}")


if __name__ == "__main__":
    main()
