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
    "Electronic music with heavy bass and synths",
    "Acoustic guitar playing a gentle melody",
    "Jazz piano with smooth chords",
    "Orchestral music with strings and brass",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sample audio from a Lightning checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt.")
    parser.add_argument("--dacvae-weights", type=str, default=None,
                        help="Path to DACVAE weights (default: facebook/dacvae-watermarked)")
    parser.add_argument("--clap-model", type=str, default="laion/larger_clap_music",
                        help="HuggingFace CLAP model ID")
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
                        help="Path to pre-computed CLAP embedding .pt file")
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


def get_text_embedding(clap_model, clap_processor, text: str, device: torch.device):
    inputs = clap_processor(
        text=[text],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        embedding = clap_model.get_text_features(**inputs)
    return embedding


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

    # Get prompts
    if args.embedding is not None:
        # Use pre-computed embedding
        print(f"Loading embedding from {args.embedding}")
        embeddings = torch.load(args.embedding, map_location=device)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        prompts = [f"embedding_{i}" for i in range(len(embeddings))]
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

        # Generate embeddings for all prompts
        print("Generating CLAP embeddings...")
        embeddings = []
        for prompt in prompts:
            emb = get_text_embedding(clap_model, clap_processor, prompt, device)
            embeddings.append(emb)
        embeddings = torch.cat(embeddings, dim=0)

        # Free CLAP model memory
        del clap_model, clap_processor
        torch.cuda.empty_cache()

    # Load DACVAE for decoding
    print("Loading DACVAE...")
    dacvae_model = load_dacvae(args.dacvae_weights, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate samples
    print(f"Generating {len(prompts)} audio samples...")
    for i, (prompt, embedding) in enumerate(tqdm(zip(prompts, embeddings), total=len(prompts))):
        with autocast:
            latents = lit.sample_latents(
                embedding.unsqueeze(0) if embedding.dim() == 1 else embedding,
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
