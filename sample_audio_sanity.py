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
from sample_audio import (
    DEFAULT_PROMPTS,
    get_autocast_dtype,
    load_clap_model,
    load_t5_model,
    get_text_embeddings,
    load_dacvae,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity sampling with multiple CFG scales.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt.")
    parser.add_argument("--dacvae-weights", type=str, default=None)
    parser.add_argument("--clap-model", type=str, default="laion/larger_clap_music")
    parser.add_argument("--t5-model", type=str, default="google/flan-t5-large")
    parser.add_argument("--max-t5-tokens", type=int, default=None)
    parser.add_argument("--clap-dim", type=int, default=512)
    parser.add_argument("--t5-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["bf16-mixed", "16-mixed", "32"],
    )
    parser.add_argument("--cfg-scales", type=str, default="0,4")
    parser.add_argument("--num-inference-steps", type=int, default=250)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="audio_samples_sanity")
    parser.add_argument("--sample-rate", type=int, default=48000)
    return parser.parse_args()


def _parse_cfg_scales(value: str) -> list[float]:
    if value is None:
        return []
    scales = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        scales.append(float(item))
    return scales


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

    if args.text is not None:
        prompts = [args.text]
    elif args.text_file is not None:
        with open(args.text_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    cfg_scales = _parse_cfg_scales(args.cfg_scales)
    if not cfg_scales:
        raise ValueError("No cfg scales provided")

    print(f"Loading CLAP model: {args.clap_model}")
    clap_model, clap_processor = load_clap_model(args.clap_model, device)
    print(f"Loading T5 model: {args.t5_model}")
    t5_model, t5_tokenizer = load_t5_model(args.t5_model, device)

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

    del clap_model, clap_processor, t5_model, t5_tokenizer
    torch.cuda.empty_cache()

    print("Loading DACVAE...")
    dacvae_model = load_dacvae(args.dacvae_weights, device)

    os.makedirs(args.output_dir, exist_ok=True)

    for cfg_scale in cfg_scales:
        cfg_dir = os.path.join(args.output_dir, f"cfg_{cfg_scale:g}")
        os.makedirs(cfg_dir, exist_ok=True)
        print(f"Generating samples for cfg_scale={cfg_scale:g}")

        for idx, (prompt_text, prompt_data) in enumerate(tqdm(zip(prompts, all_prompt_data), total=len(prompts))):
            sample_seed = seed + idx + int(cfg_scale * 1000)
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)

            with autocast:
                latents = lit.sample_latents(
                    prompt_data,
                    cfg_scale=cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                )

            latents = latents.transpose(1, 2)

            metadata = {
                "sample_rate": args.sample_rate,
                "latent_length": latents.shape[-1],
            }
            with torch.autocast(device_type=device.type, enabled=False):
                audio = decode_audio_latents(dacvae_model, latents.float(), metadata)

            audio = audio.detach().cpu()
            if audio.dim() == 3:
                audio = audio[0]

            if isinstance(prompt_text, str):
                safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt_text)[:50]
            else:
                safe_name = f"prompt_{idx}"
            filename = f"{idx:04d}_{safe_name}.mp3"
            filepath = os.path.join(cfg_dir, filename)
            torchaudio.save(filepath, audio, args.sample_rate, format="mp3")
            print(f"Saved: {filepath}")

    print(f"\nGenerated samples in {args.output_dir}")


if __name__ == "__main__":
    main()
