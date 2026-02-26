import argparse
import inspect
import json
import os
import types
from pathlib import Path

import torch
from diffusers import AudioLDM2Pipeline
from scipy.io import wavfile
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audio with AudioLDM2 from a JSON filename->caption mapping."
    )
    parser.add_argument(
        "--prompt-json",
        type=str,
        required=True,
        help="Path to JSON mapping: output_filename.wav -> caption.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write generated .wav files.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="cvssp/audioldm2",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of diffusion steps (higher = slower, usually better quality).",
    )
    parser.add_argument(
        "--audio-length",
        type=float,
        default=10.0,
        help="Generated audio length in seconds.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only generate the first N prompts.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts to generate per forward pass.",
    )
    return parser.parse_args()


def load_mapping(path: str) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("prompt JSON must be a dict of filename -> caption")

    items: list[tuple[str, str]] = []
    for filename, caption in data.items():
        if not isinstance(filename, str):
            raise ValueError(f"filename key must be str, got {type(filename).__name__}")
        if not isinstance(caption, str):
            raise ValueError(f"caption value for {filename!r} must be str")
        items.append((filename, caption))
    return items


def choose_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def iter_chunks(items: list[tuple[int, Path, str]], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def load_pipeline(model_id: str, dtype: torch.dtype) -> AudioLDM2Pipeline:
    # diffusers transitioned from `torch_dtype` to `dtype`; support both.
    kwargs = {"dtype": dtype}
    if "dtype" not in inspect.signature(AudioLDM2Pipeline.from_pretrained).parameters:
        kwargs = {"torch_dtype": dtype}
    return AudioLDM2Pipeline.from_pretrained(model_id, **kwargs)


def patch_language_model_generation(pipe: AudioLDM2Pipeline) -> None:
    language_model = getattr(pipe, "language_model", None)
    if language_model is None or hasattr(language_model, "_get_initial_cache_position"):
        return

    print("Applying GPT2Model compatibility fallback for language-model rollout.")

    def _compat_generate_language_model(self, inputs_embeds=None, max_new_tokens=8, **model_kwargs):
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided")

        lm = self.language_model
        if max_new_tokens is None:
            max_new_tokens = getattr(getattr(lm, "config", None), "max_new_tokens", 8)
        max_new_tokens = int(max_new_tokens)

        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if max_new_tokens == 0:
            return inputs_embeds[:, :0, :]

        attention_mask = model_kwargs.get("attention_mask")
        for _ in range(max_new_tokens):
            output = lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            next_hidden = output.hidden_states[-1][:, -1:, :]
            inputs_embeds = torch.cat([inputs_embeds, next_hidden], dim=1)

            if attention_mask is not None:
                next_mask = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, next_mask], dim=1)

        return inputs_embeds[:, -max_new_tokens:, :]

    pipe.generate_language_model = types.MethodType(_compat_generate_language_model, pipe)


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")

    device, dtype = choose_device()
    prompt_items = load_mapping(args.prompt_json)
    if args.limit is not None:
        prompt_items = prompt_items[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} on {device} ({dtype}) ...")
    pipe = load_pipeline(args.model_id, dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    patch_language_model_generation(pipe)

    sample_rate = 16000
    generated = 0
    skipped = 0

    pending_items: list[tuple[int, Path, str]] = []
    for idx, (filename, caption) in enumerate(prompt_items):
        out_path = out_dir / os.path.basename(filename)
        if out_path.suffix.lower() != ".wav":
            out_path = out_path.with_suffix(".wav")

        if args.resume and out_path.exists():
            skipped += 1
            continue

        pending_items.append((idx, out_path, caption))

    num_batches = (len(pending_items) + args.batch_size - 1) // args.batch_size
    for chunk in tqdm(
        iter_chunks(pending_items, args.batch_size),
        total=num_batches,
        desc="Generating",
    ):
        prompts = [caption for _, _, caption in chunk]
        out_paths = [out_path for _, out_path, _ in chunk]
        generators = [
            torch.Generator(device=device).manual_seed(args.seed + idx)
            for idx, _, _ in chunk
        ]

        output = pipe(
            prompts,
            num_inference_steps=args.steps,
            audio_length_in_s=args.audio_length,
            guidance_scale=args.guidance_scale,
            generator=generators,
        )
        audios = output.audios
        if len(audios) != len(out_paths):
            raise RuntimeError(f"Pipeline returned {len(audios)} audios for {len(out_paths)} prompts")

        for out_path, audio in zip(out_paths, audios):
            # Match AudioLDM2 examples: write float waveform directly.
            wavfile.write(out_path, sample_rate, audio)
            generated += 1

    print(f"Done. generated={generated} skipped={skipped} output_dir={out_dir}")


if __name__ == "__main__":
    main()
