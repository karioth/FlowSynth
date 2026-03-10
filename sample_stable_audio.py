import argparse
import csv
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers import StableAudioPipeline
from scipy.io import wavfile
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audio with Stable Audio Open from an AudioCaps CSV."
    )
    parser.add_argument(
        "--prompt-csv",
        type=str,
        required=True,
        help=(
            "Path to AudioCaps CSV with columns audiocap_id,youtube_id,caption. "
            "Outputs are saved as {audiocap_id}.wav."
        ),
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
        default="stabilityai/stable-audio-open-1.0",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of denoising steps.",
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
        default=7.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, average quality",
        help="Optional negative prompt.",
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
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["bf16-mixed", "16-mixed", "32"],
        help="Autocast / weight precision.",
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_audiocaps_csv(path: str) -> list[tuple[int, str]]:
    required_columns = {"audiocap_id", "youtube_id", "caption"}
    items: list[tuple[int, str]] = []
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
            items.append((audiocap_id, caption))

    if not items:
        raise ValueError("Prompt CSV is empty.")
    return items


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


def get_autocast_dtype(precision: str, device: torch.device):
    if device.type != "cuda":
        return None
    if precision == "bf16-mixed":
        return torch.bfloat16
    if precision == "16-mixed":
        return torch.float16
    return None


def get_weight_dtype(precision: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if precision == "bf16-mixed":
        return torch.bfloat16
    if precision == "16-mixed":
        return torch.float16
    return torch.float32


def iter_chunks(items: list[tuple[int, Path, str]], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = get_dist_context(args.local_rank)

    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.audio_length <= 0:
        raise ValueError(f"--audio-length must be > 0, got {args.audio_length}")

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
    weight_dtype = get_weight_dtype(args.precision, device)
    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )

    prompt_items_all = load_audiocaps_csv(args.prompt_csv)
    if args.limit is not None:
        prompt_items_all = prompt_items_all[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} on {device} (weights={weight_dtype}, precision={args.precision}) ...")
    pipe = StableAudioPipeline.from_pretrained(args.model_id, torch_dtype=weight_dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    global_total = len(prompt_items_all)
    global_indices = shard_indices(global_total, rank, world_size)
    prompt_items = [prompt_items_all[idx] for idx in global_indices]
    print(f"[rank {rank}/{world_size}] handling {len(prompt_items)} of {global_total} prompts")

    sample_rate = pipe.vae.sampling_rate
    generated = 0
    skipped = 0

    pending_items: list[tuple[int, Path, str]] = []
    for global_idx, (audiocap_id, caption) in zip(global_indices, prompt_items):
        out_path = out_dir / f"{audiocap_id}.wav"

        if args.resume and out_path.exists():
            skipped += 1
            continue

        pending_items.append((global_idx, out_path, caption))

    num_batches = (len(pending_items) + args.batch_size - 1) // args.batch_size
    for chunk in tqdm(
        iter_chunks(pending_items, args.batch_size),
        total=num_batches,
        desc="Generating",
        disable=rank != 0,
    ):
        prompts = [caption for _, out_path, caption in chunk]
        out_paths = [out_path for _, out_path, _ in chunk]
        generators = [
            torch.Generator(device=device).manual_seed(args.seed + idx)
            for idx, _, _ in chunk
        ]
        negative_prompts = [args.negative_prompt] * len(prompts)

        with autocast:
            output = pipe(
                prompt=prompts,
                negative_prompt=negative_prompts,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                audio_start_in_s=0.0,
                audio_end_in_s=args.audio_length,
                generator=generators,
                output_type="pt",
            )

        audios = output.audios
        if len(audios) != len(out_paths):
            raise RuntimeError(f"Pipeline returned {len(audios)} audios for {len(out_paths)} prompts")

        for out_path, audio in zip(out_paths, audios):
            # StableAudioPipeline returns [channels, samples]; scipy expects [samples, channels].
            audio_np = audio.detach().float().cpu().numpy().T
            wavfile.write(out_path, sample_rate, audio_np)
            generated += 1

    print(f"[rank {rank}/{world_size}] Done. generated={generated} skipped={skipped} output_dir={out_dir}")


if __name__ == "__main__":
    main()
