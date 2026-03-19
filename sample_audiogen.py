import argparse
import csv
import os
from pathlib import Path

import torch
import torchaudio
from audiocraft.models import AudioGen
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audio with AudioGen from an AudioCaps CSV."
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
        default="facebook/audiogen-medium",
        help="AudioCraft model ID.",
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
        default=3.0,
        help="Classifier-free guidance scale (AudioGen cfg_coef).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=250,
        help="Top-k sampling value. Use 0 to disable top-k filtering.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top-p sampling value in [0, 1]. Use 0.0 to disable top-p filtering.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--use-sampling",
        dest="use_sampling",
        action="store_true",
        help="Use sampling-based generation (default).",
    )
    sampling_group.add_argument(
        "--no-use-sampling",
        dest="use_sampling",
        action="store_false",
        help="Disable sampling and use greedy decoding.",
    )
    parser.set_defaults(use_sampling=True)
    parser.add_argument(
        "--two-step-cfg",
        action="store_true",
        help="Enable two-step classifier-free guidance.",
    )
    parser.add_argument(
        "--extend-stride",
        type=float,
        default=2.0,
        help="Extension stride in seconds when generating beyond model max duration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only generate the first N prompts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts to generate per forward pass.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
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


def iter_chunks(items: list[tuple[int, Path, str]], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    audio = audio.detach().to(torch.float32).cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(f"Expected rank-2 audio tensor [channels, samples], got shape {tuple(audio.shape)}")
    torchaudio.save(str(path), audio, sample_rate)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = get_dist_context(args.local_rank)

    if args.audio_length <= 0:
        raise ValueError(f"--audio-length must be > 0, got {args.audio_length}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.top_k < 0:
        raise ValueError(f"--top-k must be >= 0, got {args.top_k}")
    if not (0.0 <= args.top_p <= 1.0):
        raise ValueError(f"--top-p must be in [0, 1], got {args.top_p}")
    if args.temperature <= 0:
        raise ValueError(f"--temperature must be > 0, got {args.temperature}")
    if args.extend_stride <= 0:
        raise ValueError(f"--extend-stride must be > 0, got {args.extend_stride}")

    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        if local_rank >= cuda_count:
            raise ValueError(
                f"LOCAL_RANK {local_rank} is out of range for {cuda_count} visible CUDA device(s)."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        device_str = f"cuda:{local_rank}"
        audiocraft_device = "cuda"
    else:
        device = torch.device("cpu")
        device_str = "cpu"
        audiocraft_device = "cpu"

    prompt_items_all = load_audiocaps_csv(args.prompt_csv)
    if args.limit is not None:
        prompt_items_all = prompt_items_all[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} on {device_str} ...")
    model = AudioGen.get_pretrained(args.model_id, device=audiocraft_device)

    max_duration = getattr(model, "max_duration", None)
    if max_duration is not None and args.audio_length > float(max_duration) and args.extend_stride >= float(
        max_duration
    ):
        raise ValueError(
            f"--extend-stride must be < model max duration ({max_duration}) when --audio-length exceeds it."
        )

    model.set_generation_params(
        use_sampling=args.use_sampling,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        duration=args.audio_length,
        cfg_coef=args.guidance_scale,
        two_step_cfg=args.two_step_cfg,
        extend_stride=args.extend_stride,
    )
    sample_rate = int(model.sample_rate)

    global_total = len(prompt_items_all)
    global_indices = shard_indices(global_total, rank, world_size)
    prompt_items = [prompt_items_all[idx] for idx in global_indices]
    print(f"[rank {rank}/{world_size}] handling {len(prompt_items)} of {global_total} prompts")

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
        global_indices_chunk = [global_idx for global_idx, _, _ in chunk]
        out_paths = [out_path for _, out_path, _ in chunk]
        prompts = [caption for _, _, caption in chunk]

        # Batch-level seed keeps runs deterministic for a fixed sharding/chunking setup.
        seed = args.seed + global_indices_chunk[0]
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        generated_audio = model.generate(prompts)
        if generated_audio.ndim != 3 or generated_audio.shape[0] != len(prompts):
            raise RuntimeError(
                f"Expected AudioGen output shape [{len(prompts)}, channels, samples], "
                f"got {tuple(generated_audio.shape)}"
            )

        for out_path, audio in zip(out_paths, generated_audio):
            save_audio(out_path, audio, sample_rate)
            generated += 1

    print(f"[rank {rank}/{world_size}] Done. generated={generated} skipped={skipped} output_dir={out_dir}")


if __name__ == "__main__":
    main()
