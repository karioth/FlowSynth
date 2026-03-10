import argparse
import csv
import os
import types
from contextlib import nullcontext
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from tangoflux import TangoFluxInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audio with TangoFlux-base from an AudioCaps CSV."
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
        default="declare-lab/TangoFlux-base",
        help="Hugging Face model ID. Use TangoFlux-base for the non-CRPO checkpoint.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of flow/Euler steps. 50 is the recommended quality setting.",
    )
    parser.add_argument(
        "--audio-length",
        type=float,
        default=10.0,
        help="Generated audio length in seconds. Must be in [1, 30].",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale.",
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
        "--resume",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["bf16-mixed", "16-mixed", "32"],
        help="Autocast precision.",
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


def get_autocast_dtype(precision: str, device: torch.device):
    if device.type != "cuda":
        return None
    if precision == "bf16-mixed":
        return torch.bfloat16
    if precision == "16-mixed":
        return torch.float16
    return None


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    # torchaudio.save does not accept bfloat16/float16 tensors.
    audio = audio.detach().to(torch.float32).cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(path), audio, sample_rate)


def patched_encode_text(self, prompt, num_samples_per_prompt=1):
    """Backwards-compatible encode_text for TangoFlux no-CFG path."""
    device = self.text_encoder.device
    batch = self.tokenizer(
        prompt,
        max_length=self.max_text_seq_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)

    encoder_hidden_states = self.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )[0]
    boolean_encoder_mask = (attention_mask == 1).to(device)

    if num_samples_per_prompt != 1:
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(
            num_samples_per_prompt, dim=0
        )
        boolean_encoder_mask = boolean_encoder_mask.repeat_interleave(
            num_samples_per_prompt, dim=0
        )

    return encoder_hidden_states, boolean_encoder_mask


def main() -> None:
    args = parse_args()

    if not (1.0 <= args.audio_length <= 30.0):
        raise ValueError(f"--audio-length must be in [1, 30], got {args.audio_length}")

    rank, world_size, local_rank = get_dist_context(args.local_rank)

    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        if local_rank >= cuda_count:
            raise ValueError(
                f"LOCAL_RANK {local_rank} is out of range for {cuda_count} visible CUDA device(s)."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        device_str = f"cuda:{local_rank}"
    else:
        device = torch.device("cpu")
        device_str = "cpu"

    amp_dtype = get_autocast_dtype(args.precision, device)
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

    print(f"Loading {args.model_id} on {device_str} (precision={args.precision}) ...")
    model = TangoFluxInference(name=args.model_id, device=device_str)
    if "num_samples_per_prompt" not in model.model.encode_text.__code__.co_varnames:
        model.model.encode_text = types.MethodType(patched_encode_text, model.model)

    global_total = len(prompt_items_all)
    global_indices = shard_indices(global_total, rank, world_size)
    prompt_items = [prompt_items_all[idx] for idx in global_indices]
    print(f"[rank {rank}/{world_size}] handling {len(prompt_items)} of {global_total} prompts")

    sample_rate = 44100
    generated = 0
    skipped = 0

    pending_items: list[tuple[int, Path, str]] = []
    for global_idx, (audiocap_id, caption) in zip(global_indices, prompt_items):
        out_path = out_dir / f"{audiocap_id}.wav"
        if args.resume and out_path.exists():
            skipped += 1
            continue
        pending_items.append((global_idx, out_path, caption))

    for global_idx, out_path, caption in tqdm(
        pending_items,
        desc="Generating",
        disable=rank != 0,
    ):
        torch.manual_seed(args.seed + global_idx)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + global_idx)

        with autocast:
            audio = model.generate(
                caption,
                steps=args.steps,
                duration=float(args.audio_length),
                guidance_scale=float(args.guidance_scale),
            )

        save_audio(out_path, audio, sample_rate)
        generated += 1

    print(f"[rank {rank}/{world_size}] Done. generated={generated} skipped={skipped} output_dir={out_dir}")


if __name__ == "__main__":
    main()
