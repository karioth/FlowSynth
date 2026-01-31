import argparse
import os
import random
from contextlib import nullcontext
from typing import List, Tuple

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.lightning import LitModule
from src.data_utils.preprocess.img_items import load_vae, sequence_to_image

DEFAULT_CLASS_LABELS = [281, 282, 283, 284, 285, 4, 7, 963]


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images from a Lightning checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt.")
    parser.add_argument("--vae", type=str, default=None, help="Path to VAE checkpoint.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["bf16-mixed", "16-mixed", "32"],
        help="Autocast precision (matches training defaults).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Deprecated: use --precision instead.",
    )
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ardiff_step", type=int, default=None)
    parser.add_argument("--base_num_frames", type=int, default=None)
    parser.add_argument("--num_images", type=int, default=None, help="Sample N random classes.")
    parser.add_argument("--class_labels", type=str, default=None, help="Comma-separated class list.")
    parser.add_argument("--output_dir", type=str, default="visuals")
    parser.add_argument("--speed-only", action="store_true", help="Run sampling only, no decode or saving.")
    parser.add_argument("--speed-iters", type=int, default=5)
    return parser.parse_args()


def init_distributed() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def get_dtype(mixed_precision: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def resolve_precision(args) -> str:
    precision = args.precision
    if args.mixed_precision is None:
        return precision
    mapping = {
        "no": "32",
        "fp16": "16-mixed",
        "bf16": "bf16-mixed",
    }
    return mapping[args.mixed_precision]


def get_autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if precision == "bf16-mixed":
        return torch.bfloat16
    if precision == "16-mixed":
        return torch.float16
    return None


def parse_class_labels(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x.strip()]


def shard_indices(total: int, rank: int, world_size: int) -> List[int]:
    return list(range(rank, total, world_size))


def build_labels(
    args,
    num_classes: int,
    rank: int,
    world_size: int,
) -> Tuple[List[int], List[int]]:
    if args.class_labels:
        all_labels = parse_class_labels(args.class_labels)
        indices = shard_indices(len(all_labels), rank, world_size)
        labels = [all_labels[i] for i in indices]
        return labels, indices

    if args.num_images is None:
        indices = shard_indices(len(DEFAULT_CLASS_LABELS), rank, world_size)
        labels = [DEFAULT_CLASS_LABELS[i] for i in indices]
        return labels, indices

    counts = [args.num_images // world_size + (r < args.num_images % world_size) for r in range(world_size)]
    local_count = counts[rank]
    start = sum(counts[:rank])
    indices = list(range(start, start + local_count))
    labels = torch.randint(0, num_classes, (local_count,)).tolist()
    return labels, indices


@torch.no_grad()
def main():
    args = parse_args()
    rank, world_size, _ = init_distributed()

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(args)
    amp_dtype = get_autocast_dtype(precision, device)
    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )

    lit = LitModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    lit.to(device=device)
    lit.eval()

    labels, indices = build_labels(args, lit.hparams.num_classes, rank, world_size)
    if len(labels) == 0:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    if args.speed_only:
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.speed_iters):
                batch_labels = torch.randint(0, lit.hparams.num_classes, (args.batch_size,), device=device)
                with autocast:
                    _ = lit.sample_latents(
                        batch_labels,
                        cfg_scale=args.cfg_scale,
                        num_inference_steps=args.num_inference_steps,
                        ardiff_step=args.ardiff_step,
                        base_num_frames=args.base_num_frames,
                    )
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            images = args.speed_iters * args.batch_size
            if rank == 0:
                print(f"Time: {elapsed_ms / 1000:.2f}s, FPS: {images / (elapsed_ms / 1000):.2f}")
        else:
            import time
            start = time.time()
            for _ in range(args.speed_iters):
                batch_labels = torch.randint(0, lit.hparams.num_classes, (args.batch_size,), device=device)
                with autocast:
                    _ = lit.sample_latents(
                        batch_labels,
                        cfg_scale=args.cfg_scale,
                        num_inference_steps=args.num_inference_steps,
                        ardiff_step=args.ardiff_step,
                        base_num_frames=args.base_num_frames,
                    )
            elapsed = time.time() - start
            images = args.speed_iters * args.batch_size
            if rank == 0:
                print(f"Time: {elapsed:.2f}s, FPS: {images / elapsed:.2f}")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return

    if args.vae is None:
        raise ValueError("--vae is required unless --speed-only is set.")
    vae, _, _, _ = load_vae(args.vae, args.image_size)
    vae.to(device=device)
    vae.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    total_batches = (len(labels) + args.batch_size - 1) // args.batch_size
    batch_iter = range(0, len(labels), args.batch_size)
    if rank == 0:
        batch_iter = tqdm(batch_iter, total=total_batches, desc="Sampling", unit="batch")
    for start_idx in batch_iter:
        batch_labels = labels[start_idx:start_idx + args.batch_size]
        batch_indices = indices[start_idx:start_idx + args.batch_size]
        batch_labels_tensor = torch.tensor(batch_labels, device=device, dtype=torch.long)

        with autocast:
            latents = lit.sample_latents(
                batch_labels_tensor,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                ardiff_step=args.ardiff_step,
                base_num_frames=args.base_num_frames,
            )
        with torch.autocast(device_type=device.type, enabled=False):
            images = vae.decode(sequence_to_image(latents.float()))

        for img, label, idx in zip(images, batch_labels, batch_indices):
            filename = f"{idx:06d}_class{label}.png"
            save_path = os.path.join(args.output_dir, filename)
            save_image(img, save_path, normalize=True, value_range=(-1, 1))

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if rank == 0:
        print(f"Saved samples to {args.output_dir}")


if __name__ == "__main__":
    main()
