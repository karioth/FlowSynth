import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.data_utils.audio_utils import scan_cached_outputs
from src.data_utils.preencode_audio_latents import cache_audio_latents


class HFParquetAudioDataset(Dataset):
    """
    Loads decoded waveforms from HF dataset (audio decoded by datasets).
    Returns (uid, wav[C,T], sr) or None if skipped or already cached.
    """
    def __init__(
        self,
        hf_name: str,
        split: str,
        data_dir: str,
        cache_dir: str | None,
        out_root: Path,
        min_duration_seconds: float,
        done_cache: set[str] | None = None,
    ):
        self.hf_name = hf_name
        self.split = split
        self.ds = load_dataset(hf_name, data_dir=data_dir, split=split, cache_dir=cache_dir)
        self.out_root = out_root
        self.min_duration_seconds = float(min_duration_seconds)
        self.done = done_cache or set()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        a = ex["audio"]  # decoded dict with "array", "sampling_rate", optional "path"
        uid = f"{idx:09d}"
        out_rel = f"{uid}.pt"
        if out_rel in self.done:
            return None
        out_path = self.out_root / out_rel

        arr = a["array"]  # (T,) or (T,C) float32/float64 np array
        sr = int(a["sampling_rate"])

        x = torch.from_numpy(arr).to(torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)          # (T,) -> (1, T)
        else:
            x = x.transpose(0, 1)        # (T, C) -> (C, T)

        x = x.clamp_(-1, 1)

        if x.shape[-1] < self.min_duration_seconds * float(sr):
            return None

        return str(uid), x, sr, str(out_path)




def init_distributed_mode(device: str):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() and device.startswith("cuda") else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        return True, rank, world_size
    return False, 0, 1


def _configure_determinism(device: str):
    if not device.startswith("cuda"):
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args_parser():
    parser = argparse.ArgumentParser("Cache DACVAE audio latents (HF dataset)", add_help=True)
    parser.add_argument("--hf_dataset", required=True, type=str)
    parser.add_argument("--hf_split", default="train", type=str)
    parser.add_argument("--hf_data_dir", default="data", type=str)
    parser.add_argument("--hf_cache_dir", default=None, type=str)
    parser.add_argument("--cached_path", default=None, type=str)
    parser.add_argument("--weights", default="facebook/dacvae-watermarked", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--chunk_size_latents", default=1024, type=int)
    parser.add_argument("--overlap_latents", default=12, type=int)
    parser.add_argument("--min_duration_seconds", default=10.0, type=float)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    return parser


@torch.no_grad()
def main(args):
    device = args.device
    distributed, rank, world_size = init_distributed_mode(device)

    if args.deterministic:
        _configure_determinism(device)

    if args.cached_path:
        cached_path = os.path.normpath(args.cached_path)
    else:
        dataset_tag = args.hf_dataset.replace("/", "_")
        cached_path = os.path.normpath(f"{dataset_tag}_{args.hf_split}_cached")
    os.makedirs(cached_path, exist_ok=True)

    weights_ref = args.weights
    if os.path.exists(weights_ref):
        weights_path = weights_ref
    elif weights_ref.startswith("facebook/"):
        weights_path = weights_ref
    else:
        raise FileNotFoundError(f"Missing weights: {weights_ref}")

    import dacvae

    model = dacvae.DACVAE.load(weights_path).eval().to(device)
    out_root = Path(cached_path).resolve()
    os.makedirs(out_root, exist_ok=True)

    if rank == 0:
        print(f"Scanning cache output for existing .pt files under: {out_root}")
    done_cache = scan_cached_outputs(out_root)
    if rank == 0:
        print(f"Found {len(done_cache)} cached files.")

    dataset = HFParquetAudioDataset(
        args.hf_dataset,
        split=args.hf_split,
        data_dir=args.hf_data_dir,
        cache_dir=args.hf_cache_dir,
        out_root=out_root,
        min_duration_seconds=args.min_duration_seconds,
        done_cache=done_cache,
    )
    if rank == 0:
        print(f"HF dataset: {args.hf_dataset} ({args.hf_split}), n={len(dataset)}. Cache output: {cached_path}")

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
        collate_fn=lambda b: b,
    )

    data_iter = tqdm(loader, total=len(loader), desc="Caching", unit="batch") if rank == 0 else loader

    cache_audio_latents(
        data_iter,
        model,
        chunk_size_latents=args.chunk_size_latents,
        overlap_latents=args.overlap_latents,
        rank=rank,
    )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
