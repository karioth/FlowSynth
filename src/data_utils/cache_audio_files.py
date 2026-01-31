import argparse
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.data_utils.audio_utils import list_audio_files, scan_cached_outputs
from src.data_utils.preencode_audio_latents import cache_audio_latents


class AudioFileDataset(Dataset):
    def __init__(
        self,
        root: str,
        input_root: Path,
        out_root: Path,
        min_duration_seconds: float,
        max_duration_seconds: float,
        done_cache: set[str] | None = None,
    ):
        self.root = Path(root).resolve()
        self.input_root = input_root
        self.out_root = out_root
        self.min_duration_seconds = float(min_duration_seconds)
        self.max_duration_seconds = float(max_duration_seconds)
        self.files = list_audio_files(self.root)
        self.done = done_cache or set()
        if not self.files:
            raise RuntimeError(f"No audio files found under: {self.root}")

    def _non_silence_ratio(self, wav: torch.Tensor) -> float:
        non_silent = wav.abs() > 1e-4
        if non_silent.ndim == 2:
            non_silent = non_silent.any(dim=0)
        return non_silent.float().mean().item()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx].resolve()
        rel = path.relative_to(self.input_root)              # preserve subtree
        out_rel = rel.with_suffix(".pt")                     # NOT flat
        if out_rel.as_posix() in self.done:
            return None
        out_path = self.out_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wav, sr = torchaudio.load(str(path))  # [C, T]
        wav = wav.clamp_(-1, 1)

        num_frames = wav.shape[-1]
        min_frames = int(self.min_duration_seconds * float(sr))
        if num_frames < min_frames:
            print(f"too short, skipping: {path}")
            return None
        max_frames = int(self.max_duration_seconds * float(sr))
        if max_frames > 0 and num_frames > max_frames:
            print(f"file too long, taking random 10min crop: {path}")
            max_start = num_frames - max_frames
            found = False
            for _ in range(10):
                start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
                seg = wav[:, start : start + max_frames]
                if self._non_silence_ratio(seg) >= 0.7:
                    wav = seg
                    found = True
                    break
            if not found:
                wav = seg

        return str(path), wav, int(sr), str(out_path)


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
    parser = argparse.ArgumentParser("Cache DACVAE audio latents (files)", add_help=True)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--cached_path", default=None, type=str)
    parser.add_argument("--weights", default="facebook/dacvae-watermarked", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--chunk_size_latents", default=1024, type=int)
    parser.add_argument("--overlap_latents", default=12, type=int)
    parser.add_argument("--min_duration_seconds", default=0.05, type=float)
    parser.add_argument("--max_duration_seconds", default=600.0, type=float)
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

    data_dir = os.path.normpath(args.data_dir)
    cached_path = os.path.normpath(args.cached_path or f"{data_dir}_cached")
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

    input_root = Path(data_dir).resolve()
    if rank == 0:
        print(f"Scanning already cached .pt files under: {out_root}")
    done_cache = scan_cached_outputs(out_root)
    if rank == 0:
        print(f"Found {len(done_cache)} cached files.")
    dataset = AudioFileDataset(
        data_dir,
        input_root=input_root,
        out_root=out_root,
        min_duration_seconds=args.min_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
        done_cache=done_cache,
    )
    if rank == 0:
        print(f"Found {len(dataset)} files. Cache output: {cached_path}")

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

    data_iter = tqdm(loader, total=len(loader), desc="Caching", unit="batch") #if rank == 0 else loader

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
