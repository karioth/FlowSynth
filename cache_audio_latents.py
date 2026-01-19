import argparse
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from audiotools import AudioSignal


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
DEFAULT_WEIGHTS_REL_PATH = os.path.join("pretrained_models", "dacvae_watermarked.pth")


class AudioFileDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.files = self._list_audio_files(self.root)
        if not self.files:
            raise RuntimeError(f"No audio files found under: {self.root}")

    @staticmethod
    def _list_audio_files(root: Path):
        files = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.startswith("."):
                    continue
                if Path(name).suffix.lower() in AUDIO_EXTS:
                    files.append(Path(dirpath) / name)
        files.sort()
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = torchaudio.load(str(path))  # [C, T]
        wav = wav.clamp_(-1, 1)
        return str(path), wav, int(sr)


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


def get_args_parser():
    parser = argparse.ArgumentParser("Cache DACVAE audio latents", add_help=True)
    parser.add_argument("--data_dir", required=True, type=str,
                        help="Root directory to scan for audio files.")
    parser.add_argument("--cached_path", default=None, type=str,
                        help="Output path for cached latents (default: data_dir + '_cached').")
    parser.add_argument("--weights", default=None, type=str,
                        help="Path to DACVAE weights (defaults to pretrained_models/dacvae_watermarked.pth).")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use for caching.")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batch size per GPU (kept at 1 for variable-length audio).")
    parser.add_argument("--chunk_size_latents", default=128, type=int,
                        help="Chunk size in latent frames.")
    parser.add_argument("--overlap_latents", default=12, type=int,
                        help="Overlap in latent frames (even).")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic CUDA settings (default).")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.set_defaults(deterministic=True)

    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--pin_mem", action="store_true",
                        help="Pin CPU memory in DataLoader for more efficient transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    return parser


def resolve_weights_path(args):
    if args.weights:
        return args.weights
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_WEIGHTS_REL_PATH)
    return default_path


def _configure_determinism(device: str):
    if not device.startswith("cuda"):
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def main(args):
    device = args.device
    distributed, rank, world_size = init_distributed_mode(device)

    if args.deterministic:
        _configure_determinism(device)

    data_dir = os.path.normpath(args.data_dir)
    cached_path = os.path.normpath(args.cached_path or f"{data_dir}_cached")
    os.makedirs(cached_path, exist_ok=True)

    weights_path = resolve_weights_path(args)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    from src.audio_utils import encode_audio_latents
    import dacvae

    model = dacvae.DACVAE.load(weights_path).eval().to(device)

    dataset = AudioFileDataset(data_dir)
    if rank == 0:
        print(f"Found {len(dataset)} files. Cache output: {cached_path}")

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

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

    use_tqdm = rank == 0
    data_iter = loader
    if use_tqdm:
        data_iter = tqdm(loader, total=len(loader), desc="Caching", unit="batch")

    input_root = Path(data_dir).resolve()
    out_root = Path(cached_path).resolve()

    for batch in data_iter:
        for path_str, wav, sr in batch:
            path = Path(path_str).resolve()
            rel = path.relative_to(input_root)
            out_path = (out_root / rel).with_suffix(".pt")
            if out_path.exists():
                continue

            audio = AudioSignal(wav.unsqueeze(0), sr)
            posterior_params, _ = encode_audio_latents(
                model,
                audio,
                chunked=True,
                chunk_size_latents=args.chunk_size_latents,
                overlap_latents=args.overlap_latents,
            )
            tensor = posterior_params[0].to(torch.float32).cpu()

            os.makedirs(out_path.parent, exist_ok=True)
            tmp = str(out_path) + ".tmp"
            torch.save(tensor, tmp)
            os.replace(tmp, out_path)

        if device.startswith("cuda"):
            torch.cuda.synchronize()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
