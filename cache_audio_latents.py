import argparse
import io
import os
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset, Audio
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from audiotools import AudioSignal


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
DEFAULT_WEIGHTS = "facebook/dacvae-watermarked"


class AudioFileDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.files = self._list_audio_files(self.root)
        if not self.files:
            raise RuntimeError(f"No audio files found under: {self.root}")

    @staticmethod
    def _fast_scandir(root, exts):
        exts = {e if e.startswith(".") else f".{e}" for e in exts}
        subdirs, files = [], []
        try:
            for entry in os.scandir(root):
                try:
                    if entry.is_dir():
                        subdirs.append(entry.path)
                    elif entry.is_file():
                        name = entry.name
                        if not name.startswith(".") and Path(name).suffix.lower() in exts:
                            files.append(entry.path)
                except:
                    pass
        except:
            pass
        for d in list(subdirs):
            sd, f = AudioFileDataset._fast_scandir(d, exts)
            subdirs.extend(sd)
            files.extend(f)
        return subdirs, files

    @staticmethod
    def _list_audio_files(root: Path):
        _, files = AudioFileDataset._fast_scandir(str(root), AUDIO_EXTS)
        files = [Path(p) for p in files]
        files.sort()
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = torchaudio.load(str(path))  # [C, T]
        wav = wav.clamp_(-1, 1)
        return str(path), wav, int(sr)


class HFParquetAudioDataset(Dataset):
    """
    Loads decoded waveforms from HF parquet shards (no MP3 extraction).
    Returns (uid, wav[C,T], sr) - same format as AudioFileDataset.
    """
    def __init__(self, hf_name: str, split: str = "train", data_dir: str = "data", cache_dir: str | None = None):
        self.ds = load_dataset(hf_name, data_dir=data_dir, split=split, cache_dir=cache_dir)
        # CRITICAL: Disable automatic decoding to avoid torchcodec
        self.ds = self.ds.cast_column("audio", Audio(decode=False))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        a = ex["audio"]  # dict with "bytes" and "path"

        # Decode MP3 bytes with soundfile (not torchcodec)
        try:
            arr, sr = sf.read(io.BytesIO(a["bytes"]))  # arr is (T, C) numpy array
        except Exception as e:
            uid = a.get("path", f"{idx:09d}")
            print(f"[WARNING] Skipping corrupted audio {uid}: {e}")
            # Return next valid item (wrap around if needed)
            return self.__getitem__((idx + 1) % len(self))

        # Convert to torch tensor [C, T] to match torchaudio.load() format
        x = torch.from_numpy(arr).to(torch.float32)

        # soundfile returns (T,) for mono, (T, C) for stereo
        # Convert to [C, T] to match AudioFileDataset format
        if x.ndim == 1:
            x = x.unsqueeze(0)  # (T,) -> (1, T)
        elif x.ndim == 2:
            x = x.transpose(0, 1)  # (T, C) -> (C, T)

        x = x.clamp_(-1, 1)
        uid = a.get("path", f"{idx:09d}")

        # Skip audio shorter than 10 seconds
        MIN_DURATION_SECONDS = 10
        min_samples = MIN_DURATION_SECONDS * sr
        if x.shape[-1] < min_samples:
            duration = x.shape[-1] / sr
            print(f"[WARNING] Skipping too-short audio {uid}: {duration:.2f}s (< {MIN_DURATION_SECONDS}s)")
            return self.__getitem__((idx + 1) % len(self))

        # Return same format as AudioFileDataset: (str, tensor[C,T], int)
        return str(uid), x, int(sr)


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
    parser.add_argument("--hf_dataset", default=None, type=str,
                        help="If set, load audio from this HF dataset id instead of scanning --data_dir.")
    parser.add_argument("--hf_split", default="train", type=str,
                        help="HF dataset split to use (default: train).")
    parser.add_argument("--hf_data_dir", default="data", type=str,
                        help="Subdir inside the HF dataset repo that contains parquet shards (JamendoMaxCaps uses 'data').")
    parser.add_argument("--hf_cache_dir", default=None, type=str,
                        help="Cache directory for HF datasets.")
    parser.add_argument("--cached_path", default=None, type=str,
                        help="Output path for cached latents (default: data_dir + '_cached').")
    parser.add_argument("--weights", default=None, type=str,
                        help="Path or HF id for DACVAE weights (default: facebook/dacvae-watermarked).")
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
    return DEFAULT_WEIGHTS


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

    weights_ref = resolve_weights_path(args)
    if os.path.exists(weights_ref):
        weights_path = weights_ref
    elif weights_ref.startswith("facebook/"):
        weights_path = weights_ref
    else:
        raise FileNotFoundError(f"Missing weights: {weights_ref}")

    from src.audio_utils import encode_audio_latents
    import dacvae

    model = dacvae.DACVAE.load(weights_path).eval().to(device)

    out_root = Path(cached_path).resolve()

    if args.hf_dataset:
        dataset = HFParquetAudioDataset(
            args.hf_dataset, split=args.hf_split, data_dir=args.hf_data_dir, cache_dir=args.hf_cache_dir
        )
        input_root = None
        if rank == 0:
            print(f"HF dataset: {args.hf_dataset} ({args.hf_split}), n={len(dataset)}. Cache output: {cached_path}")
    else:
        dataset = AudioFileDataset(data_dir)
        input_root = Path(data_dir).resolve()
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

    for batch in data_iter:
        for path_str, wav, sr in batch:
            if input_root is None:
                # HF dataset: flat structure for easy training access
                stem = Path(path_str).stem
                out_path = out_root / f"{stem}.pt"
            else:
                # Local files: preserve directory structure
                path = Path(path_str).resolve()
                rel = path.relative_to(input_root)
                out_path = (out_root / rel).with_suffix(".pt")

            if out_path.exists():
                continue

            audio = AudioSignal(wav.unsqueeze(0), sr)
            posterior_params, metadata = encode_audio_latents(
                model,
                audio,
                chunked=True,
                chunk_size_latents=args.chunk_size_latents,
                overlap_latents=args.overlap_latents,
            )
            tensor = posterior_params[0].to(torch.float32).cpu()
            latent_length = int(metadata.get("latent_length", tensor.shape[-1]))
            payload = {
                "posterior_params": tensor,
                "latent_length": latent_length,
            }

            os.makedirs(out_path.parent, exist_ok=True)
            tmp = str(out_path) + ".tmp"
            torch.save(payload, tmp)
            os.replace(tmp, out_path)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
