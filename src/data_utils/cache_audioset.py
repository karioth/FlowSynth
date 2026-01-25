import argparse
import io
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from datasets import load_dataset, Audio
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

from audiotools import AudioSignal


DEFAULT_DACVAE_WEIGHTS = "facebook/dacvae-watermarked"
DEFAULT_CLAP_MODEL = "laion/larger_clap_music"
# DACVAE (sample_rate=48000, hop_length=1920):
# latent_length = ceil((duration_seconds * sample_rate) / hop_length)
# Require latent_length >= 251 => duration_seconds >= (250 * hop_length + 1) / sample_rate
MIN_DURATION_SECONDS = (250 * 1920 + 1) / 48000


class AudioSetDataset(Dataset):
    """
    Loads audio and captions from laion/audioset-with-captions.
    Returns (uid, wav[C,T], sr, caption) tuples, or None for skipped samples.
    """
    def __init__(
        self,
        hf_name: str = "laion/audioset-with-captions",
        split: str = "train",
        cache_dir: str | None = None,
    ):
        self.ds = load_dataset(hf_name, split=split, cache_dir=cache_dir)
        # Disable automatic decoding to avoid torchcodec issues
        self.ds = self.ds.cast_column("mp3", Audio(decode=False))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """Returns (uid, wav, sr, caption) or None if sample should be skipped."""
        try:
            ex = self.ds[idx]
        except Exception as e:
            print(f"[WARNING] Failed to load example {idx}: {e}")
            return None

        # Get audio bytes
        audio_data = ex.get("mp3")  # dict with "bytes" and "path"
        uid = ex.get("__key__", f"{idx:09d}")
        if not isinstance(audio_data, dict):
            print(f"[WARNING] Missing mp3 for {uid}, skipping")
            return None
        if audio_data.get("path"):
            uid = audio_data.get("path", uid)
        uid = Path(uid).stem  # Remove any extension

        # Decode MP3 bytes with soundfile
        audio_bytes = audio_data.get("bytes")
        if audio_bytes is None:
            path = audio_data.get("path")
            if path and os.path.isfile(path):
                try:
                    arr, sr = sf.read(path)
                except Exception as e:
                    print(f"[WARNING] Skipping unreadable audio {uid}: {e}")
                    return None
            else:
                print(f"[WARNING] Missing audio bytes for {uid}, skipping")
                return None
        else:
            try:
                arr, sr = sf.read(io.BytesIO(audio_bytes))
            except Exception as e:
                print(f"[WARNING] Skipping corrupted audio {uid}: {e}")
                return None

        # Convert to torch tensor [C, T]
        x = torch.from_numpy(arr).to(torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)  # (T,) -> (1, T)
        elif x.ndim == 2:
            x = x.transpose(0, 1)  # (T, C) -> (C, T)
        x = x.clamp_(-1, 1)

        # Skip audio shorter than minimum duration
        min_samples = MIN_DURATION_SECONDS * sr
        if x.shape[-1] < min_samples:
            duration = x.shape[-1] / sr
            print(f"[WARNING] Skipping too-short audio {uid}: {duration:.2f}s (< {MIN_DURATION_SECONDS}s)")
            return None

        # Extract caption from JSON metadata
        json_data = ex.get("json", {})
        if isinstance(json_data, (bytes, bytearray)):
            try:
                json_data = json.loads(json_data.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                json_data = {}
        elif isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError:
                json_data = {}
        elif not isinstance(json_data, dict):
            json_data = {}
        caption = json_data.get("comprehensive_caption", "")
        if not caption:
            # Fallback to other caption fields
            caption = json_data.get("caption", "")
            print(f"[WARNING] falling to simpler caption found for {uid}")
        if not caption:
            print(f"[WARNING] No caption found for {uid}, skipping")
            return None

        return uid, x, int(sr), caption


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
    parser = argparse.ArgumentParser(
        "Cache DACVAE audio latents and CLAP text embeddings for AudioSet",
        add_help=True
    )
    # Dataset arguments
    parser.add_argument("--hf_dataset", default="laion/audioset-with-captions", type=str,
                        help="HuggingFace dataset ID")
    parser.add_argument("--hf_split", default="train", type=str,
                        help="Dataset split (default: train)")
    parser.add_argument("--hf_cache_dir", default=None, type=str,
                        help="Cache directory for HF datasets")

    # Output directory
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output directory for cached .pt and .json files")

    # Model arguments
    parser.add_argument("--dacvae_weights", default=None, type=str,
                        help=f"Path or HF ID for DACVAE weights (default: {DEFAULT_DACVAE_WEIGHTS})")
    parser.add_argument("--clap_model", default=DEFAULT_CLAP_MODEL, type=str,
                        help=f"HuggingFace model ID for CLAP (default: {DEFAULT_CLAP_MODEL})")

    # DACVAE encoding arguments
    parser.add_argument("--chunk_size_latents", default=256, type=int,
                        help="Chunk size in latent frames for DACVAE")
    parser.add_argument("--overlap_latents", default=12, type=int,
                        help="Overlap in latent frames (even)")

    # Processing arguments
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batch size per GPU")
    parser.add_argument("--num_workers", default=6, type=int,
                        help="Number of DataLoader workers")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic CUDA settings")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--pin_mem", action="store_true",
                        help="Pin CPU memory in DataLoader")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    return parser


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

    # Create output directory
    output_dir = Path(args.output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    # Load DACVAE model
    dacvae_weights = args.dacvae_weights or DEFAULT_DACVAE_WEIGHTS
    if rank == 0:
        print(f"Loading DACVAE model: {dacvae_weights}")

    from src.data_utils.audio_utils import encode_audio_latents
    import dacvae

    dacvae_model = dacvae.DACVAE.load(dacvae_weights).eval().to(device)

    # Load CLAP model (import lazily to avoid torchcodec conflicts)
    if rank == 0:
        print(f"Loading CLAP model: {args.clap_model}")

    from transformers import ClapModel, ClapProcessor
    clap_model = ClapModel.from_pretrained(args.clap_model, use_safetensors=True).eval().to(device)
    clap_processor = ClapProcessor.from_pretrained(args.clap_model)

    # Load dataset
    if rank == 0:
        print(f"Loading dataset: {args.hf_dataset} ({args.hf_split})")

    dataset = AudioSetDataset(
        hf_name=args.hf_dataset,
        split=args.hf_split,
        cache_dir=args.hf_cache_dir,
    )

    if rank == 0:
        print(f"Dataset size: {len(dataset):,}")
        print(f"Output directory: {output_dir}")

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
        for item in batch:
            if item is None:
                continue
            uid, wav, sr, caption = item
            pt_path = output_dir / f"{uid}.pt"
            json_path = output_dir / f"{uid}.json"

            # Skip if already processed
            if pt_path.exists() and json_path.exists():
                continue

            # Encode audio with DACVAE
            audio = AudioSignal(wav.unsqueeze(0), sr)
            posterior_params, metadata = encode_audio_latents(
                dacvae_model,
                audio,
                chunked=True,
                chunk_size_latents=args.chunk_size_latents,
                overlap_latents=args.overlap_latents,
            )
            audio_tensor = posterior_params[0].to(torch.float32).cpu()
            latent_length = int(metadata.get("latent_length", audio_tensor.shape[-1]))

            # Encode text with CLAP
            inputs = clap_processor(
                text=[caption],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            text_embed = clap_model.get_text_features(**inputs)

            # Save combined .pt file
            payload = {
                "posterior_params": audio_tensor,
                "latent_length": latent_length,
                "text_embedding": text_embed[0].cpu().to(torch.float32),
            }
            tmp = str(pt_path) + ".tmp"
            torch.save(payload, tmp)
            os.replace(tmp, pt_path)

            # Save JSON sidecar with caption
            json_payload = {"caption": caption}
            tmp = str(json_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(json_payload, f)
            os.replace(tmp, json_path)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
