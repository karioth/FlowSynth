import argparse
import json
import os
import random

import torch
import torchaudio
import dacvae


DEFAULT_IN_DIR = "/share/users/student/f/friverossego/datasets/AudioCaps/train/latents"
DEFAULT_OUT_DIR = "/share/users/student/f/friverossego/LatentLM/smoketest"


def _save_audio(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    wav = wav.detach().cpu()
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    if wav.dim() != 3 or wav.size(0) != 1:
        raise ValueError(f"Expected shape (1, C, T), got {tuple(wav.shape)}")
    torchaudio.save(path, wav[0], sample_rate=sample_rate, format="mp3")


def _read_caption_from_manifest(manifest_path: str | None, target_id: str) -> tuple[str | None, str]:
    if not manifest_path or not os.path.exists(manifest_path):
        return None, "manifest_missing"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                key = entry.get("key")
                if key is None:
                    key = entry.get("id")
                if key is None:
                    continue
                key_str = str(key)
                key_base = os.path.splitext(os.path.basename(key_str))[0]
                if key_str == target_id or key_base == target_id:
                    caption = entry.get("caption")
                    if isinstance(caption, str) and caption:
                        return caption, "found"
                    return None, "caption_missing"
    except OSError:
        return None, "manifest_unreadable"
    return None, "id_not_found"


def _infer_manifest_path(in_path: str | None, in_dir: str | None) -> str | None:
    if in_path:
        latents_dir = os.path.dirname(os.path.abspath(in_path))
        return os.path.join(os.path.dirname(latents_dir), "manifest.jsonl")
    if in_dir:
        dir_path = os.path.abspath(in_dir)
        if os.path.basename(dir_path) == "latents":
            return os.path.join(os.path.dirname(dir_path), "manifest.jsonl")
        return os.path.join(dir_path, "manifest.jsonl")
    return None


def _sample_pt_from_dir(dir_path: str) -> str:
    choice = None
    count = 0
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if not entry.name.endswith(".pt"):
                    continue
                count += 1
                if random.randrange(count) == 0:
                    choice = entry.path
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input dir not found: {dir_path}") from exc
    if choice is None:
        raise FileNotFoundError(f"No .pt files found in: {dir_path}")
    return choice


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Decode cached DACVAE latents to audio.")
    parser.add_argument("--in_path", default=None, type=str,
                        help="Path to cached .pt file (if omitted, sample from --in_dir)")
    parser.add_argument("--in_dir", default=DEFAULT_IN_DIR, type=str,
                        help="Directory to sample a .pt file from when --in_path is omitted")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, type=str,
                        help="Output directory for decoded audio")
    args = parser.parse_args()

    from src.data_utils.utils import decode_audio_latents

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.in_path:
        args.in_path = _sample_pt_from_dir(args.in_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dacvae.DACVAE.load("facebook/dacvae-watermarked").eval().to(device)

    payload = torch.load(args.in_path, map_location="cpu")
    if isinstance(payload, dict) and "posterior_params" in payload:
        posterior = payload["posterior_params"]
    else:
        posterior = payload

    if posterior.dim() == 2:
        posterior = posterior.unsqueeze(0)
    if posterior.dim() != 3:
        raise ValueError(f"Unexpected posterior shape: {tuple(posterior.shape)}")

    if isinstance(payload, dict) and payload.get("latent_length") is not None:
        latent_length = int(payload["latent_length"])
        posterior = posterior[..., :latent_length]

    posterior = posterior.to(device)
    mean, logvar = torch.chunk(posterior, 2, dim=1)
    std = torch.exp(0.5 * logvar)
    sample = mean + std * torch.randn_like(std)

    metadata = {"sample_rate": int(model.sample_rate)}
    mean_audio = decode_audio_latents(model, mean, metadata)
    sample_audio = decode_audio_latents(model, sample, metadata)

    base = os.path.splitext(os.path.basename(args.in_path))[0]
    mean_path = os.path.join(args.out_dir, f"{base}_mean.mp3")
    sample_path = os.path.join(args.out_dir, f"{base}_sample.mp3")

    _save_audio(mean_path, mean_audio, int(model.sample_rate))
    _save_audio(sample_path, sample_audio, int(model.sample_rate))

    manifest_path = _infer_manifest_path(args.in_path, args.in_dir)

    target_id = os.path.splitext(os.path.basename(args.in_path))[0]
    caption, status = _read_caption_from_manifest(manifest_path, target_id)
    if status == "found":
        print("caption:", caption)
    elif status == "caption_missing":
        print("caption: <missing in manifest>")
    elif status == "id_not_found":
        print("caption: <id not found in manifest>")
    elif status == "manifest_unreadable":
        print("caption: <manifest unreadable>")
    else:
        print("caption: <manifest not found>")

    print("saved", mean_path)
    print("saved", sample_path)


if __name__ == "__main__":
    main()
