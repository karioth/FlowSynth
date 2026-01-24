import argparse
import json
import os
import sys

import torch
import torchaudio
import dacvae


DEFAULT_IN_PATH = "/share/users/student/f/friverossego/audioset_FBdacvae/4875DncHORw_66899.pt"
DEFAULT_OUT_DIR = "/share/users/student/f/friverossego/LatentLM/smoketest"


def _save_audio(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    wav = wav.detach().cpu()
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    if wav.dim() != 3 or wav.size(0) != 1:
        raise ValueError(f"Expected shape (1, C, T), got {tuple(wav.shape)}")
    torchaudio.save(path, wav[0], sample_rate=sample_rate, format="mp3")


def _read_caption(sidecar_path: str) -> str | None:
    if not os.path.exists(sidecar_path):
        return None
    try:
        with open(sidecar_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        return data.get("caption")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode cached DACVAE latents to audio.")
    parser.add_argument("--in_path", default=DEFAULT_IN_PATH, type=str,
                        help="Path to cached .pt file")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, type=str,
                        help="Output directory for decoded audio")
    parser.add_argument("--sidecar_path", default=None, type=str,
                        help="Optional path to caption .json sidecar")
    args = parser.parse_args()

    sys.path.insert(0, "/share/users/student/f/friverossego/LatentLM/src")
    from audio_utils import decode_audio_latents

    os.makedirs(args.out_dir, exist_ok=True)

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

    sidecar_path = args.sidecar_path
    if sidecar_path is None:
        sidecar_path = os.path.splitext(args.in_path)[0] + ".json"
    caption = _read_caption(sidecar_path)
    if caption:
        print("caption:", caption)
    elif os.path.exists(sidecar_path):
        print("caption: <missing or unreadable>")
    else:
        print("caption: <sidecar not found>")

    print("saved", mean_path)
    print("saved", sample_path)


if __name__ == "__main__":
    main()
