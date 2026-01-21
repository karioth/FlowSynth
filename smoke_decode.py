import os
import sys

import torch
import torchaudio
import dacvae


IN_PATH = "/share/users/student/f/friverossego/jamendo_cached/131470.pt"
OUT_DIR = "/share/users/student/f/friverossego/LatentLM/smoketest"


def _save_audio(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    wav = wav.detach().cpu()
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    if wav.dim() != 3 or wav.size(0) != 1:
        raise ValueError(f"Expected shape (1, C, T), got {tuple(wav.shape)}")
    torchaudio.save(path, wav[0], sample_rate=sample_rate, format="mp3")


def main() -> None:
    sys.path.insert(0, "/share/users/student/f/friverossego/LatentLM/src")
    from audio_utils import decode_audio_latents

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dacvae.DACVAE.load("facebook/dacvae-watermarked").eval().to(device)

    payload = torch.load(IN_PATH, map_location="cpu")
    if isinstance(payload, dict) and "posterior_params" in payload:
        posterior = payload["posterior_params"]
    else:
        posterior = payload

    if posterior.dim() == 2:
        posterior = posterior.unsqueeze(0)
    if posterior.dim() != 3:
        raise ValueError(f"Unexpected posterior shape: {tuple(posterior.shape)}")

    posterior = posterior.to(device)
    mean, logvar = torch.chunk(posterior, 2, dim=1)
    std = torch.exp(0.5 * logvar)
    sample = mean + std * torch.randn_like(std)

    metadata = {"sample_rate": int(model.sample_rate)}
    mean_audio = decode_audio_latents(model, mean, metadata)
    sample_audio = decode_audio_latents(model, sample, metadata)

    base = os.path.splitext(os.path.basename(IN_PATH))[0]
    mean_path = os.path.join(OUT_DIR, f"{base}_mean.mp3")
    sample_path = os.path.join(OUT_DIR, f"{base}_sample.mp3")

    _save_audio(mean_path, mean_audio, int(model.sample_rate))
    _save_audio(sample_path, sample_audio, int(model.sample_rate))

    print("saved", mean_path)
    print("saved", sample_path)


if __name__ == "__main__":
    main()
