import argparse
import json
import math
import os
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader, Dataset

from src.lightning import LitModule, EMAWeightAveraging
from src.data_utils.audio_datamodule import audio_collate_fn


class OverfitAudioDataset(Dataset):
    def __init__(
        self,
        file_pairs: list[tuple[str, str]],
        silence_latent_path: str,
        target_seq_len: int,
        max_t5_tokens: int,
    ) -> None:
        self.files = file_pairs
        self.target_seq_len = target_seq_len
        self.max_t5_tokens = max_t5_tokens

        silence_path = Path(silence_latent_path).expanduser().resolve()
        if not silence_path.exists():
            raise FileNotFoundError(f"Missing silence latent: {silence_path}")
        self.silence_latent = self._load_silence_latent(silence_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path, text_path = self.files[idx]

        audio_data = torch.load(audio_path, map_location="cpu", weights_only=True)
        posterior_params = audio_data["posterior_params"]
        moments = posterior_params.transpose(0, 1)
        latent_length = int(audio_data.get("latent_length", moments.shape[0]))
        if latent_length > moments.shape[0]:
            raise ValueError("latent_length exceeds moments length")
        moments = moments[:latent_length]
        moments = self._adjust_audio_length(moments)

        text_data = torch.load(text_path, map_location="cpu", weights_only=True)
        clap_emb = text_data["clap_embedding"]
        t5_hidden = text_data["t5_last_hidden"]
        t5_len = int(text_data["t5_len"])
        if t5_len != t5_hidden.shape[0]:
            raise ValueError("t5_len mismatch")
        t5_padded, t5_mask = self._prepare_t5(t5_hidden, t5_len)

        prompt_data = {
            "clap": clap_emb,
            "t5": t5_padded,
            "t5_mask": t5_mask,
        }
        return moments, prompt_data

    def _load_silence_latent(self, silence_path: Path) -> torch.Tensor:
        data = torch.load(silence_path, map_location="cpu", weights_only=True)
        posterior_params = data["posterior_params"]
        moments = posterior_params.transpose(0, 1)
        latent_length = int(data.get("latent_length", moments.shape[0]))
        if latent_length > moments.shape[0]:
            raise ValueError("silence latent_length exceeds moments length")
        moments = moments[:latent_length]
        if moments.shape[0] == 0:
            raise ValueError(f"Silence latent has zero length: {silence_path}")
        return moments

    def _adjust_audio_length(self, moments: torch.Tensor) -> torch.Tensor:
        current_len = moments.shape[0]
        if current_len == self.target_seq_len:
            return moments
        if current_len < self.target_seq_len:
            pad_needed = self.target_seq_len - current_len
            silence_len = self.silence_latent.shape[0]
            num_tiles = (pad_needed // silence_len) + 1
            padding = self.silence_latent.repeat(num_tiles, 1)[:pad_needed]
            return torch.cat([moments, padding], dim=0)
        max_start = current_len - self.target_seq_len
        start = torch.randint(0, max_start + 1, (1,)).item()
        return moments[start : start + self.target_seq_len]

    def _prepare_t5(self, t5_hidden: torch.Tensor, t5_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if t5_len <= 0:
            t5_padded = t5_hidden.new_zeros(self.max_t5_tokens, t5_hidden.shape[1])
            t5_mask = torch.zeros(self.max_t5_tokens, dtype=torch.bool)
            return t5_padded, t5_mask
        if t5_len >= self.max_t5_tokens:
            t5_padded = t5_hidden[: self.max_t5_tokens].clone()
            t5_mask = torch.ones(self.max_t5_tokens, dtype=torch.bool)
            return t5_padded, t5_mask
        pad_size = self.max_t5_tokens - t5_len
        padding = t5_hidden.new_zeros(pad_size, t5_hidden.shape[1])
        t5_padded = torch.cat([t5_hidden[:t5_len], padding], dim=0)
        t5_mask = torch.zeros(self.max_t5_tokens, dtype=torch.bool)
        t5_mask[:t5_len] = True
        return t5_padded, t5_mask


def _iter_manifest_entries(manifest_paths: list[str]):
    for manifest_path in manifest_paths:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                yield json.loads(stripped)


def _resolve_path(path_str: str, data_root: str | None) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return path.as_posix()
    if data_root is None:
        raise ValueError(f"Relative path without data_root: {path_str}")
    return (Path(data_root) / path).as_posix()


def _select_pairs(
    manifest_paths: list[str],
    data_root: str | None,
    seq_len: int,
    num_samples: int,
    require_exact_length: bool,
    max_scan: int,
) -> list[tuple[str, str]]:
    selected = []
    scanned = 0
    for entry in _iter_manifest_entries(manifest_paths):
        if max_scan > 0 and scanned >= max_scan:
            break
        scanned += 1
        audio_path = _resolve_path(entry["audio_path"], data_root)
        text_path = _resolve_path(entry["text_path"], data_root)
        if not os.path.exists(audio_path) or not os.path.exists(text_path):
            continue
        if require_exact_length:
            audio_data = torch.load(audio_path, map_location="cpu", weights_only=True)
            posterior_params = audio_data["posterior_params"]
            moments = posterior_params.transpose(0, 1)
            latent_length = int(audio_data.get("latent_length", moments.shape[0]))
            if latent_length != seq_len:
                continue
        selected.append((audio_path, text_path))
        if len(selected) >= num_samples:
            break
    if len(selected) < num_samples:
        raise RuntimeError(f"Found only {len(selected)} samples (need {num_samples}).")
    return selected


def parse_args():
    parser = argparse.ArgumentParser("Overfit a small audio subset for sanity checks")
    parser.add_argument(
        "--manifest-paths",
        action="append",
        default=["/share/users/student/f/friverossego/datasets/audio_manifest_train.jsonl"],
        help="JSONL manifest paths (repeatable or comma-separated)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/share/users/student/f/friverossego/datasets",
        help="Base directory for relative manifest entries",
    )
    parser.add_argument(
        "--silence-latent-path",
        type=str,
        default="silence_samples/silence_10s_dacvae.pt",
        help="Path to silence latent for padding",
    )
    parser.add_argument("--model", type=str, default="DiT-Base")
    parser.add_argument("--seq-len", type=int, default=251)
    parser.add_argument("--latent-size", type=int, default=128)
    parser.add_argument("--conditioning-type", type=str, default="continuous")
    parser.add_argument("--clap-dim", type=int, default=512)
    parser.add_argument("--t5-dim", type=int, default=1024)
    parser.add_argument("--prompt-seq-len", type=int, default=69)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--allow-random-crop", action="store_true")
    parser.add_argument("--max-scan", type=int, default=20000)
    parser.add_argument("--results-dir", type=str, default="overfit_sanity")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["bf16-mixed", "16-mixed", "32"],
    )
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="ddp")
    return parser.parse_args()


def _parse_manifest_paths(values: list[str] | None) -> list[str]:
    if not values:
        return []
    manifest_paths = []
    for value in values:
        if value is None:
            continue
        for item in value.split(","):
            item = item.strip()
            if item:
                manifest_paths.append(item)
    return manifest_paths


def main():
    args = parse_args()
    seed_everything(args.global_seed, workers=True)

    manifest_paths = _parse_manifest_paths(args.manifest_paths)
    if not manifest_paths:
        raise ValueError("No manifest paths provided")

    for path in manifest_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing manifest: {path}")

    require_exact_length = not args.allow_random_crop
    selected = _select_pairs(
        manifest_paths=manifest_paths,
        data_root=args.data_root,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        require_exact_length=require_exact_length,
        max_scan=args.max_scan,
    )

    silence_path = args.silence_latent_path
    if not os.path.isabs(silence_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        silence_path = os.path.join(script_dir, silence_path)

    dataset = OverfitAudioDataset(
        file_pairs=selected,
        silence_latent_path=silence_path,
        target_seq_len=args.seq_len,
        max_t5_tokens=args.prompt_seq_len - 1,
    )

    batch_size = args.batch_size or args.num_samples
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=audio_collate_fn,
    )

    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("No batches available in the dataloader")

    max_epochs = args.epochs
    if args.max_steps is not None:
        max_epochs = max(max_epochs, math.ceil(args.max_steps / steps_per_epoch))

    lit = LitModule(
        model_name=args.model,
        seq_len=args.seq_len,
        latent_size=args.latent_size,
        num_classes=1,
        conditioning_type=args.conditioning_type,
        clap_dim=args.clap_dim,
        t5_dim=args.t5_dim,
        prompt_seq_len=args.prompt_seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
    )

    ckpt_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{step:07d}",
        every_n_train_steps=200,
        save_last=True,
        save_top_k=-1,
    )
    pbar = TQDMProgressBar(refresh_rate=10)

    trainer = L.Trainer(
        default_root_dir=args.results_dir,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        max_epochs=max_epochs,
        max_steps=args.max_steps,
        precision=args.precision,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[ckpt_cb, EMAWeightAveraging(), pbar],
        log_every_n_steps=10,
    )

    trainer.fit(lit, train_dataloaders=loader)

    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        print(f"Last checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()
