import argparse
import json
from pathlib import Path

import torch
import lightning as L

from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader as DataLoader


class MultiSourceAudioDataset(Dataset):
    """
    Loads merged audio latents + text embeddings from JSONL manifests.

    Each manifest entry provides:
        - path: path to latents/{id}.pt

    Returns:
        - moments: [T, 2C] adjusted to target_seq_len
        - prompt_data: dict with clap/t5 embeddings and t5_mask
    """

    def __init__(
        self,
        manifest_paths: list[str],
        data_root: str | None,
        silence_latent_path: str,
        target_seq_len: int = 251,
        max_t5_tokens: int = 68,
    ):
        assert manifest_paths, "manifest_paths is empty"

        self.target_seq_len = target_seq_len
        self.max_t5_tokens = max_t5_tokens

        self.data_root = Path(data_root).expanduser().resolve() if data_root else None

        silence_path = Path(silence_latent_path).expanduser().resolve()
        assert silence_path.exists(), f"Missing silence latent: {silence_path}"
        self.silence_latent = self._load_silence_latent(silence_path)

        self.files = []
        for manifest_path in manifest_paths:
            self._load_manifest(Path(manifest_path))

        assert self.files, "No entries found in manifest_paths"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        latent_path = self.files[idx]

        data = torch.load(latent_path, map_location="cpu", weights_only=True)
        posterior_params = data["posterior_params"]
        moments = posterior_params.transpose(0, 1)
        latent_length = int(data.get("latent_length", moments.shape[0]))
        assert latent_length <= moments.shape[0], "latent_length exceeds moments length"
        moments = moments[:latent_length]
        moments = self._adjust_audio_length(moments)

        clap_emb = data["clap_embedding"]
        t5_hidden = data["t5_last_hidden"]
        t5_len = int(data["t5_len"])
        assert t5_len == t5_hidden.shape[0], "t5_len mismatch"
        t5_padded, t5_mask = self._prepare_t5(t5_hidden, t5_len)

        prompt_data = {
            "clap": clap_emb,
            "t5": t5_padded,
            "t5_mask": t5_mask,
        }

        return moments, prompt_data

    def _load_manifest(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                latent_path = entry["path"]
                resolved_path = self._resolve_path(latent_path)
                self.files.append(resolved_path)

    def _resolve_path(self, path_str: str) -> str:
        path = Path(path_str)
        if path.is_absolute():
            return path.as_posix()
        assert self.data_root is not None, f"Relative path without data_root: {path_str}"
        return (self.data_root / path).as_posix()

    def _load_silence_latent(self, silence_path: Path) -> torch.Tensor:
        data = torch.load(silence_path, map_location="cpu", weights_only=True)
        posterior_params = data["posterior_params"]
        moments = posterior_params.transpose(0, 1)
        latent_length = int(data.get("latent_length", moments.shape[0]))
        assert latent_length <= moments.shape[0], "silence latent_length exceeds moments length"
        moments = moments[:latent_length]
        assert moments.shape[0] > 0, f"Silence latent has zero length: {silence_path}"
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


def audio_collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, dict]:
    moments_list = [item[0] for item in batch]
    prompt_list = [item[1] for item in batch]

    moments = torch.stack(moments_list, dim=0)
    prompt_data = {
        "clap": torch.stack([p["clap"] for p in prompt_list], dim=0),
        "t5": torch.stack([p["t5"] for p in prompt_list], dim=0),
        "t5_mask": torch.stack([p["t5_mask"] for p in prompt_list], dim=0),
    }
    return moments, prompt_data


class CachedAudioDataModule(L.LightningDataModule):
    """
    Lightning DataModule for cached audio latents and text embeddings.
    """

    def __init__(
        self,
        manifest_paths: list[str],
        data_root: str | None,
        silence_latent_path: str,
        batch_size: int,
        target_seq_len: int = 251,
        max_t5_tokens: int = 68,
        num_workers: int = 4,
    ):
        super().__init__()
        self.manifest_paths = manifest_paths
        self.data_root = data_root
        self.silence_latent_path = silence_latent_path
        self.batch_size = batch_size
        self.target_seq_len = target_seq_len
        self.max_t5_tokens = max_t5_tokens
        self.num_workers = num_workers
        self.train_dataset = None

    def setup(self, stage=None):
        self.train_dataset = MultiSourceAudioDataset(
            manifest_paths=self.manifest_paths,
            data_root=self.data_root,
            silence_latent_path=self.silence_latent_path,
            target_seq_len=self.target_seq_len,
            max_t5_tokens=self.max_t5_tokens,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            collate_fn=audio_collate_fn,
            in_order=False, ## we need this beacuse some latents are very big, and take longer to load. So the gpu starves waiting for those batches.
        )


def _parse_manifest_paths(value: str) -> list[str]:
    if value is None:
        return []
    cleaned = value.strip()
    if cleaned == "" or cleaned.lower() in {"none", "null"}:
        return []
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def _run_smoke_tests(args: argparse.Namespace) -> None:
    manifest_paths = _parse_manifest_paths(args.manifest_paths)
    assert manifest_paths, "No manifest paths provided"

    dataset = MultiSourceAudioDataset(
        manifest_paths=manifest_paths,
        data_root=args.data_root,
        silence_latent_path=args.silence_latent_path,
        target_seq_len=args.seq_len,
        max_t5_tokens=args.max_t5_tokens,
    )

    assert len(dataset) > 0, "Dataset is empty"

    moments, prompt_data = dataset[0]
    assert moments.shape == (args.seq_len, args.latent_dim), f"Unexpected moments shape: {moments.shape}"
    assert prompt_data["clap"].shape == (args.clap_dim,), f"Unexpected clap shape: {prompt_data['clap'].shape}"
    assert prompt_data["t5"].shape == (
        args.max_t5_tokens,
        args.t5_dim,
    ), f"Unexpected t5 shape: {prompt_data['t5'].shape}"
    assert prompt_data["t5_mask"].shape == (
        args.max_t5_tokens,
    ), f"Unexpected t5_mask shape: {prompt_data['t5_mask'].shape}"
    assert (
        prompt_data["t5_mask"].dtype == torch.bool
    ), f"Unexpected t5_mask dtype: {prompt_data['t5_mask'].dtype}"

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=audio_collate_fn)
    batch = next(iter(loader))
    batch_moments, batch_prompt = batch
    assert batch_moments.shape == (
        args.batch_size,
        args.seq_len,
        args.latent_dim,
    ), f"Unexpected batch moments shape: {batch_moments.shape}"
    assert batch_prompt["clap"].shape == (
        args.batch_size,
        args.clap_dim,
    ), f"Unexpected batch clap shape: {batch_prompt['clap'].shape}"
    assert batch_prompt["t5"].shape == (
        args.batch_size,
        args.max_t5_tokens,
        args.t5_dim,
    ), f"Unexpected batch t5 shape: {batch_prompt['t5'].shape}"
    assert batch_prompt["t5_mask"].shape == (
        args.batch_size,
        args.max_t5_tokens,
    ), f"Unexpected batch t5_mask shape: {batch_prompt['t5_mask'].shape}"

    print("PASS: dataset item shapes OK")
    print("PASS: dataloader batch shapes OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Audio data pipeline smoke tests", add_help=True)
    parser.add_argument(
        "--manifest-paths",
        type=str,
        default="/share/users/student/f/friverossego/datasets/audio_manifest_train.jsonl",
        help="Comma-separated JSONL manifest paths",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/share/users/student/f/friverossego/datasets",
        help="Base path for relative manifest entries",
    )
    parser.add_argument(
        "--silence-latent-path",
        type=str,
        default="/share/users/student/f/friverossego/LatentLM/silence_samples/silence_10s_dacvae.pt",
        help="Path to silence latent",
    )
    parser.add_argument("--seq-len", type=int, default=251)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--max-t5-tokens", type=int, default=68)
    parser.add_argument("--clap-dim", type=int, default=512)
    parser.add_argument("--t5-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    _run_smoke_tests(args)
