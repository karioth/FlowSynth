import argparse
import bisect
import os
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback

from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader as DataLoader

from src.data_utils.datamodule import audio_collate_fn


REQUIRED_CACHE_KEYS = {
    "version",
    "num_items",
    "keys",
    "posterior_cat",
    "posterior_offsets",
    "latent_lengths",
    "clap",
    "t5_cat",
    "t5_offsets",
    "t5_lens",
}

CONSOLIDATED_FILENAME = "consolidated_latents_bf16.pt"


def parse_consolidated_paths(path: str) -> list[str]:
    if path is None or path.strip() == "":
        raise ValueError("No consolidated path provided")

    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Missing consolidated path: {path_obj}")

    if path_obj.is_file():
        return [path_obj.as_posix()]

    if not path_obj.is_dir():
        raise ValueError(f"Unsupported consolidated path type: {path_obj}")

    discovered: list[str] = []
    seen = set()
    matches = sorted(path_obj.rglob(CONSOLIDATED_FILENAME))
    if not matches:
        raise ValueError(f"No '{CONSOLIDATED_FILENAME}' files found under directory: {path_obj}")

    for match in matches:
        resolved = match.resolve().as_posix()
        if resolved not in seen:
            discovered.append(resolved)
            seen.add(resolved)
    return discovered


class ConsolidatedAudioDataset(Dataset):
    """
    Dataset backed by one or more consolidated ragged cache .pt files.

    Each cache is expected to follow the schema produced by consolidate_cache.py.
    """

    def __init__(
        self,
        consolidated_paths: list[str],
        silence_latent_path: str,
        target_seq_len: int = 251,
        max_t5_tokens: int = 68,
        mmap: bool = True,
    ):
        assert consolidated_paths, "consolidated_paths is empty"

        self.target_seq_len = target_seq_len
        self.max_t5_tokens = max_t5_tokens
        self.mmap = bool(mmap)

        silence_path = Path(silence_latent_path).expanduser().resolve()
        assert silence_path.exists(), f"Missing silence latent: {silence_path}"
        self.silence_latent = self._load_silence_latent(silence_path)

        self.caches = []
        self.cache_paths = []
        self.prefix = [0]

        for raw_path in consolidated_paths:
            cache_path = Path(raw_path).expanduser().resolve()
            if not cache_path.exists():
                raise FileNotFoundError(f"Missing consolidated cache: {cache_path}")
            cache = self._load_cache(cache_path)
            num_items = int(cache["num_items"])
            self.caches.append(cache)
            self.cache_paths.append(cache_path.as_posix())
            self.prefix.append(self.prefix[-1] + num_items)

        self.total_items = self.prefix[-1]
        assert self.total_items > 0, "No entries found in consolidated caches"

    def __len__(self):
        return self.total_items

    def __getitem__(self, idx):
        cache_idx, local_idx = self._locate_global_index(idx)
        cache = self.caches[cache_idx]

        p_offsets = cache["posterior_offsets"]
        p_start = int(p_offsets[local_idx].item())
        p_end = int(p_offsets[local_idx + 1].item())
        posterior_params = cache["posterior_cat"][p_start:p_end]
        latent_length = int(cache["latent_lengths"][local_idx].item())
        assert latent_length <= posterior_params.shape[0], "latent_length exceeds posterior span"
        posterior_params = posterior_params[:latent_length]
        posterior_params = self._adjust_audio_length(posterior_params)

        clap_emb = cache["clap"][local_idx]

        t_offsets = cache["t5_offsets"]
        t_start = int(t_offsets[local_idx].item())
        t_end = int(t_offsets[local_idx + 1].item())
        t5_hidden = cache["t5_cat"][t_start:t_end]
        t5_len = int(cache["t5_lens"][local_idx].item())
        assert t5_len <= t5_hidden.shape[0], "t5_len exceeds t5 span"
        t5_hidden = t5_hidden[:t5_len]
        t5_padded, t5_mask = self._prepare_t5(t5_hidden, t5_len)

        prompt_data = {
            "clap": clap_emb,
            "t5": t5_padded,
            "t5_mask": t5_mask,
        }
        return posterior_params, prompt_data

    def _locate_global_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self.total_items
        if idx < 0 or idx >= self.total_items:
            raise IndexError(f"Index {idx} out of range for dataset size {self.total_items}")

        cache_idx = bisect.bisect_right(self.prefix, idx) - 1
        local_idx = idx - self.prefix[cache_idx]
        return cache_idx, local_idx

    def _load_cache(self, cache_path: Path) -> dict:
        try:
            cache = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
                mmap=self.mmap,
            )
        except TypeError:
            if self.mmap:
                raise RuntimeError(
                    "torch.load(..., mmap=...) is not supported by this torch build."
                )
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)

        missing = REQUIRED_CACHE_KEYS - set(cache.keys())
        if missing:
            raise KeyError(f"{cache_path} missing required keys: {sorted(missing)}")

        self._validate_cache(cache, cache_path)
        return cache

    def _validate_cache(self, cache: dict, cache_path: Path) -> None:
        num_items = int(cache["num_items"])
        if num_items < 0:
            raise ValueError(f"{cache_path}: num_items must be >= 0, got {num_items}")

        keys = cache["keys"]
        if not isinstance(keys, list):
            raise TypeError(f"{cache_path}: keys must be a list")
        if len(keys) != num_items:
            raise ValueError(
                f"{cache_path}: len(keys)={len(keys)} does not match num_items={num_items}"
            )

        posterior_cat = cache["posterior_cat"]
        posterior_offsets = cache["posterior_offsets"]
        latent_lengths = cache["latent_lengths"]
        clap = cache["clap"]
        t5_cat = cache["t5_cat"]
        t5_offsets = cache["t5_offsets"]
        t5_lens = cache["t5_lens"]

        if posterior_cat.ndim != 2 or posterior_cat.shape[1] != 256:
            raise ValueError(
                f"{cache_path}: posterior_cat must be [*, 256], got {tuple(posterior_cat.shape)}"
            )
        if clap.ndim != 2 or clap.shape != (num_items, 512):
            raise ValueError(
                f"{cache_path}: clap must be [{num_items}, 512], got {tuple(clap.shape)}"
            )
        if t5_cat.ndim != 2 or t5_cat.shape[1] != 1024:
            raise ValueError(
                f"{cache_path}: t5_cat must be [*, 1024], got {tuple(t5_cat.shape)}"
            )

        if posterior_offsets.ndim != 1 or posterior_offsets.shape[0] != num_items + 1:
            raise ValueError(
                f"{cache_path}: posterior_offsets must have len num_items+1 ({num_items + 1}), "
                f"got {tuple(posterior_offsets.shape)}"
            )
        if latent_lengths.ndim != 1 or latent_lengths.shape[0] != num_items:
            raise ValueError(
                f"{cache_path}: latent_lengths must have len num_items ({num_items}), "
                f"got {tuple(latent_lengths.shape)}"
            )

        if t5_offsets.ndim != 1 or t5_offsets.shape[0] != num_items + 1:
            raise ValueError(
                f"{cache_path}: t5_offsets must have len num_items+1 ({num_items + 1}), "
                f"got {tuple(t5_offsets.shape)}"
            )
        if t5_lens.ndim != 1 or t5_lens.shape[0] != num_items:
            raise ValueError(
                f"{cache_path}: t5_lens must have len num_items ({num_items}), "
                f"got {tuple(t5_lens.shape)}"
            )

        p_offsets = posterior_offsets.to(torch.int64)
        p_lens = latent_lengths.to(torch.int64)
        if int(p_offsets[0].item()) != 0:
            raise ValueError(f"{cache_path}: posterior_offsets[0] must be 0")
        if int(p_offsets[-1].item()) != posterior_cat.shape[0]:
            raise ValueError(
                f"{cache_path}: posterior_offsets[-1]={int(p_offsets[-1].item())} "
                f"does not match posterior_cat rows={posterior_cat.shape[0]}"
            )
        if torch.any(p_lens < 0):
            raise ValueError(f"{cache_path}: latent_lengths must be non-negative")
        p_diff = p_offsets[1:] - p_offsets[:-1]
        if not torch.equal(p_diff, p_lens):
            raise ValueError(f"{cache_path}: posterior offset diffs do not match latent_lengths")

        t_offsets = t5_offsets.to(torch.int64)
        t_lens = t5_lens.to(torch.int64)
        if int(t_offsets[0].item()) != 0:
            raise ValueError(f"{cache_path}: t5_offsets[0] must be 0")
        if int(t_offsets[-1].item()) != t5_cat.shape[0]:
            raise ValueError(
                f"{cache_path}: t5_offsets[-1]={int(t_offsets[-1].item())} "
                f"does not match t5_cat rows={t5_cat.shape[0]}"
            )
        if torch.any(t_lens < 0):
            raise ValueError(f"{cache_path}: t5_lens must be non-negative")
        t_diff = t_offsets[1:] - t_offsets[:-1]
        if not torch.equal(t_diff, t_lens):
            raise ValueError(f"{cache_path}: t5 offset diffs do not match t5_lens")

    def _load_silence_latent(self, silence_path: Path) -> torch.Tensor:
        data = torch.load(silence_path, map_location="cpu", weights_only=True)
        posterior_params = data["posterior_params"].transpose(0, 1)
        latent_length = int(data.get("latent_length", posterior_params.shape[0]))
        assert latent_length <= posterior_params.shape[0], "silence latent_length exceeds posterior_params length"
        posterior_params = posterior_params[:latent_length]
        assert posterior_params.shape[0] > 0, f"Silence latent has zero length: {silence_path}"
        # Consolidated caches are bf16; keep silence padding tensor aligned.
        return posterior_params.to(torch.bfloat16)

    def _adjust_audio_length(self, posterior_params: torch.Tensor) -> torch.Tensor:
        current_len = posterior_params.shape[0]

        if current_len == self.target_seq_len:
            return posterior_params

        if current_len < self.target_seq_len:
            pad_needed = self.target_seq_len - current_len
            silence_latent = self.silence_latent
            if silence_latent.dtype != posterior_params.dtype:
                silence_latent = silence_latent.to(dtype=posterior_params.dtype)
            silence_len = silence_latent.shape[0]
            num_tiles = (pad_needed // silence_len) + 1
            padding = silence_latent.repeat(num_tiles, 1)[:pad_needed]
            return torch.cat([posterior_params, padding], dim=0)

        max_start = current_len - self.target_seq_len
        start = torch.randint(0, max_start + 1, (1,)).item()
        return posterior_params[start : start + self.target_seq_len]

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

@torch.no_grad()
def warm_pages(t: torch.Tensor):
    try:
        page = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError):
        page = 4096
    step = max(1, int(page) // max(1, t.element_size()))
    _ = t.reshape(-1)[::step].sum(dtype=torch.float64)


class ConsolidatedAudioDataModule(L.LightningDataModule):
    """
    Lightning DataModule for consolidated cached audio tensors.
    """

    def __init__(
        self,
        consolidated_paths: list[str],
        silence_latent_path: str,
        batch_size: int,
        target_seq_len: int = 251,
        max_t5_tokens: int = 68,
        num_workers: int = 4,
        pin_memory: bool = True,
        mmap: bool = True,
    ):
        super().__init__()
        self.consolidated_paths = consolidated_paths
        self.silence_latent_path = silence_latent_path
        self.batch_size = batch_size
        self.target_seq_len = target_seq_len
        self.max_t5_tokens = max_t5_tokens
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mmap = mmap
        self.train_dataset = None

    def setup(self, stage=None):
        self.train_dataset = ConsolidatedAudioDataset(
            consolidated_paths=self.consolidated_paths,
            silence_latent_path=self.silence_latent_path,
            target_seq_len=self.target_seq_len,
            max_t5_tokens=self.max_t5_tokens,
            mmap=self.mmap,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            collate_fn=audio_collate_fn,
        )


class ConsolidatedCacheWarmupCallback(Callback):
    """
    Node-local page cache warmup for consolidated datasets.
    Runs once per node on LOCAL_RANK==0, then syncs all ranks.
    """

    def on_fit_start(self, trainer, pl_module):
        dm = trainer.datamodule
        if not isinstance(dm, ConsolidatedAudioDataModule):
            return

        local_rank = int(getattr(trainer, "local_rank", 0))
        should_warm = local_rank == 0

        if should_warm:
            dataset = dm.train_dataset
            if dataset is None:
                raise RuntimeError("Consolidated datamodule train_dataset is not initialized")

            total_caches = len(dataset.caches)
            print(
                f"[Cache Warmup] Starting page-cache warmup for {total_caches} subsets.",
                flush=True,
            )

            for cache_idx, cache in enumerate(dataset.caches, start=1):
                cache_path = Path(dataset.cache_paths[cache_idx - 1])
                subset_name = f"{cache_path.parent.parent.name}/{cache_path.parent.name}"
                warm_pages(cache["posterior_cat"])
                warm_pages(cache["t5_cat"])
                warm_pages(cache["clap"])
                print(
                    f"[Cache Warmup] [{cache_idx}/{total_caches}] Done {subset_name}.",
                    flush=True,
                )

        trainer.strategy.barrier()


def _run_smoke_tests(args: argparse.Namespace) -> None:
    consolidated_paths = parse_consolidated_paths(args.consolidated_paths)

    dataset = ConsolidatedAudioDataset(
        consolidated_paths=consolidated_paths,
        silence_latent_path=args.silence_latent_path,
        target_seq_len=args.seq_len,
        max_t5_tokens=args.max_t5_tokens,
        mmap=not args.no_mmap,
    )

    expected_len = sum(int(cache["num_items"]) for cache in dataset.caches)
    assert len(dataset) == expected_len, f"Expected len {expected_len}, got {len(dataset)}"

    posterior_params, prompt_data = dataset[0]
    assert posterior_params.shape == (
        args.seq_len,
        args.latent_dim,
    ), f"Unexpected posterior_params shape: {posterior_params.shape}"
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
    batch_posterior_params, batch_prompt = batch
    assert batch_posterior_params.shape == (
        args.batch_size,
        args.seq_len,
        args.latent_dim,
    ), f"Unexpected batch posterior_params shape: {batch_posterior_params.shape}"
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

    print("PASS: consolidated dataset item shapes OK")
    print("PASS: consolidated dataloader batch shapes OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Consolidated audio data pipeline smoke tests", add_help=True)
    parser.add_argument(
        "--consolidated-paths",
        type=str,
        required=True,
        help="Comma-separated consolidated cache paths",
    )
    parser.add_argument(
        "--silence-latent-path",
        type=str,
        default="silence_samples/silence_10s_dacvae.pt",
        help="Path to silence latent",
    )
    parser.add_argument("--seq-len", type=int, default=251)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--max-t5-tokens", type=int, default=68)
    parser.add_argument("--clap-dim", type=int, default=512)
    parser.add_argument("--t5-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-mmap", action="store_true")
    args = parser.parse_args()
    _run_smoke_tests(args)
