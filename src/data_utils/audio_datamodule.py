import glob
import os

import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader


class CachedAudioDataset(Dataset):
    """
    Loads pre-computed AudioSet latents and CLAP embeddings.

    Each .pt file contains:
        - posterior_params: [2C, T] DACVAE latent (mean + logvar concatenated)
        - latent_length: int (should be 251)
        - text_embedding: [D] CLAP vector

    Returns:
        - moments: [T, 2C] (transposed to sequence-first format)
        - clap_embedding: [D] CLAP vector
    """

    def __init__(self, root: str, seq_len: int = 251):
        self.files = sorted(glob.glob(os.path.join(root, "*.pt")))
        self.seq_len = seq_len
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)

        # posterior_params: [2C, T] -> [T, 2C]
        moments = data["posterior_params"].transpose(0, 1)

        # Validate sequence length
        if moments.shape[0] != self.seq_len:
            raise ValueError(
                f"Expected seq_len {self.seq_len}, got {moments.shape[0]} "
                f"for file {self.files[idx]}"
            )

        clap_embedding = data["text_embedding"]
        return moments, clap_embedding


class CachedAudioDataModule(L.LightningDataModule):
    """
    Lightning DataModule for cached audio latents and CLAP embeddings.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        seq_len: int = 251,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.train_dataset = None

    def setup(self, stage=None):
        self.train_dataset = CachedAudioDataset(self.data_path, seq_len=self.seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
