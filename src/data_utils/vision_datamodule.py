import numpy as np

import torch
import torchvision.datasets as datasets
import lightning as L

from torch.utils.data import DataLoader


class CachedImageFolder(datasets.DatasetFolder):
    def __init__(self, root: str, seq_len: int | None = None):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )
        self.seq_len = seq_len

    def _to_sequence(self, moments: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(moments)
        if tensor.dim() == 3:
            channels, height, width = tensor.shape
            seq_len = int(height * width)
            if self.seq_len is not None and seq_len != self.seq_len:
                raise ValueError(f"Expected seq_len {self.seq_len}, got {seq_len}")
            tensor = tensor.permute(1, 2, 0).reshape(seq_len, channels)
        elif tensor.dim() == 2:
            if self.seq_len is not None and tensor.shape[0] != self.seq_len:
                raise ValueError(f"Expected seq_len {self.seq_len}, got {tensor.shape[0]}")
        else:
            raise ValueError(f"Unsupported moments shape: {tuple(tensor.shape)}")
        return tensor

    def __getitem__(self, index: int):
        """
        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        with np.load(path) as data:
            if torch.rand(1) < 0.5:
                moments = data["moments"]
            else:
                moments = data["moments_flip"]

        moments = self._to_sequence(moments)
        return moments, target


class CachedLatentsDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, seq_len: int, num_workers: int = 4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.train_dataset = None

    def setup(self, stage=None):
        self.train_dataset = CachedImageFolder(self.data_path, seq_len=self.seq_len)

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
