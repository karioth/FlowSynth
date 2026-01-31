import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.metrics import (
    compute_activations_from_dataset,
    compute_inception_score_from_dataset,
    frechet_distance,
    get_inception_model,
    mean_covar_numpy,
)
from src.data_utils.preprocess.img_items import center_crop_arr, safe_blob_dump

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate metrics from generated images.")
    parser.add_argument("--images_path", type=str, required=True, help="Directory of images or .npz file.")
    parser.add_argument("--ref_stat_path", type=str, default=None, help="Path to reference stats .npz (mu/sigma).")
    parser.add_argument("--train_data_dir", type=str, default=None, help="ImageFolder root for torch_fidelity.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fid", action="store_true", help="Compute FID using ref_stat_path.")
    parser.add_argument("--is", dest="inception_score", action="store_true", help="Compute Inception Score.")
    parser.add_argument("--fidelity", action="store_true", help="Compute torch_fidelity metrics.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to write metrics as JSON.")
    return parser.parse_args()


def list_image_files(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def load_npz_array(path: str) -> np.ndarray:
    data = np.load(path)
    if "arr_0" in data:
        return data["arr_0"]
    if "images" in data:
        return data["images"]
    raise KeyError(f"Expected 'arr_0' or 'images' in {path}.")


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, paths: List[str], image_size: int, as_float: bool):
        self.paths = paths
        self.image_size = image_size
        self.as_float = as_float

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.image_size is not None:
            if img.size[0] != self.image_size or img.size[1] != self.image_size:
                img = center_crop_arr(img, self.image_size)
        arr = np.array(img)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        if self.as_float:
            tensor = tensor.float().div(255.0).clamp(0.0, 1.0)
        return (tensor,)


class FirstItemDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.dataset[idx]
        return item[0] if isinstance(item, (tuple, list)) else item


class RefImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: ImageFolder):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.dataset[idx]
        item = np.array(item[0])
        item = torch.from_numpy(item).permute(2, 0, 1)
        return item


def build_dataset(images_path: str, image_size: int, as_float: bool) -> torch.utils.data.Dataset:
    if images_path.endswith(".npz"):
        arr = load_npz_array(images_path)
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D array in {images_path}, got shape {arr.shape}.")
        if arr.shape[-1] == 3:
            arr = arr.transpose(0, 3, 1, 2)
        tensor = torch.from_numpy(arr)
        if as_float:
            tensor = tensor.float()
            if tensor.numel() > 0 and float(tensor.max()) > 1.0:
                tensor = tensor.div(255.0)
            tensor = tensor.clamp(0.0, 1.0)
        else:
            if tensor.dtype != torch.uint8:
                max_val = float(tensor.max()) if tensor.numel() > 0 else 0.0
                if max_val <= 1.0:
                    tensor = tensor.mul(255.0)
                tensor = tensor.round().clamp(0, 255).to(torch.uint8)
        return torch.utils.data.TensorDataset(tensor)

    if not os.path.isdir(images_path):
        raise ValueError(f"{images_path} is not a directory or .npz file.")
    paths = list_image_files(images_path)
    if not paths:
        raise ValueError(f"No images found in {images_path}.")
    return ImagePathDataset(paths, image_size=image_size, as_float=as_float)


def compute_fid_from_dataset(
    dataset: torch.utils.data.Dataset,
    ref_stat_path: str,
    batch_size: int,
    device: torch.device,
    inception_model: torch.nn.Module,
) -> float:
    if ref_stat_path is None:
        raise ValueError("--ref_stat_path is required to compute FID.")
    if not os.path.exists(ref_stat_path):
        raise FileNotFoundError(f"Reference stats not found: {ref_stat_path}")
    acts = compute_activations_from_dataset(
        dataset,
        batch_size=batch_size,
        inception_model=inception_model,
        device=device,
        normalized=False,
    )
    mu_fake, sigma_fake = mean_covar_numpy(acts)
    stats_ref = np.load(ref_stat_path)
    mu_ref, sigma_ref = stats_ref["mu"], stats_ref["sigma"]
    return frechet_distance(mu_ref, sigma_ref, mu_fake, sigma_fake)


def serialize_metrics(metrics: dict) -> dict:
    output = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.item()
        elif isinstance(value, (np.floating, np.integer)):
            output[key] = value.item()
        else:
            output[key] = value
    return output


@torch.no_grad()
def main():
    args = parse_args()
    if not (args.fid or args.inception_score or args.fidelity):
        args.fid = True
        args.inception_score = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    if args.fid or args.inception_score:
        dataset_float = build_dataset(args.images_path, args.image_size, as_float=True)
        inception_model = get_inception_model().to(device)
        if args.fid:
            fid = compute_fid_from_dataset(
                dataset_float,
                ref_stat_path=args.ref_stat_path,
                batch_size=args.batch_size,
                device=device,
                inception_model=inception_model,
            )
            results["fid"] = float(fid)
        if args.inception_score:
            is_mean, is_std = compute_inception_score_from_dataset(
                dataset_float,
                splits=10,
                batch_size=args.batch_size,
                device=device,
                inception_model=inception_model,
            )
            results["is_mean"] = float(is_mean)
            results["is_std"] = float(is_std)

    if args.fidelity:
        if args.train_data_dir is None:
            raise ValueError("--train_data_dir is required for --fidelity.")
        try:
            import torch_fidelity
        except ImportError as exc:
            raise RuntimeError("torch_fidelity is required for --fidelity.") from exc
        dataset_uint8 = FirstItemDataset(build_dataset(args.images_path, args.image_size, as_float=False))
        ref_dataset = ImageFolder(
            args.train_data_dir,
            transform=transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        )
        ref_dataset = RefImageDataset(ref_dataset)
        fidelity = torch_fidelity.calculate_metrics(
            input1=dataset_uint8,
            input2=ref_dataset,
            batch_size=args.batch_size,
            cuda=(device.type == "cuda"),
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            save_cpu_ram=True,
            verbose=True,
        )
        results["fidelity"] = serialize_metrics(fidelity)

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output_json is not None:
        safe_blob_dump(args.output_json, results)


if __name__ == "__main__":
    main()
