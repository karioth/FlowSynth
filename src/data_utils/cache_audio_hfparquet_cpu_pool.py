import argparse
import hashlib
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.data_utils.audio_utils import scan_cached_outputs
from src.data_utils.preencode_audio_latents import cache_audio_latents

_MODEL = None
_DS = None
_WORKER_STATE = {}


def _resolve_node_setting(value, env_keys, default=None):
    if value is not None:
        return value
    for key in env_keys:
        env_val = os.environ.get(key)
        if env_val is None:
            continue
        try:
            return int(env_val)
        except ValueError:
            continue
    return default


def _belongs_to_node(rel_path: str, node_rank: int, node_world_size: int) -> bool:
    if node_world_size <= 1:
        return True
    digest = hashlib.md5(rel_path.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % int(node_world_size)
    return bucket == int(node_rank)


def _non_silence_ratio(wav: torch.Tensor) -> float:
    non_silent = wav.abs() > 1e-4
    if non_silent.ndim == 2:
        non_silent = non_silent.any(dim=0)
    return non_silent.float().mean().item()


def _resolve_uid(idx: int, name_value: object | None) -> tuple[str, str]:
    if name_value is None:
        source_id = f"{int(idx):09d}"
    else:
        source_id = str(name_value)
    if source_id.endswith(".pt"):
        out_name = source_id
    else:
        out_name = f"{source_id}.pt"
    return source_id, out_name


def _init_worker(
    weights_path: str,
    device: str,
    hf_name: str,
    split: str,
    data_dir: str,
    cache_dir: str | None,
    out_root: str,
    name_column: str | None,
    min_duration_seconds: float,
    max_duration_seconds: float,
    chunk_size_latents: int,
    overlap_latents: int,
    worker_threads: int,
    node_rank: int,
):
    global _MODEL, _DS, _WORKER_STATE

    torch.set_num_threads(int(worker_threads))
    torch.set_num_interop_threads(1)

    import dacvae

    _MODEL = dacvae.DACVAE.load(weights_path).eval().to(device)
    _DS = load_dataset(hf_name, data_dir=data_dir, split=split, cache_dir=cache_dir)
    out_root_path = Path(out_root)
    log_dir = out_root_path.parent
    _WORKER_STATE = {
        "out_root": out_root_path,
        "name_column": name_column,
        "min_duration_seconds": float(min_duration_seconds),
        "max_duration_seconds": float(max_duration_seconds),
        "chunk_size_latents": int(chunk_size_latents),
        "overlap_latents": int(overlap_latents),
        "skip_log_path": str(log_dir / f"skipped_files.node{int(node_rank)}.log"),
        "event_log_path": str(log_dir / f"processing_events.node{int(node_rank)}.log"),
    }


def _process_index(idx: int):
    out_root = _WORKER_STATE["out_root"]
    skip_log_path = _WORKER_STATE["skip_log_path"]
    event_log_path = _WORKER_STATE["event_log_path"]
    source_id = f"{int(idx):09d}"

    try:
        ex = _DS[int(idx)]
        name_column = _WORKER_STATE["name_column"]
        name_value = ex.get(name_column) if name_column else None
        source_id, out_name = _resolve_uid(idx, name_value)
        out_path = out_root / out_name

        if out_path.exists():
            return

        a = ex["audio"]

        arr = a["array"]
        sr = int(a["sampling_rate"])

        x = torch.from_numpy(arr).to(torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        else:
            x = x.transpose(0, 1)

        x = x.clamp_(-1, 1)

        num_frames = x.shape[-1]
        min_frames = int(_WORKER_STATE["min_duration_seconds"] * float(sr))
        if num_frames < min_frames:
            try:
                with open(event_log_path, "a", encoding="utf-8") as f:
                    f.write(f"too short, skipping: {source_id}\n")
            except Exception:
                pass
            msg = f"{source_id}\ttoo_short\n"
            try:
                with open(skip_log_path, "a", encoding="utf-8") as f:
                    f.write(msg)
            except Exception:
                pass
            return

        max_frames = int(_WORKER_STATE["max_duration_seconds"] * float(sr))
        if max_frames > 0 and num_frames > max_frames:
            try:
                with open(event_log_path, "a", encoding="utf-8") as f:
                    f.write(f"file too long, taking random 10min crop: {source_id}\n")
            except Exception:
                pass
            max_start = num_frames - max_frames
            found = False
            for _ in range(10):
                start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
                seg = x[:, start : start + max_frames]
                if _non_silence_ratio(seg) >= 0.7:
                    x = seg
                    found = True
                    break
            if not found:
                x = seg

        item = (source_id, x, sr, str(out_path))
        cache_audio_latents(
            [[item]],
            _MODEL,
            chunk_size_latents=_WORKER_STATE["chunk_size_latents"],
            overlap_latents=_WORKER_STATE["overlap_latents"],
            rank=os.getpid(),
        )
    except Exception as exc:
        msg = f"{source_id}\t{type(exc).__name__}: {exc}\n"
        try:
            with open(skip_log_path, "a", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass


def get_args_parser():
    parser = argparse.ArgumentParser("Cache DACVAE audio latents (HF dataset, CPU pool)", add_help=True)
    parser.add_argument("--hf_dataset", required=True, type=str)
    parser.add_argument("--hf_split", default="train", type=str)
    parser.add_argument("--hf_data_dir", default=None, type=str)
    parser.add_argument("--hf_cache_dir", default=None, type=str)
    parser.add_argument("--cached_path", default=None, type=str)
    parser.add_argument("--weights", default="facebook/dacvae-watermarked", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument(
        "--name_column",
        nargs="?",
        const="audiocap_id",
        default=None,
        type=str,
        help="Use this dataset column for output filenames; omit to use row index.",
    )

    parser.add_argument("--chunk_size_latents", default=1024, type=int)
    parser.add_argument("--overlap_latents", default=12, type=int)
    parser.add_argument("--min_duration_seconds", default=0.05, type=float)
    parser.add_argument("--max_duration_seconds", default=600.0, type=float)

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--worker_threads", default=1, type=int)
    parser.add_argument("--mp_start_method", default="fork", choices=["fork", "spawn", "forkserver"])

    parser.add_argument("--node_rank", default=None, type=int)
    parser.add_argument("--node_world_size", default=None, type=int)
    return parser


@torch.no_grad()
def main(args):
    data_dir = args.hf_data_dir
    if isinstance(data_dir, str):
        if data_dir.strip() == "" or data_dir.strip().lower() in {"none", "null"}:
            data_dir = None
    cache_dir = args.hf_cache_dir
    if isinstance(cache_dir, str) and cache_dir.strip() == "":
        cache_dir = None

    if args.cached_path:
        cached_path = Path(os.path.normpath(args.cached_path)).resolve()
    else:
        dataset_tag = args.hf_dataset.replace("/", "_")
        cached_path = Path(os.path.normpath(f"{dataset_tag}_{args.hf_split}_cached")).resolve()
    cached_path.mkdir(parents=True, exist_ok=True)

    weights_ref = args.weights
    if os.path.exists(weights_ref):
        weights_path = weights_ref
    elif weights_ref.startswith("facebook/"):
        weights_path = weights_ref
    else:
        raise FileNotFoundError(f"Missing weights: {weights_ref}")

    node_rank = _resolve_node_setting(args.node_rank, ["NODE_RANK", "SLURM_NODEID"], default=None)
    node_world_size = _resolve_node_setting(
        args.node_world_size, ["NODE_WORLD_SIZE", "SLURM_JOB_NUM_NODES"], default=1
    )
    if node_world_size > 1 and node_rank is None:
        raise ValueError("node_rank is required when node_world_size > 1")
    if node_rank is None:
        node_rank = 0
    if node_rank < 0 or node_rank >= node_world_size:
        raise ValueError("node_rank must be in [0, node_world_size)")

    if node_rank == 0:
        print(f"Scanning cache output for existing .pt files under: {cached_path}")
    done_cache = scan_cached_outputs(cached_path)
    if node_rank == 0:
        print(f"Found {len(done_cache)} cached files.")

    ds = load_dataset(args.hf_dataset, data_dir=data_dir, split=args.hf_split, cache_dir=cache_dir)
    total = len(ds)
    if node_rank == 0:
        print(f"HF dataset: {args.hf_dataset} ({args.hf_split}), n={total}.")

    tasks = []
    name_column = args.name_column
    name_ds = None
    if name_column is not None:
        if name_column not in ds.column_names:
            raise ValueError(f"name_column {name_column!r} not found in dataset columns: {ds.column_names}")
        name_ds = ds.select_columns([name_column])
    for idx in range(total):
        name_value = name_ds[idx][name_column] if name_ds is not None else None
        _, out_name = _resolve_uid(idx, name_value)
        if out_name in done_cache:
            continue
        if not _belongs_to_node(out_name, node_rank, node_world_size):
            continue
        tasks.append(idx)

    print(
        f"Node {node_rank}/{node_world_size}: {len(tasks)} files to process. "
        f"Cache output: {cached_path}"
    )

    if not tasks:
        return

    ctx = torch.multiprocessing.get_context(args.mp_start_method)
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(
            weights_path,
            args.device,
            args.hf_dataset,
            args.hf_split,
            data_dir,
            cache_dir,
            str(cached_path),
            name_column,
            args.min_duration_seconds,
            args.max_duration_seconds,
            args.chunk_size_latents,
            args.overlap_latents,
            args.worker_threads,
            node_rank,
        ),
    ) as pool:
        for _ in tqdm(pool.imap_unordered(_process_index, tasks), total=len(tasks), desc="Caching"):
            pass


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
