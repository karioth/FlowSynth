import argparse
import hashlib
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from src.data_utils.audio_utils import list_audio_files, scan_cached_outputs
from src.data_utils.preencode_audio_latents import cache_audio_latents

_MODEL = None
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


def _init_worker(
    weights_path: str,
    device: str,
    input_root: str,
    out_root: str,
    min_duration_seconds: float,
    max_duration_seconds: float,
    chunk_size_latents: int,
    overlap_latents: int,
    worker_threads: int,
    node_rank: int,
):
    global _MODEL, _WORKER_STATE

    torch.set_num_threads(int(worker_threads))
    torch.set_num_interop_threads(1)

    import dacvae

    _MODEL = dacvae.DACVAE.load(weights_path).eval().to(device)
    out_root_path = Path(out_root)
    log_dir = out_root_path.parent
    _WORKER_STATE = {
        "input_root": Path(input_root),
        "out_root": out_root_path,
        "min_duration_seconds": float(min_duration_seconds),
        "max_duration_seconds": float(max_duration_seconds),
        "chunk_size_latents": int(chunk_size_latents),
        "overlap_latents": int(overlap_latents),
        "skip_log_path": str(log_dir / f"skipped_files.node{int(node_rank)}.log"),
        "event_log_path": str(log_dir / f"processing_events.node{int(node_rank)}.log"),
    }


def _process_path(path_str: str):
    path = Path(path_str)
    input_root = _WORKER_STATE["input_root"]
    out_root = _WORKER_STATE["out_root"]
    skip_log_path = _WORKER_STATE["skip_log_path"]
    event_log_path = _WORKER_STATE["event_log_path"]

    rel = path.relative_to(input_root)
    out_rel = rel.with_suffix(".pt")
    out_path = out_root / out_rel

    if out_path.exists():
        return

    try:
        wav, sr = torchaudio.load(str(path))
        wav = wav.clamp_(-1, 1)

        num_frames = wav.shape[-1]
        min_frames = int(_WORKER_STATE["min_duration_seconds"] * float(sr))
        if num_frames < min_frames:
            print(f"too short, skipping: {path}")
            try:
                with open(event_log_path, "a", encoding="utf-8") as f:
                    f.write(f"too short, skipping: {path}\n")
            except Exception:
                pass
            msg = f"{path}\ttoo_short\n"
            try:
                with open(skip_log_path, "a", encoding="utf-8") as f:
                    f.write(msg)
            except Exception:
                pass
            return

        max_frames = int(_WORKER_STATE["max_duration_seconds"] * float(sr))
        if max_frames > 0 and num_frames > max_frames:
            print(f"file too long, taking random 10min crop: {path}")
            try:
                with open(event_log_path, "a", encoding="utf-8") as f:
                    f.write(f"file too long, taking random 10min crop: {path}\n")
            except Exception:
                pass
            max_start = num_frames - max_frames
            found = False
            for _ in range(10):
                start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
                seg = wav[:, start : start + max_frames]
                if _non_silence_ratio(seg) >= 0.7:
                    wav = seg
                    found = True
                    break
            if not found:
                wav = seg

        item = (str(path), wav, int(sr), str(out_path))
        cache_audio_latents(
            [[item]],
            _MODEL,
            chunk_size_latents=_WORKER_STATE["chunk_size_latents"],
            overlap_latents=_WORKER_STATE["overlap_latents"],
            rank=os.getpid(),
        )
    except Exception as exc:
        msg = f"{path}\t{type(exc).__name__}: {exc}\n"
        try:
            with open(skip_log_path, "a", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass


def get_args_parser():
    parser = argparse.ArgumentParser("Cache DACVAE audio latents (files, CPU pool)", add_help=True)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--cached_path", default=None, type=str)
    parser.add_argument("--weights", default="facebook/dacvae-watermarked", type=str)
    parser.add_argument("--device", default="cpu", type=str)

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
    data_dir = Path(os.path.normpath(args.data_dir)).resolve()
    cached_path = Path(os.path.normpath(args.cached_path or f"{data_dir}_cached")).resolve()
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
        print(f"Scanning already cached .pt files under: {cached_path}")
    done_cache = scan_cached_outputs(cached_path)
    if node_rank == 0:
        print(f"Found {len(done_cache)} cached files.")

    files = list_audio_files(data_dir)
    if node_rank == 0:
        print(f"Found {len(files)} files under: {data_dir}")

    tasks = []
    for path in files:
        rel = path.relative_to(data_dir)
        out_rel = rel.with_suffix(".pt").as_posix()
        if out_rel in done_cache:
            continue
        if not _belongs_to_node(out_rel, node_rank, node_world_size):
            continue
        tasks.append(str(path))

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
            str(data_dir),
            str(cached_path),
            args.min_duration_seconds,
            args.max_duration_seconds,
            args.chunk_size_latents,
            args.overlap_latents,
            args.worker_threads,
            node_rank,
        ),
    ) as pool:
        for _ in tqdm(pool.imap_unordered(_process_path, tasks), total=len(tasks), desc="Caching"):
            pass


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
