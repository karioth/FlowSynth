"""
Unified execution runners for preprocessing.

Provides two execution strategies:
- run_pool: CPU multiprocessing pool
- run_gpu_pool: Multi-GPU with shared task queue
"""

from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from .common import configure_worker_threads


# Type aliases
ProcessFn = Callable[[Any], None]  # Process single item
InitFn = Callable[..., None]  # Worker initialization


def _validate_gpu_count(requested: int) -> int:
    """
    Validate GPU request does not exceed available GPUs.

    Args:
        requested: Number of GPUs requested

    Returns:
        Validated GPU count

    Raises:
        ValueError: If requested > available GPUs
    """
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available but gpu_pool was requested")

    available = torch.cuda.device_count()
    if requested > available:
        raise ValueError(
            f"Requested {requested} GPUs but only {available} are visible. "
            f"Set CUDA_VISIBLE_DEVICES or reduce --processes."
        )
    return requested


def _configure_determinism() -> None:
    """
    Configure CUDA determinism settings.

    Sets:
        - CUBLAS_WORKSPACE_CONFIG=:4096:8
        - torch.backends.cuda.matmul.allow_tf32 = False
        - torch.backends.cudnn.allow_tf32 = False
        - torch.backends.cudnn.benchmark = False
        - torch.backends.cudnn.deterministic = True
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_pool(
    items: list[Any],
    process_fn: ProcessFn,
    init_fn: InitFn,
    init_args: tuple,
    num_workers: int,
    threads_per_worker: int = 1,
    mp_start_method: str = "fork",
    desc: str = "Processing",
) -> None:
    """
    Run processing using a multiprocessing Pool (CPU-oriented).

    Args:
        items: List of items to process
        process_fn: Function to process each item (called in worker)
        init_fn: Worker initialization function
        init_args: Arguments for init_fn
        num_workers: Number of worker processes
        threads_per_worker: CPU threads per worker
        mp_start_method: Multiprocessing start method
        desc: Progress bar description
    """
    if not items:
        return

    # Wrap init_fn to also configure threads
    def wrapped_init(*args):
        configure_worker_threads(threads_per_worker)
        init_fn(*args)

    ctx = mp.get_context(mp_start_method)
    with ctx.Pool(
        processes=num_workers,
        initializer=wrapped_init,
        initargs=init_args,
    ) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_fn, items),
            total=len(items),
            desc=desc,
        ):
            pass


def _gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    done_queue: mp.Queue,
    process_fn: ProcessFn,
    init_fn: InitFn,
    init_args: tuple,
    threads_per_gpu: int,
    deterministic: bool,
) -> None:
    """
    GPU worker process that pulls tasks from shared queue.

    Args:
        gpu_id: GPU index to use
        task_queue: Queue to pull tasks from
        done_queue: Queue to signal completion
        process_fn: Function to process each item
        init_fn: Worker initialization function
        init_args: Arguments for init_fn
        threads_per_gpu: CPU threads for this worker
        deterministic: Whether to enable determinism
    """
    # Pin to GPU
    torch.cuda.set_device(gpu_id)

    # Configure threads
    configure_worker_threads(threads_per_gpu)

    # Configure determinism if requested
    if deterministic:
        _configure_determinism()

    # Initialize worker (load model, etc.)
    init_fn(*init_args)

    # Process tasks from queue
    while True:
        item = task_queue.get()
        if item is None:  # Poison pill
            break
        try:
            process_fn(item)
        except Exception as e:
            # Log error but continue processing
            print(f"[GPU {gpu_id}] Error processing item: {e}")
        done_queue.put(1)


def run_gpu_pool(
    items: list[Any],
    process_fn: ProcessFn,
    init_fn: InitFn,
    init_args_fn: Callable[[int], tuple],
    num_gpus: int | None = None,
    threads_per_gpu: int = 2,
    deterministic: bool = True,
    desc: str = "Processing",
) -> None:
    """
    Run processing using one process per GPU with shared task queue.

    This is NOT the DataLoader/DistributedSampler approach. Instead:
    - Spawns one worker process per GPU
    - Uses a shared mp.Queue for dynamic task distribution
    - Each worker does its own I/O + model encoding

    Args:
        items: List of items to process
        process_fn: Function to process each item
        init_fn: Worker initialization function
        init_args_fn: Function that takes gpu_index and returns init_args
        num_gpus: Number of GPUs (default: count visible GPUs)
        threads_per_gpu: CPU threads per GPU worker
        deterministic: Whether to enable CUDA determinism
        desc: Progress bar description
    """
    if not items:
        return

    # Validate and resolve GPU count
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    _validate_gpu_count(num_gpus)

    # Use spawn for CUDA (fork can cause issues)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    done_queue = ctx.Queue()

    # Start worker processes
    workers = []
    for gpu_id in range(num_gpus):
        init_args = init_args_fn(gpu_id)
        p = ctx.Process(
            target=_gpu_worker,
            args=(
                gpu_id,
                task_queue,
                done_queue,
                process_fn,
                init_fn,
                init_args,
                threads_per_gpu,
                deterministic,
            ),
        )
        p.start()
        workers.append(p)

    # Feed tasks to queue
    for item in items:
        task_queue.put(item)

    # Send poison pills to signal completion
    for _ in range(num_gpus):
        task_queue.put(None)

    # Wait for all tasks to complete with progress bar
    for _ in tqdm(range(len(items)), desc=desc):
        done_queue.get()

    # Wait for workers to finish
    for p in workers:
        p.join()

