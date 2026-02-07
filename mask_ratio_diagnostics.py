import os
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.flow_matching import FlowMatchingSchedulerMaskedAR


# -----------------------------------------------------------------------------
# Hardcoded experiment config (edit these manually)
# -----------------------------------------------------------------------------
SEED = 1234
DEVICE = "cpu"  # "cpu", "mps", or "cuda"

NUM_STEPS = 5000
BATCH_SIZE = 256
SEQ_LEN = 251

MASK_PROB = 0.75

# Empirical mask-ratio prior controls (post-Gumbel semantics).
# - Bounds are where tails should softly die in observed per-sequence ratios.
# - Mode controls where the peak sits inside those bounds.
EMPIRICAL_MIN_RATIO = 0.3
EMPIRICAL_MAX_RATIO = 0.9
EMPIRICAL_MODE_RATIO = 0.8

# Fixed internal shape constant (matches scheduler default).
SPLIT_SIGMA_DIV = 2.5


# CPU parallelism knobs
NUM_WORKERS = 62
CHUNK_STEPS = 250
TORCH_THREADS_PER_WORKER = 2

STORE_ALL_MASK_BATCHES = True
PLOT_BATCH_INDEX = 0

SAVE_PLOT = True
SHOW_PLOT = True
PLOT_PATH = "mask_ratio_distribution.png"

SAVE_RATIOS = False
RATIOS_PATH = "mask_ratios.npy"


def _resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE is set to cuda, but CUDA is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("DEVICE is set to mps, but MPS is not available.")
    return device


def _build_scheduler() -> FlowMatchingSchedulerMaskedAR:
    scheduler = FlowMatchingSchedulerMaskedAR(mask_prob=MASK_PROB, batch_mul=1)

    scheduler.mask_ratio_empirical_min = EMPIRICAL_MIN_RATIO
    scheduler.mask_ratio_empirical_max = EMPIRICAL_MAX_RATIO
    scheduler.mask_ratio_empirical_mode = EMPIRICAL_MODE_RATIO
    scheduler.mask_ratio_split_sigma_div = SPLIT_SIGMA_DIV

    return scheduler


def _split_normal_params(
    lower: float,
    mode: float,
    upper: float,
    sigma_div: float,
) -> Tuple[float, float, float, float, float]:
    if not (0.0 < lower < mode < upper < 1.0):
        raise ValueError(
            "Expected 0 < lower < mode < upper < 1, "
            f"got lower={lower}, mode={mode}, upper={upper}."
        )
    if (not np.isfinite(sigma_div)) or sigma_div <= 0.0:
        raise ValueError(f"sigma_div must be finite and > 0, got {sigma_div}.")

    left_span = mode - lower
    right_span = upper - mode
    sigma_left = left_span / sigma_div
    sigma_right = right_span / sigma_div
    if sigma_left <= 0.0 or sigma_right <= 0.0:
        raise ValueError(
            "Invalid split-normal spans. "
            f"left_span={left_span}, right_span={right_span}."
        )

    sqrt_two = math.sqrt(2.0)
    left_erf_max = math.erf(left_span / (sigma_left * sqrt_two))
    right_erf_max = math.erf(right_span / (sigma_right * sqrt_two))
    if left_erf_max <= 0.0 or right_erf_max <= 0.0:
        raise ValueError(
            "Degenerate split-normal truncation; erf limits must be > 0. "
            f"Got left={left_erf_max}, right={right_erf_max}."
        )

    left_area = sigma_left * left_erf_max
    right_area = sigma_right * right_erf_max
    total_area = left_area + right_area
    if total_area <= 0.0:
        raise ValueError("Degenerate split-normal area; total area must be > 0.")

    p_left = left_area / total_area
    return sigma_left, sigma_right, left_erf_max, right_erf_max, p_left


def _split_normal_ratio_cdf(
    x: np.ndarray,
    lower: float,
    mode: float,
    upper: float,
    sigma_div: float,
) -> np.ndarray:
    sigma_left, sigma_right, left_erf_max, right_erf_max, p_left = _split_normal_params(
        lower=lower,
        mode=mode,
        upper=upper,
        sigma_div=sigma_div,
    )

    tx = torch.as_tensor(x, dtype=torch.float64)
    cdf_x = torch.zeros_like(tx)

    sqrt_two = math.sqrt(2.0)
    left_mask = (tx >= lower) & (tx < mode)
    if bool(left_mask.any()):
        z_left = (mode - tx[left_mask]) / (sigma_left * sqrt_two)
        cdf_x[left_mask] = p_left * (1.0 - torch.erf(z_left) / left_erf_max)

    right_mask = (tx >= mode) & (tx < upper)
    if bool(right_mask.any()):
        z_right = (tx[right_mask] - mode) / (sigma_right * sqrt_two)
        cdf_x[right_mask] = p_left + (1.0 - p_left) * (torch.erf(z_right) / right_erf_max)

    cdf_x = torch.where(tx >= upper, torch.ones_like(cdf_x), cdf_x)
    return cdf_x.cpu().numpy()


def _configured_schedule_pmf(seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}.")

    ratio_grid = np.arange(seq_len + 1, dtype=np.float64) / float(seq_len)
    left_edges = (np.arange(seq_len + 1, dtype=np.float64) - 0.5) / float(seq_len)
    right_edges = (np.arange(seq_len + 1, dtype=np.float64) + 0.5) / float(seq_len)
    left_edges = np.clip(left_edges, 0.0, 1.0)
    right_edges = np.clip(right_edges, 0.0, 1.0)

    lower = float(EMPIRICAL_MIN_RATIO)
    upper = float(EMPIRICAL_MAX_RATIO)
    mode = float(EMPIRICAL_MODE_RATIO)

    cdf_left = _split_normal_ratio_cdf(
        left_edges,
        lower=lower,
        mode=mode,
        upper=upper,
        sigma_div=float(SPLIT_SIGMA_DIV),
    )
    cdf_right = _split_normal_ratio_cdf(
        right_edges,
        lower=lower,
        mode=mode,
        upper=upper,
        sigma_div=float(SPLIT_SIGMA_DIV),
    )

    pmf = np.clip(cdf_right - cdf_left, 0.0, 1.0)
    pmf /= pmf.sum()
    return ratio_grid, pmf


def _sample_mask_batch(
    scheduler: FlowMatchingSchedulerMaskedAR,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    mask, _ = scheduler._sample_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    return mask


def _run_chunk(
    start_step: int,
    num_steps: int,
    seed: int,
    device_str: str,
    store_masks: bool,
) -> Tuple[int, np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
    device = _resolve_device(device_str)
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(TORCH_THREADS_PER_WORKER)))

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))

    scheduler = _build_scheduler()

    all_masks = [] if store_masks else None
    ratios_chunk = np.empty(num_steps * BATCH_SIZE, dtype=np.float32)
    quantiles_chunk = np.empty((num_steps, 3), dtype=np.float32)  # q10, q50, q90

    cursor = 0
    for step in range(num_steps):
        mask = _sample_mask_batch(
            scheduler=scheduler,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            device=device,
        )

        if all_masks is not None:
            all_masks.append(mask.cpu().numpy())

        ratios = mask.float().mean(dim=1).cpu().numpy()
        ratios_chunk[cursor:cursor + BATCH_SIZE] = ratios
        quantiles_chunk[step] = np.quantile(ratios, [0.10, 0.50, 0.90])
        cursor += BATCH_SIZE

    return start_step, ratios_chunk, quantiles_chunk, all_masks


def run_experiment() -> Tuple[Optional[List[np.ndarray]], np.ndarray, np.ndarray]:
    device = _resolve_device(DEVICE)
    print(
        f"Running mask diagnostic: device={DEVICE}, steps={NUM_STEPS}, "
        f"batch={BATCH_SIZE}, seq_len={SEQ_LEN}, workers={NUM_WORKERS}, "
        f"empirical_prior=(min={EMPIRICAL_MIN_RATIO}, max={EMPIRICAL_MAX_RATIO}, "
        f"mode={EMPIRICAL_MODE_RATIO}, split_sigma_div={SPLIT_SIGMA_DIV})"
    )

    if NUM_WORKERS <= 1:
        start_step, ratios_chunk, quantiles_chunk, all_masks = _run_chunk(
            start_step=0,
            num_steps=NUM_STEPS,
            seed=SEED,
            device_str=DEVICE,
            store_masks=STORE_ALL_MASK_BATCHES,
        )
        del start_step
        return all_masks, ratios_chunk, quantiles_chunk

    if device.type != "cpu":
        raise ValueError("NUM_WORKERS > 1 is currently supported only for DEVICE='cpu'.")

    if CHUNK_STEPS <= 0:
        raise ValueError(f"CHUNK_STEPS must be > 0, got {CHUNK_STEPS}.")

    if STORE_ALL_MASK_BATCHES:
        print(
            "Warning: STORE_ALL_MASK_BATCHES=True with multiprocessing can use a lot of RAM."
        )

    total_cpus = os.cpu_count() or 1
    worker_count = max(1, min(int(NUM_WORKERS), total_cpus))
    if worker_count != NUM_WORKERS:
        print(f"Clamping NUM_WORKERS from {NUM_WORKERS} to {worker_count} (available CPUs).")

    chunk_specs: list[tuple[int, int]] = []
    for start in range(0, NUM_STEPS, CHUNK_STEPS):
        chunk_specs.append((start, min(CHUNK_STEPS, NUM_STEPS - start)))

    per_sequence_ratios = np.empty(NUM_STEPS * BATCH_SIZE, dtype=np.float32)
    per_step_quantiles = np.empty((NUM_STEPS, 3), dtype=np.float32)
    all_masks: Optional[List[np.ndarray]]
    if STORE_ALL_MASK_BATCHES:
        all_masks = [np.zeros((0, 0), dtype=np.float32) for _ in range(NUM_STEPS)]
    else:
        all_masks = None

    completed_steps = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for chunk_idx, (start, count) in enumerate(chunk_specs):
            chunk_seed = SEED + 100_003 * (chunk_idx + 1)
            futures.append(
                executor.submit(
                    _run_chunk,
                    start,
                    count,
                    chunk_seed,
                    DEVICE,
                    STORE_ALL_MASK_BATCHES,
                )
            )

        for future in as_completed(futures):
            start, ratios_chunk, quantiles_chunk, masks_chunk = future.result()
            count = quantiles_chunk.shape[0]
            end = start + count

            per_step_quantiles[start:end] = quantiles_chunk
            ratio_start = start * BATCH_SIZE
            ratio_end = end * BATCH_SIZE
            per_sequence_ratios[ratio_start:ratio_end] = ratios_chunk

            if all_masks is not None and masks_chunk is not None:
                all_masks[start:end] = masks_chunk

            completed_steps += count
            print(f"[{completed_steps:5d}/{NUM_STEPS}] steps complete")

    return all_masks, per_sequence_ratios, per_step_quantiles


def _plot_results(
    all_masks: Optional[List[np.ndarray]],
    per_sequence_ratios: np.ndarray,
    per_step_quantiles: np.ndarray,
    configured_ratio_grid: np.ndarray,
    configured_pmf: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Distribution over all sequence ratios across simulated training steps.
    # Ratios are discrete (m / SEQ_LEN), so plot the exact PMF to avoid binning artifacts.
    masked_counts = np.rint(per_sequence_ratios * SEQ_LEN).astype(np.int32)
    masked_counts = np.clip(masked_counts, 0, SEQ_LEN)
    pmf = np.bincount(masked_counts, minlength=SEQ_LEN + 1).astype(np.float64)
    pmf /= pmf.sum()
    ratio_grid = np.arange(SEQ_LEN + 1, dtype=np.float64) / float(SEQ_LEN)
    axes[0].bar(ratio_grid, pmf, width=0.92 / float(SEQ_LEN), alpha=0.85, align="center")
    # axes[0].plot(
    #     configured_ratio_grid,
    #     configured_pmf,
    #     color="crimson",
    #     linewidth=1.8,
    #     label="configured empirical prior",
    # )
    axes[0].axvline(MASK_PROB, linestyle="--", label="target p_global")
    axes[0].axvline(per_sequence_ratios.mean(), linestyle="-", label="empirical mean")
    axes[0].set_title("Per-sequence mask ratio distribution")
    axes[0].set_xlabel("mask ratio")
    axes[0].set_ylabel("probability mass")
    axes[0].set_xlim(-0.5 / float(SEQ_LEN), 1.0 + 0.5 / float(SEQ_LEN))
    axes[0].legend()

    # 2) Evolution of sequence-level spread over steps
    steps = np.arange(NUM_STEPS)
    axes[1].plot(steps, per_step_quantiles[:, 0], label="q10")
    axes[1].plot(steps, per_step_quantiles[:, 1], label="q50")
    axes[1].plot(steps, per_step_quantiles[:, 2], label="q90")
    axes[1].axhline(MASK_PROB, linestyle="--", label="target p_global")
    axes[1].set_title("Per-step sequence ratio quantiles")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("mask ratio")
    axes[1].legend()

    # 3) Visual snapshot of one sampled batch mask
    if all_masks is None or len(all_masks) == 0:
        preview_mask = np.zeros((BATCH_SIZE, SEQ_LEN), dtype=np.float32)
        preview_title = "Batch mask snapshot (not stored)"
    else:
        idx = max(0, min(PLOT_BATCH_INDEX, len(all_masks) - 1))
        preview_mask = all_masks[idx].astype(np.float32)
        preview_title = f"Batch mask snapshot (step {idx})"

    im = axes[2].imshow(preview_mask, aspect="auto", interpolation="nearest")
    axes[2].set_title(preview_title)
    axes[2].set_xlabel("token position")
    axes[2].set_ylabel("sequence in batch")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()

    if SAVE_PLOT:
        fig.savefig(PLOT_PATH, dpi=180)
        print(f"Saved plot to: {PLOT_PATH}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


def main() -> Tuple[Optional[List[np.ndarray]], np.ndarray, np.ndarray]:
    all_masks, per_sequence_ratios, per_step_quantiles = run_experiment()
    configured_ratio_grid, configured_pmf = _configured_schedule_pmf(SEQ_LEN)
    configured_mean = float(np.sum(configured_ratio_grid * configured_pmf))

    print("\nDiagnostics")
    print(f"  samples: {per_sequence_ratios.size}")
    print(
        f"  empirical prior (min, max, mode): "
        f"({EMPIRICAL_MIN_RATIO:.6f}, {EMPIRICAL_MAX_RATIO:.6f}, {EMPIRICAL_MODE_RATIO:.6f})"
    )
    print(f"  split sigma divisor (fixed): {SPLIT_SIGMA_DIV:.6f}")
    print(f"  target p_global: {MASK_PROB:.6f}")
    print(f"  configured prior mean: {configured_mean:.6f}")
    print(f"  empirical mean:  {per_sequence_ratios.mean():.6f}")
    print(f"  empirical std:   {per_sequence_ratios.std():.6f}")
    print(
        "  percentiles (1,5,50,95,99):",
        np.percentile(per_sequence_ratios, [1, 5, 50, 95, 99]).round(6).tolist(),
    )

    if SAVE_RATIOS:
        np.save(RATIOS_PATH, per_sequence_ratios)
        print(f"Saved ratios to: {RATIOS_PATH}")

    _plot_results(
        all_masks,
        per_sequence_ratios,
        per_step_quantiles,
        configured_ratio_grid,
        configured_pmf,
    )
    return all_masks, per_sequence_ratios, per_step_quantiles


if __name__ == "__main__":
    mask_batches, mask_ratios, step_quantiles = main()
