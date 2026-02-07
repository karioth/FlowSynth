import numpy as np
import torch
import matplotlib.pyplot as plt

from src.flow_matching import FlowMatchingSchedulerMaskedAR


# -----------------------------------------------------------------------------
# Hardcoded experiment config (edit these manually)
# -----------------------------------------------------------------------------
SEED = 123
DEVICE = "cpu"  # "cpu", "mps", or "cuda"

NUM_STEPS = 5000
BATCH_SIZE = 32
SEQ_LEN = 251

MASK_PROB = 0.77
MASK_SIGMA = 64
MASK_EXPANSION = 2
MASK_KAPPA = 1.0

STORE_ALL_MASK_BATCHES = True
PLOT_BATCH_INDEX = 0

SAVE_PLOT = True
SHOW_PLOT = True
PLOT_PATH = "mask_ratio_distribution.png"

SAVE_RATIOS = True
RATIOS_PATH = "mask_ratios.npy"


def _resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE is set to cuda, but CUDA is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("DEVICE is set to mps, but MPS is not available.")
    return device


def _sample_mask_batch(
    scheduler: FlowMatchingSchedulerMaskedAR,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    total_tokens = batch_size * seq_len
    num_masked = round(float(scheduler.mask_prob) * total_tokens)
    if num_masked <= 0:
        raise ValueError(
            "round(mask_prob * total_tokens) must be > 0 for this diagnostic. "
            f"Got mask_prob={scheduler.mask_prob}, total_tokens={total_tokens}."
        )

    flat_mask_indices = scheduler._sample_mask_indices(
        total_tokens=total_tokens,
        num_masked=num_masked,
        device=device,
    )
    mask_flat = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    mask_flat[flat_mask_indices] = True
    return mask_flat.view(batch_size, seq_len)


def run_experiment() -> tuple[list[torch.Tensor] | None, np.ndarray, np.ndarray]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = _resolve_device(DEVICE)

    scheduler = FlowMatchingSchedulerMaskedAR(mask_prob=MASK_PROB, batch_mul=1)
    scheduler.mask_sigma = MASK_SIGMA
    scheduler.mask_expansion = MASK_EXPANSION
    scheduler.mask_kappa = MASK_KAPPA

    all_masks = [] if STORE_ALL_MASK_BATCHES else None
    per_sequence_ratios = np.empty(NUM_STEPS * BATCH_SIZE, dtype=np.float32)
    per_step_quantiles = np.empty((NUM_STEPS, 3), dtype=np.float32)  # q10, q50, q90

    cursor = 0
    progress_every = max(1, NUM_STEPS // 10)

    for step in range(NUM_STEPS):
        mask = _sample_mask_batch(
            scheduler=scheduler,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            device=device,
        )

        if all_masks is not None:
            all_masks.append(mask.cpu())

        ratios = mask.float().mean(dim=1).cpu().numpy()
        per_sequence_ratios[cursor:cursor + BATCH_SIZE] = ratios
        per_step_quantiles[step] = np.quantile(ratios, [0.10, 0.50, 0.90])
        cursor += BATCH_SIZE

        if (step + 1) % progress_every == 0 or step == 0:
            print(f"[{step + 1:5d}/{NUM_STEPS}] mean ratio this batch: {ratios.mean():.4f}")

    return all_masks, per_sequence_ratios, per_step_quantiles


def _plot_results(
    all_masks: list[torch.Tensor] | None,
    per_sequence_ratios: np.ndarray,
    per_step_quantiles: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Distribution over all sequence ratios across simulated training steps
    bins = min(100, max(30, int(np.sqrt(per_sequence_ratios.size))))
    axes[0].hist(per_sequence_ratios, bins=bins, density=True, alpha=0.85)
    axes[0].axvline(MASK_PROB, linestyle="--", label="target p_global")
    axes[0].axvline(per_sequence_ratios.mean(), linestyle="-", label="empirical mean")
    axes[0].set_title("Per-sequence mask ratio distribution")
    axes[0].set_xlabel("mask ratio")
    axes[0].set_ylabel("density")
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
        preview_mask = all_masks[idx].numpy().astype(np.float32)
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


def main() -> tuple[list[torch.Tensor] | None, np.ndarray, np.ndarray]:
    all_masks, per_sequence_ratios, per_step_quantiles = run_experiment()

    print("\nDiagnostics")
    print(f"  samples: {per_sequence_ratios.size}")
    print(f"  target p_global: {MASK_PROB:.6f}")
    print(f"  empirical mean:  {per_sequence_ratios.mean():.6f}")
    print(f"  empirical std:   {per_sequence_ratios.std():.6f}")
    print(
        "  percentiles (1,5,50,95,99):",
        np.percentile(per_sequence_ratios, [1, 5, 50, 95, 99]).round(6).tolist(),
    )

    if SAVE_RATIOS:
        np.save(RATIOS_PATH, per_sequence_ratios)
        print(f"Saved ratios to: {RATIOS_PATH}")

    _plot_results(all_masks, per_sequence_ratios, per_step_quantiles)
    return all_masks, per_sequence_ratios, per_step_quantiles


if __name__ == "__main__":
    mask_batches, mask_ratios, step_quantiles = main()
