"""Quick batch-size finder using dummy data (no real dataset needed)."""

import torch
import sys
sys.path.insert(0, ".")

from src.lightning import LitModule

# ── Config ──────────────────────────────────────────────────────────────
MODEL       = "AR-DiT-Medium" #"MaskedAR-H2"   # or "AR-DiT-H"
IS_GATED    = True
SEQ_LEN     = 251
LATENT_SIZE = 128              # posterior_params dim = 256 (mean + logvar)
CLAP_DIM    = 512
T5_DIM      = 1024
T5_TOKENS   = 68
DEVICE      = "cuda"
DTYPE       = torch.float32

MIN_BS      = 1
MAX_BS      = 128
# ────────────────────────────────────────────────────────────────────────

def make_batch(bs):
    posterior_params = torch.randn(bs, SEQ_LEN, LATENT_SIZE * 2, device=DEVICE, dtype=DTYPE)
    prompts = {
        "clap":    torch.randn(bs, CLAP_DIM, device=DEVICE, dtype=DTYPE),
        "t5":      torch.randn(bs, T5_TOKENS, T5_DIM, device=DEVICE, dtype=DTYPE),
        "t5_mask": torch.ones(bs, T5_TOKENS, device=DEVICE, dtype=torch.bool),
    }
    return posterior_params, prompts


def try_batch(model, bs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = make_batch(bs)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.training_step(batch, 0)
        loss.backward()
        model.zero_grad(set_to_none=True)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  bs={bs:>3d}  ✓  peak={peak_gb:.2f} GB")
        del loss, batch
        torch.cuda.empty_cache()
        return True, peak_gb
    except torch.cuda.OutOfMemoryError:
        del batch
        torch.cuda.empty_cache()
        print(f"  bs={bs:>3d}  ✗  OOM")
        return False, None


def main():
    print(f"Model: {MODEL}  gated={IS_GATED}")
    lit = LitModule(
        model_name=MODEL,
        seq_len=SEQ_LEN,
        latent_size=LATENT_SIZE,
        clap_dim=CLAP_DIM,
        t5_dim=T5_DIM,
        prompt_seq_len=1 + T5_TOKENS,  # 1 CLAP + T5
        is_gated=IS_GATED,
    )
    lit = lit.to(DEVICE, dtype=DTYPE)
    lit.train()

    total_params = sum(p.numel() for p in lit.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Binary search
    lo, hi = MIN_BS, MAX_BS
    best_bs, best_peak = 0, 0.0

    while lo <= hi:
        mid = (lo + hi) // 2
        ok, peak = try_batch(lit, mid)
        if ok:
            best_bs, best_peak = mid, peak
            lo = mid + 1
        else:
            hi = mid - 1

    print(f"\nMax batch size: {best_bs}  (peak {best_peak:.2f} GB)")


if __name__ == "__main__":
    main()
