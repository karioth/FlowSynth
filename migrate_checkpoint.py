from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch


FALLBACK_ALLOWED_KEYS = {
    "model_name",
    "seq_len",
    "latent_size",
    "clap_dim",
    "t5_dim",
    "prompt_seq_len",
    "prediction_type",
    "t_m",
    "t_s",
    "batch_mul",
    "mask_prob",
    "lr",
    "weight_decay",
    "lr_scheduler",
    "lr_warmup_steps",
}


def _get_allowed_keys() -> set[str]:
    try:
        from src.lightning import LitModule

        signature = inspect.signature(LitModule.__init__)
        return {name for name in signature.parameters if name != "self"}
    except Exception as exc:
        print(
            "Warning: could not import LitModule to read its signature. "
            f"Using fallback allowlist. ({exc})"
        )
        return set(FALLBACK_ALLOWED_KEYS)


def _to_dict(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _remap_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        if key in {"scaling_factor", "bias_factor", "has_scaling"}:
            continue
        if "model.label_embedder." in key:
            key = key.replace("model.label_embedder.", "model.prompt_embedder.")
        remapped[key] = value
    return remapped


def _sanitize_hparams(
    hparams: dict,
    allowed_keys: set[str],
    overrides: dict[str, object],
) -> dict:
    filtered = {key: value for key, value in hparams.items() if key in allowed_keys}

    if "model_name" not in filtered and "model" in hparams:
        filtered["model_name"] = hparams["model"]
    if "latent_size" not in filtered and "latent_dim" in hparams:
        filtered["latent_size"] = hparams["latent_dim"]

    for key, value in overrides.items():
        if value is not None:
            filtered[key] = value

    if "seq_len" not in filtered:
        if "input_size" in hparams:
            filtered["seq_len"] = int(hparams["input_size"]) ** 2
        else:
            raise ValueError(
                "seq_len is missing after sanitization. Provide --seq-len to this script "
                "or ensure the checkpoint has seq_len in its hyperparameters."
            )

    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip legacy hparams from an audio checkpoint for current LitModule loading."
    )
    parser.add_argument("checkpoint", type=str, help="Path to .ckpt file")
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint path")
    parser.add_argument("--seq-len", type=int, default=None, help="Override seq_len")
    parser.add_argument("--model-name", type=str, default=None, help="Override model_name")
    parser.add_argument("--latent-size", type=int, default=None, help="Override latent_size")
    parser.add_argument("--clap-dim", type=int, default=None, help="Override clap_dim")
    parser.add_argument("--t5-dim", type=int, default=None, help="Override t5_dim")
    parser.add_argument(
        "--prompt-seq-len",
        type=int,
        default=None,
        help="Override prompt_seq_len",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = args.output
    if output_path is None:
        output_path = checkpoint_path.with_suffix(".audioonly.ckpt")
    output_path = Path(output_path).expanduser().resolve()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    hparams = checkpoint.get("hyper_parameters")
    hparams_dict = _to_dict(hparams) if hparams is not None else {}
    if not hparams_dict:
        hparams_dict = _to_dict(checkpoint.get("hparams", {}))

    allowed_keys = _get_allowed_keys()
    overrides = {
        "seq_len": args.seq_len,
        "model_name": args.model_name,
        "latent_size": args.latent_size,
        "clap_dim": args.clap_dim,
        "t5_dim": args.t5_dim,
        "prompt_seq_len": args.prompt_seq_len,
    }
    sanitized_hparams = _sanitize_hparams(hparams_dict, allowed_keys, overrides)

    checkpoint["hyper_parameters"] = sanitized_hparams
    if "hparams" in checkpoint:
        checkpoint["hparams"] = sanitized_hparams

    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = _remap_state_dict_keys(checkpoint["state_dict"])

    torch.save(checkpoint, output_path)

    removed = sorted(set(hparams_dict) - set(sanitized_hparams))
    print(f"Wrote sanitized checkpoint: {output_path}")
    if removed:
        print(f"Removed hparams keys: {', '.join(removed)}")


if __name__ == "__main__":
    main()
