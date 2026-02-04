# train_audio.py
import argparse
import os

import torch
import lightning as L
from lightning.pytorch import seed_everything

from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from src.lightning import LitModule, EMAWeightAveraging
from src.data_utils.audio_datamodule import CachedAudioDataModule


def _parse_manifest_paths(values: list[str] | None) -> list[str]:
    if not values:
        raise ValueError("No manifest paths provided")
    manifest_paths = []
    for value in values:
        if value is None:
            continue
        for item in value.split(","):
            item = item.strip()
            if item:
                manifest_paths.append(item)
    if not manifest_paths:
        raise ValueError("No manifest paths provided")
    for path in manifest_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing manifest: {path}")
    return manifest_paths


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seed_everything(args.global_seed, workers=True)

    manifest_paths = _parse_manifest_paths(args.manifest_paths)
    data_root = args.data_root

    silence_path = args.silence_latent_path
    if not os.path.isabs(silence_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        silence_path = os.path.join(script_dir, silence_path)
    if not os.path.exists(silence_path):
        raise FileNotFoundError(f"Missing silence latent: {silence_path}")

    dm = CachedAudioDataModule(
        manifest_paths=manifest_paths,
        data_root=data_root,
        silence_latent_path=silence_path,
        batch_size=args.batch_size,
        target_seq_len=args.seq_len,
        max_t5_tokens=args.prompt_seq_len - 1,
        num_workers=args.num_workers,
    )

    lit = LitModule(
        model_name=args.model,
        seq_len=args.seq_len,
        latent_size=args.latent_size,
        num_classes=1,  # unused for continuous conditioning
        conditioning_type=args.conditioning_type,
        clap_dim=args.clap_dim,
        t5_dim=args.t5_dim,
        prompt_seq_len=args.prompt_seq_len,
        prediction_type=args.prediction_type,
        batch_mul=args.batch_mul,
        mask_prob_min=args.mask_prob_min,
        mask_prob_max=args.mask_prob_max,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
    )

    ckpt_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{step:07d}",
        every_n_train_steps=args.ckpt_every,
        save_last=True,
        save_top_k=-1,
    )

    ema_cb = EMAWeightAveraging()
    pbar = TQDMProgressBar(refresh_rate=args.log_every)

    trainer = L.Trainer(
        default_root_dir=args.results_dir,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        max_epochs=args.epochs,
        precision=args.precision,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[ckpt_cb, ema_cb, pbar],
        log_every_n_steps=args.log_every,
    )

    trainer.fit(lit, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest-paths",
        action="append",
        default=["/share/users/student/f/friverossego/datasets/audio_manifest_train.jsonl"],
        help="JSONL manifest paths (repeatable or comma-separated)",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="/share/users/student/f/friverossego/datasets",
        help="Base directory for relative manifest entries",
    )
    p.add_argument(
        "--silence-latent-path",
        type=str,
        default="silence_samples/silence_10s_dacvae.pt",
        help="Path to silence latent for padding",
    )
    p.add_argument("--results-dir", type=str, default="results_audio")

    p.add_argument("--model", type=str, default="MaskedAR-L")
    p.add_argument("--seq-len", type=int, default=251,
                   help="Audio sequence length (fixed at 251 for DACVAE)")
    p.add_argument("--latent-size", type=int, default=128,
                   help="DACVAE latent dim (128, moments=256)")
    p.add_argument("--conditioning-type", type=str, default="continuous",
                   help="Conditioning type: 'class' or 'continuous'")
    p.add_argument("--clap-dim", type=int, default=512,
                   help="CLAP pooled embedding dimension")
    p.add_argument("--t5-dim", type=int, default=1024,
                   help="T5 hidden state dimension")
    p.add_argument("--prompt-seq-len", type=int, default=69,
                   help="Prompt sequence length (1 CLAP + max T5 tokens)")
    p.add_argument("--prediction-type", type=str, default="flow")
    p.add_argument("--batch-mul", type=int, default=2)
    p.add_argument("--mask-prob-min", type=float, default=0.5)
    p.add_argument("--mask-prob-max", type=float, default=0.5)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=12)

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=10000)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume training.")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-scheduler", type=str, default="cosine")
    p.add_argument("--lr-warmup-steps", type=int, default=1000)

    p.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["bf16-mixed", "16-mixed", "32"],
        help="Lightning Trainer precision."
    )
    p.add_argument("--devices", type=str, default="auto")
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--strategy", type=str, default="ddp")

    main(p.parse_args())
