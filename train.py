# train.py
import argparse
import os

import torch
import lightning as L
from lightning.pytorch import seed_everything

from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from src.lightning import LitModule, EMAWeightAveraging
from src.data_utils.datamodule import CachedLatentsDataModule


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seed_everything(args.global_seed, workers=True)

    seq_len = args.input_size * args.input_size
    dm = CachedLatentsDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=seq_len,
        num_workers=args.num_workers,
    )

    lit = LitModule(
        model_name=args.model,
        seq_len=seq_len,
        latent_size=args.latent_size,
        num_classes=args.num_classes,
        prediction_type=args.prediction_type,
        batch_mul=args.batch_mul,
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
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--results-dir", type=str, default="results_lightning")

    p.add_argument("--model", type=str, default="Transformer-L")
    p.add_argument("--input-size", type=int, required=True)
    p.add_argument("--latent-size", type=int, required=True)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--prediction-type", type=str, default="flow")
    p.add_argument("--batch-mul", type=int, default=2)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training.")

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
