from typing import Any, cast

import torch
import lightning as L
from lightning.pytorch.callbacks import WeightAveraging
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.swa_utils import get_ema_avg_fn

from diffusers.optimization import get_scheduler

from .models import All_models, DiT, Transformer, AR_DiT, MaskedARTransformer
from .flow_matching import (
    FlowMatchingSchedulerDiT,
    FlowMatchingSchedulerTransformer,
    FlowMatchingSchedulerARDiff,
    FlowMatchingSchedulerMaskedAR,
)
from .utils import sample_posterior


class LitModule(L.LightningModule):
    def __init__(
        self,
        model_name: str = "Transformer-L",
        seq_len: int | None = None,
        latent_size: int = 16,
        clap_dim: int = 512,
        t5_dim: int = 1024,
        prompt_seq_len: int = 69,
        prediction_type: str = "flow",
        t_m: float = 0.0,
        t_s: float = 1.0,
        batch_mul: int = 4,
        mask_prob: float = 0.75,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        lr_scheduler: str = "cosine",
        lr_warmup_steps: int = 1000,
    ):
        super().__init__()
        if seq_len is None:
            raise ValueError("seq_len is required for audio-only training.")
        self.save_hyperparameters()

        model_kwargs = dict(
            seq_len=seq_len,
            in_channels=latent_size,
            clap_dim=clap_dim,
            t5_dim=t5_dim,
            prompt_seq_len=prompt_seq_len,
        )
        self.model = All_models[model_name](**model_kwargs)

        if isinstance(self.model, Transformer):
            self.noise_scheduler = FlowMatchingSchedulerTransformer(
                prediction_type=prediction_type,
                t_m=t_m,
                t_s=t_s,
                batch_mul=batch_mul,
            )
        elif isinstance(self.model, DiT):
            self.noise_scheduler = FlowMatchingSchedulerDiT(
                prediction_type=prediction_type,
                t_m=t_m,
                t_s=t_s,
            )
        elif isinstance(self.model, AR_DiT):
            self.noise_scheduler = FlowMatchingSchedulerARDiff(
                prediction_type=prediction_type,
                t_m=t_m,
                t_s=t_s,
            )
        elif isinstance(self.model, MaskedARTransformer):
            self.noise_scheduler = FlowMatchingSchedulerMaskedAR(
                prediction_type=prediction_type,
                t_m=t_m,
                t_s=t_s,
                mask_prob=mask_prob,
                batch_mul=batch_mul,
            )
        else:
            raise NotImplementedError("Unsupported model type.")

    def training_step(self, batch, batch_idx):
        posterior_params, prompts = batch
        x0 = sample_posterior(posterior_params)

        loss = self.noise_scheduler.get_losses(self.model, x0, prompts)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def sample_latents(
        self,
        prompt,
        cfg_scale: float = 4.0,
        num_inference_steps: int = 250,
        scheduler=None,
        ardiff_step: int | None = None,
        base_num_frames: int | None = None,
    ):
        self.eval()
        if scheduler is None:
            scheduler = self.noise_scheduler

        scheduler.configure_sampling(
            ardiff_step=ardiff_step,
            base_num_frames=base_num_frames,
        )
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        return self.model.sample_with_cfg(prompt, cfg_scale, scheduler)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        hparams = cast(Any, self.hparams)
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias") or getattr(p, "_no_weight_decay", False):
                no_decay.append(p)   # RMSNorm/LN weights, biases, scalars
            else:
                decay.append(p)      # linear/conv weights, embedding matrices, etc.

        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}],
            lr=hparams.lr,
        )

        num_training_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = get_scheduler(
            hparams.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=hparams.lr_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class EMAWeightAveraging(WeightAveraging):
    """
    Matches Lightning's stable-doc example (starts after 100 optimizer steps).
    """
    def __init__(self):
        super().__init__(avg_fn=get_ema_avg_fn(decay=0.9999))

    def should_update(self, step_idx=None, epoch_idx=None):
        return (step_idx is not None) and (step_idx >= 100)
