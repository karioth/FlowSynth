import torch
import torch.distributed as dist
import lightning as L
from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

from diffusers.optimization import get_scheduler

from .models import All_models, DiT, Transformer, AR_DiT, MaskedARTransformer
from .flow_matching import (
    FlowMatchingSchedulerDiT,
    FlowMatchingSchedulerTransformer,
    FlowMatchingSchedulerARDiff,
    FlowMatchingSchedulerMaskedAR,
)
from .models.modules.distributions import SequenceDiagonalGaussianDistribution


class LitModule(L.LightningModule):
    def __init__(
        self,
        model_name: str = "Transformer-L",
        seq_len: int | None = None,
        input_size: int | None = None,
        latent_size: int = 16,
        num_classes: int = 1000,
        prediction_type: str = "flow",
        t_m: float = 0.0,
        t_s: float = 1.0,
        batch_mul: int = 4,
        mask_prob_min: float = 0.5,
        mask_prob_max: float = 0.5,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        lr_scheduler: str = "cosine",
        lr_warmup_steps: int = 1000,
    ):
        super().__init__()
        if seq_len is None:
            if input_size is None:
                seq_len = 1024
            else:
                seq_len = input_size * input_size
        self.save_hyperparameters()

        self.model = All_models[model_name](
            seq_len=seq_len,
            in_channels=latent_size,
            num_classes=num_classes,
        )

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
                mask_prob_min=mask_prob_min,
                mask_prob_max=mask_prob_max,
                batch_mul=batch_mul,
            )
        else:
            raise NotImplementedError("Unsupported model type.")

        self.register_buffer("scaling_factor", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("bias_factor", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("has_scaling", torch.tensor(False, dtype=torch.bool))

    def training_step(self, batch, batch_idx):
        moments, prompts = batch
        posterior = SequenceDiagonalGaussianDistribution(moments)
        x0 = posterior.sample()

        if not self.has_scaling.item():
            self._init_scaling(x0)

        x0 = self._normalize(x0)
        loss = self.noise_scheduler.get_losses(self.model, x0, prompts)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def _init_scaling(self, x0):
        x0_float = x0.float()
        scaling = 1.0 / x0_float.flatten().std()
        bias = -x0_float.flatten().mean()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(scaling, op=dist.ReduceOp.SUM)
            dist.all_reduce(bias, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
            scaling /= world_size
            bias /= world_size

        mu = -bias
        std = 1.0 / scaling
        self.set_data_stats(mu, std)
        self.print(
            f"Scaling factor: {self.scaling_factor.item()}, Bias factor: {self.bias_factor.item()}"
        )

    def set_data_stats(self, mu: torch.Tensor, std: torch.Tensor):
        assert mu.shape == self.bias_factor.shape
        assert std.shape == self.scaling_factor.shape
        std = std.clamp_min(1e-8)
        self.bias_factor.copy_(-mu)
        self.scaling_factor.copy_(1.0 / std)
        self.has_scaling.fill_(True)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias_factor.to(dtype=x.dtype)
        scale = self.scaling_factor.to(dtype=x.dtype)
        return (x + bias) * scale

    def unnormalize_latents(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_scaling.item():
            raise RuntimeError("Scaling/bias not set; cannot unnormalize latents.")
        bias = self.bias_factor.to(dtype=x.dtype)
        scale = self.scaling_factor.to(dtype=x.dtype)
        return x / scale - bias

    @torch.no_grad()
    def sample_latents(
        self,
        prompt,
        cfg_scale: float = 4.0,
        num_inference_steps: int = 250,
        scheduler=None,
        ardiff_step: int = None,
        base_num_frames: int = None,
    ):
        self.eval()
        if scheduler is None:
            scheduler = self.noise_scheduler

        scheduler.configure_sampling(
            ardiff_step=ardiff_step,
            base_num_frames=base_num_frames,
        )
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = self.model.sample_with_cfg(prompt, cfg_scale, scheduler)
        return self.unnormalize_latents(latents)

    def load_state_dict(self, state_dict, strict: bool = True):
        remapped = {}
        for key, value in state_dict.items():
            if "model.label_embedder." in key:
                key = key.replace("model.label_embedder.", "model.prompt_embedder.")
            remapped[key] = value
        return super().load_state_dict(remapped, strict=strict)

    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias") or getattr(p, "_no_weight_decay", False):
                no_decay.append(p)   # RMSNorm/LN weights, biases, scalars
            else:
                decay.append(p)      # linear/conv weights, embedding matrices, etc.

        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.lr,
        )

        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_scheduler(
            self.hparams.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.hparams.lr_warmup_steps,
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
