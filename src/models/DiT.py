import functools

import torch
import torch.nn as nn

from .modules.embeddings import PromptEmbedder, TimestepEmbedder
from .modules.layers import FinalLayer, TransformerBlock


class DiT(nn.Module):
    """
    Simplified DiT without AdaLN modulation.
    Time conditioning is an additive bias applied to latent tokens.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        num_kv_heads: int | None = None,
        intermediate_size: int | None = None,
        prompt_dropout_prob: float = 0.1,
        clap_dim: int = 512,
        t5_dim: int = 1024,
        prompt_seq_len: int = 69,
        is_gated: bool = False,
        rope_theta: float = 10000.0,
        rope_interleaved: bool = False,
        rope_scale_base: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.seq_len = seq_len
        self.prompt_seq_len = prompt_seq_len

        if intermediate_size is None:
            intermediate_size = int(hidden_size * 7 / 3 / 64) * 64

        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=False)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.prompt_embedder = PromptEmbedder(
            clap_dim=clap_dim,
            t5_dim=t5_dim,
            hidden_size=hidden_size,
            prompt_seq_len=prompt_seq_len,
            dropout_prob=prompt_dropout_prob,
        )
        self.time_bias_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    intermediate_size=intermediate_size,
                    layer_idx=idx,
                    is_causal=False,
                    is_gated=is_gated,
                    rope_theta=rope_theta,
                    rope_interleaved=rope_interleaved,
                    rope_scale_base=rope_scale_base,
                )
                for idx in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.constant_(self.time_bias_proj[1].weight, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        prompt: torch.Tensor | dict,
        prompt_drop_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        hidden_states = self.input_embedder(hidden_states)
        time_emb = self.time_embedder(timesteps)
        if time_emb.shape != (hidden_states.size(0), self.hidden_size):
            raise ValueError(
                f"Expected time_emb shape {(hidden_states.size(0), self.hidden_size)}, got {tuple(time_emb.shape)}"
            )

        hidden_states = hidden_states + self.time_bias_proj(time_emb).unsqueeze(1)
        prompt_seq = self.prompt_embedder(
            prompt,
            self.training,
            force_drop_ids=prompt_drop_ids,
        )
        hidden_states = torch.cat([prompt_seq, hidden_states], dim=1)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states[:, self.prompt_seq_len :, :]
        return self.final_layer(hidden_states)

    def sample_with_cfg(self, prompt: dict, cfg_scale: float, sample_func) -> torch.Tensor:
        clap = prompt["clap"]
        t5 = prompt["t5"]
        t5_mask = prompt["t5_mask"]
        if not torch.is_tensor(clap):
            clap = torch.tensor(clap, device=self.device, dtype=self.dtype)
        else:
            clap = clap.to(device=self.device, dtype=self.dtype)
        if not torch.is_tensor(t5):
            t5 = torch.tensor(t5, device=self.device, dtype=self.dtype)
        else:
            t5 = t5.to(device=self.device, dtype=self.dtype)
        if not torch.is_tensor(t5_mask):
            t5_mask = torch.tensor(t5_mask, device=self.device, dtype=torch.bool)
        else:
            t5_mask = t5_mask.to(device=self.device, dtype=torch.bool)

        batch_size = clap.shape[0]
        prompt = {
            "clap": torch.cat([clap, clap], dim=0),
            "t5": torch.cat([t5, t5], dim=0),
            "t5_mask": torch.cat([t5_mask, t5_mask], dim=0),
        }
        prompt_drop_ids = torch.zeros(batch_size * 2, device=self.device, dtype=torch.long)
        prompt_drop_ids[batch_size:] = 1

        batch_size = prompt["clap"].shape[0]
        noise = torch.randn(
            batch_size,
            self.seq_len,
            self.in_channels,
            device=self.device,
            dtype=self.dtype,
        )
        samples = sample_func(
            functools.partial(
                self.forward_with_cfg,
                prompt=prompt,
                cfg_scale=cfg_scale,
                prompt_drop_ids=prompt_drop_ids,
            ),
            noise,
        )
        samples, _ = samples.chunk(2, dim=0)
        return samples

    def forward_with_cfg(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        prompt: dict,
        cfg_scale: float,
        prompt_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        half = hidden_states[: len(hidden_states) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, timesteps, prompt, prompt_drop_ids=prompt_drop_ids)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return torch.cat([guided_eps, guided_eps], dim=0)


#################################################################################
#                                   DiT Configs                                 #
#################################################################################


def DiT_XL(**kwargs) -> DiT:
    return DiT(depth=24, hidden_size=2048, num_heads=16, intermediate_size=5440, **kwargs)


def DiT_Large(**kwargs) -> DiT:
    return DiT(depth=24, hidden_size=1536, num_heads=12, intermediate_size=4096, **kwargs)


def DiT_Medium(**kwargs) -> DiT:
    return DiT(depth=24, hidden_size=1024, num_heads=16, intermediate_size=2688, **kwargs)


def DiT_Base(**kwargs) -> DiT:
    return DiT(depth=12, hidden_size=768, num_heads=12, intermediate_size=2048, **kwargs)


def DiT_B(**kwargs) -> DiT:
    return DiT(depth=24, hidden_size=768, num_heads=12, intermediate_size=2048, **kwargs)


DiT_models = {
    "DiT-XL": DiT_XL,
    "DiT-Large": DiT_Large,
    "DiT-Medium": DiT_Medium,
    "DiT-Base": DiT_Base,
    "DiT-B": DiT_B,
}
