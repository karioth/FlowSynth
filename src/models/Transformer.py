import functools

import torch
import torch.nn as nn
from flash_attn.utils.generation import InferenceParams

from .modules.attention import Attention
from .modules.norms import RMSNorm
from .modules.adaln import AdaLNzero, modulate, gate, FinalLayer
from .modules.embeddings import TimestepEmbedder, LabelEmbedder, ContinuousEmbedder
from .modules.ffn import SwiGLU


class Block(nn.Module):
    """
    Causal transformer block for the autoregressive conditioning path.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        layer_idx: int,
        is_gated: bool = False,
        rope_theta: float = 10000.0,
        rope_interleaved: bool = False,
        rope_scale_base: float | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            layer_idx=layer_idx,
            is_causal=True,
            is_gated=is_gated,
            rope_theta=rope_theta,
            rope_interleaved=rope_interleaved,
            rope_scale_base=rope_scale_base,
        )
        self.norm2 = RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn(
            self.norm1(hidden_states),
            inference_params=inference_params,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.norm2(hidden_states))
        hidden_states = residual + hidden_states
        
        return hidden_states


class MLPBlock(nn.Module):
    """
    AdaLN-modulated MLP block for the diffusion head.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        self.adaLN_modulation = AdaLNzero(hidden_size=hidden_size, out_mult=3)

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(3, dim=-1)
        
        residual = hidden_states
        hidden_states = self.mlp(
            modulate(self.norm(hidden_states), shift_mlp, scale_mlp)
        )
        hidden_states = residual + gate(hidden_states, gate_mlp)
        
        return hidden_states


class ConditionLayer(nn.Module):
    """
    Final projection for conditioning tokens produced by the AR stack.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_final(hidden_states)
        return self.linear(hidden_states)


class Transformer(nn.Module):
    """
    Transformer with a causal conditioning stack and a diffusion head.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        diffusion_depth: int = 3,
        num_heads: int = 16,
        num_kv_heads: int | None = None,
        intermediate_size: int | None = None,
        diffusion_intermediate_size: int | None = None,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        conditioning_type: str = "class",
        conditioning_dim: int | None = None,
        is_gated: bool = False,
        rope_theta: float = 10000.0,
        rope_interleaved: bool = False,
        rope_scale_base: float | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.conditioning_type = conditioning_type

        if intermediate_size is None:
            intermediate_size = int(hidden_size * 10 / 3 / 64) * 64
        if diffusion_intermediate_size is None:
            diffusion_intermediate_size = intermediate_size

        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=False)
        self.noisy_input_embedder = nn.Linear(in_channels, hidden_size, bias=False)
        self.time_embedder = TimestepEmbedder(hidden_size)

        if conditioning_type == "class":
            self.prompt_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        elif conditioning_type == "continuous":
            assert conditioning_dim is not None, "conditioning_dim required for continuous conditioning"
            self.prompt_embedder = ContinuousEmbedder(conditioning_dim, hidden_size, class_dropout_prob)
        else:
            raise ValueError(f"Unknown conditioning_type: {conditioning_type}")

        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_size=hidden_size,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    intermediate_size=intermediate_size,
                    layer_idx=idx,
                    is_gated=is_gated,
                    rope_theta=rope_theta,
                    rope_interleaved=rope_interleaved,
                    rope_scale_base=rope_scale_base,
                )
                for idx in range(depth)
            ]
        )
        self.diffusion_blocks = nn.ModuleList(
            [
                MLPBlock(hidden_size, diffusion_intermediate_size)
                for _ in range(diffusion_depth)
            ]
        )
        self.condition_layer = ConditionLayer(hidden_size)
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
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        x_start: torch.Tensor,
        prompt: torch.Tensor,
        batch_mul: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of Transformer.
        hidden_states: (B, T, C) tensor of noisy latent tokens
        x_start: (B, T, C) tensor of clean latent tokens
        timesteps: (B, T) or (B,) tensor of diffusion timesteps
        prompt: (B,) tensor of class prompts
        """
        del kwargs
        conditioning = self.forward_parallel(x_start, prompt)
        conditioning = conditioning.repeat_interleave(batch_mul, dim=0)
        return self.forward_diffusion(hidden_states, timesteps, conditioning)

    def forward_parallel(
        self,
        hidden_states: torch.Tensor,
        prompt: torch.Tensor,
        prompt_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.input_embedder(hidden_states)
        label_emb = self.prompt_embedder(
            prompt,
            self.training,
            force_drop_ids=prompt_drop_ids,
        )
        hidden_states = torch.cat((label_emb.unsqueeze(1), hidden_states[:, :-1]), dim=1)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.condition_layer(hidden_states)
        return hidden_states

    def forward_recurrent(
        self,
        hidden_states: torch.Tensor,
        start_pos: int = 0,
        inference_params: InferenceParams | None = None,
        prompt_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        start_pos = int(start_pos)
        if start_pos == 0:
            hidden_states = self.prompt_embedder(
                hidden_states,
                self.training,
                force_drop_ids=prompt_drop_ids,
            ).unsqueeze(1)
        else:
            hidden_states = self.input_embedder(hidden_states)

        if inference_params is not None:
            inference_params.seqlen_offset = start_pos

        for block in self.blocks:
            hidden_states = block(hidden_states, inference_params=inference_params)

        hidden_states = self.condition_layer(hidden_states[:, -1:])
        return hidden_states

    def forward_diffusion(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = timesteps.shape if timesteps.dim() > 1 else (timesteps.shape[0], 1)
        time_emb = self.time_embedder(timesteps.view(-1)).view(bsz, seq_len, -1)
        conditioning = conditioning + time_emb
        hidden_states = self.noisy_input_embedder(hidden_states)

        for block in self.diffusion_blocks:
            hidden_states = block(hidden_states, conditioning)

        hidden_states = self.final_layer(hidden_states, conditioning)
        return hidden_states

    def sample_with_cfg(self, prompt: torch.Tensor, cfg_scale: float, sample_func) -> torch.Tensor:
        # Build [cond, uncond] prompt batch for classifier-free guidance.
        prompt_drop_ids = None
        if self.conditioning_type == "class":
            if not torch.is_tensor(prompt):
                prompt = torch.tensor(prompt, device=self.device, dtype=torch.long)
            else:
                prompt = prompt.to(device=self.device, dtype=torch.long)
            y_null = torch.full_like(prompt, self.prompt_embedder.num_classes, device=self.device)
            prompt = torch.cat([prompt, y_null], dim=0)
        else:
            # continuous conditioning: prompt is [B, D] embedding
            if not torch.is_tensor(prompt):
                prompt = torch.tensor(prompt, device=self.device, dtype=self.dtype)
            else:
                prompt = prompt.to(device=self.device, dtype=self.dtype)
            prompt = torch.cat([prompt, prompt], dim=0)
            prompt_drop_ids = torch.zeros(prompt.shape[0], device=self.device, dtype=torch.long)
            prompt_drop_ids[prompt.shape[0] // 2:] = 1

        batch_size = prompt.shape[0]
        inference_params = InferenceParams(max_seqlen=self.seq_len, max_batch_size=batch_size)
        prev_token = prompt
        samples = []
        for i in range(self.seq_len):
            noise = torch.randn(
                batch_size,
                1,
                self.in_channels,
                device=self.device,
                dtype=self.dtype,
            )
            recurrent_input = torch.cat([prev_token, prev_token], dim=0) if i != 0 else prev_token
            conditioning = self.forward_recurrent(
                recurrent_input,
                start_pos=i,
                inference_params=inference_params,
                prompt_drop_ids=prompt_drop_ids,
            )
            prev_token = sample_func(
                functools.partial(self.forward_with_cfg, conditioning=conditioning, cfg_scale=cfg_scale),
                noise,
            )
            prev_token, _ = prev_token.chunk(2, dim=0)
            samples.append(prev_token)
        return torch.cat(samples, 1)

    def forward_with_cfg(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        Forward pass of Transformer, batching unconditional and conditional paths for CFG.
        """
        half = hidden_states[: len(hidden_states) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward_diffusion(combined, timesteps, conditioning)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return torch.cat([guided_eps, guided_eps], dim=0)


#################################################################################
#                               Transformer Configs                             #
#################################################################################


def Transformer_XL(**kwargs) -> Transformer:
    return Transformer(
        depth=24,
        hidden_size=2048,
        num_heads=16,
        intermediate_size=8192,
        diffusion_intermediate_size=8192,
        **kwargs,
    )

def Transformer_Large(**kwargs) -> Transformer:
    return Transformer(
        depth=24,
        hidden_size=1536,
        num_heads=12,
        intermediate_size=6144,
        diffusion_intermediate_size=6144,
        **kwargs,
    )

def Transformer_Medium(**kwargs) -> Transformer:
    return Transformer(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        intermediate_size=3456,
        diffusion_intermediate_size=3456,
        **kwargs,
    )

def Transformer_Base(**kwargs) -> Transformer:
    return Transformer(
        depth=12,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        diffusion_intermediate_size=3072,
        **kwargs,
    )

def Transformer_H(**kwargs) -> Transformer:
    return Transformer(
        depth=40,
        hidden_size=1280,
        num_heads=20,
        diffusion_depth=12,
        intermediate_size=5120,
        diffusion_intermediate_size=5120,
        **kwargs,
    )

def Transformer_L(**kwargs) -> Transformer:
    return Transformer(
        depth=32,
        hidden_size=1024,
        num_heads=16,
        diffusion_depth=8,
        intermediate_size=4096,
        diffusion_intermediate_size=4096,
        **kwargs,
    )

def Transformer_B(**kwargs) -> Transformer:
    return Transformer(
        depth=24,
        hidden_size=768,
        num_heads=12,
        diffusion_depth=6,
        intermediate_size=3072,
        diffusion_intermediate_size=3072,
        **kwargs,
    )


Transformer_models = {
    "Transformer-XL": Transformer_XL,
    "Transformer-Large": Transformer_Large,
    "Transformer-Medium": Transformer_Medium,
    "Transformer-Base": Transformer_Base,
    "Transformer-H": Transformer_H,
    "Transformer-L": Transformer_L,
    "Transformer-B": Transformer_B,
}
