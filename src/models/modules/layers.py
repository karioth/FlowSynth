import torch
import torch.nn as nn

from .adaln import AdaLNzero, gate, modulate
from .attention import Attention
from .ffn import SwiGLU
from .norms import RMSNorm


class TransformerBlock(nn.Module):
    """
    Shared pre-norm transformer block for the plain, non-AdaLN stacks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        layer_idx: int,
        is_causal: bool,
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
            is_causal=is_causal,
            is_gated=is_gated,
            rope_theta=rope_theta,
            rope_interleaved=rope_interleaved,
            rope_scale_base=rope_scale_base,
        )
        self.norm2 = RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        nn.init.constant_(self.attn.out_proj.weight, 0)
        nn.init.constant_(self.mlp.down_proj.weight, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params=None,
    ) -> torch.Tensor:
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


class FinalLayer(nn.Module):
    """
    Shared final projection for the plain, non-AdaLN stacks.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.out_proj = nn.Linear(hidden_size, output_size, bias=False)
        nn.init.constant_(self.out_proj.weight, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.norm_final(hidden_states))


class AdaLNMLPBlock(nn.Module):
    """
    Shared AdaLN-modulated MLP block for diffusion heads.
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
