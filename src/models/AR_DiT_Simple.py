import functools
from dataclasses import dataclass

import torch
import torch.nn as nn
from flash_attn.utils.generation import InferenceParams

from .AR_DiT import AR_DiT, ARDiTInferenceState
from .modules.attention import Attention
from .modules.norms import RMSNorm
from .modules.embeddings import TimestepEmbedder, PromptEmbedder
from .modules.ffn import SwiGLU


class SimpleBlock(nn.Module):
    """
    Pre-norm transformer block without AdaLN modulation.
    Identity-init: output projections of attn and mlp are zero-initialized
    so each block acts as identity at init.
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
        rope_interleaved: bool = True,
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

        # Zero-init output projections so residual branch outputs 0 at init
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


class SimpleFinalLayer(nn.Module):
    """
    Final layer without AdaLN modulation. Zero-init output projection
    so the model outputs 0 at initialization.
    """

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.out_proj = nn.Linear(hidden_size, output_size, bias=False)
        nn.init.constant_(self.out_proj.weight, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.norm_final(hidden_states))


class AR_DiT_Simple(AR_DiT):
    """
    Simplified AR_DiT without AdaLN modulation.
    Time conditioning is an additive bias applied before the transformer blocks,
    similar to positional encodings in early transformers.
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
        # Skip AR_DiT.__init__, go to nn.Module directly
        nn.Module.__init__(self)

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
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

        # Additive time bias projection (replaces AdaLNzero 6-param modulation)
        self.time_bias_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

        self.blocks = nn.ModuleList(
            [
                SimpleBlock(
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
        self.final_layer = SimpleFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Zero-init time bias so initially no time info flows
        nn.init.constant_(self.time_bias_proj[1].weight, 0)

    def _cache_prompt_tokens(
        self,
        prompt: dict,
        inference_params: InferenceParams,
        prompt_drop_ids: torch.Tensor | None = None,
    ) -> None:
        prompt_seq = self.prompt_embedder(
            prompt,
            self.training,
            force_drop_ids=prompt_drop_ids,
        )
        inference_params.seqlen_offset = 0
        for block in self.blocks:
            prompt_seq = block(prompt_seq, inference_params=inference_params)

    def forward_recurrent(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        start_pos: int,
        inference_params: InferenceParams,
    ) -> torch.Tensor:
        if timesteps.dim() != 2:
            raise ValueError(
                f"AR_DiT_Simple forward_recurrent expects tokenwise timesteps with shape (B, T), got {tuple(timesteps.shape)}."
            )
        if hidden_states.shape[:2] != timesteps.shape:
            raise ValueError(
                "hidden_states and timesteps must align on (B, T), "
                f"got {tuple(hidden_states.shape[:2])} and {tuple(timesteps.shape)}."
            )

        start_pos = int(start_pos)
        if start_pos < 0:
            raise ValueError(f"start_pos must be >= 0, got {start_pos}.")
        if start_pos + hidden_states.size(1) > self.seq_len:
            raise ValueError(
                "Chunk exceeds latent sequence length: "
                f"start_pos={start_pos}, chunk={hidden_states.size(1)}, seq_len={self.seq_len}."
            )

        token_states = self.input_embedder(hidden_states)
        timesteps = timesteps.contiguous()
        time_emb = self.time_embedder(timesteps.view(-1)).view(
            token_states.size(0),
            token_states.size(1),
            -1,
        )
        time_bias = self.time_bias_proj(time_emb)
        token_states = token_states + time_bias

        inference_params.seqlen_offset = self.prompt_seq_len + start_pos
        for block in self.blocks:
            token_states = block(token_states, inference_params=inference_params)
        return self.final_layer(token_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        prompt: dict,
        inference_params=None,
        *,
        prompt_drop_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of AR_DiT_Simple.
        hidden_states: (B, T, C) tensor of noisy latent tokens
        timesteps: (B, T) tensor of diffusion timesteps
        prompt: dict with clap/t5 embeddings
        """
        del kwargs
        assert timesteps.dim() == 2, "AR_DiT_Simple expects tokenwise timesteps with shape (B, T)"

        hidden_states = self.input_embedder(hidden_states)  # (B, T, D)
        prompt_seq = self.prompt_embedder(
            prompt,
            self.training,
            force_drop_ids=prompt_drop_ids,
        )
        timesteps = timesteps.contiguous()
        time_emb = self.time_embedder(timesteps.view(-1)).view(
            hidden_states.size(0),
            hidden_states.size(1),
            -1,
        )  # (B, T, D)

        # Additive time bias on latent tokens only
        time_bias = self.time_bias_proj(time_emb)  # (B, T, D)
        hidden_states = hidden_states + time_bias

        # Concatenate prompt + latent tokens
        hidden_states = torch.cat([prompt_seq, hidden_states], dim=1)  # (B, T+P, D)

        for block in self.blocks:
            hidden_states = block(hidden_states, inference_params=inference_params)

        # Remove conditioning tokens before the final layer
        hidden_states = hidden_states[:, self.prompt_seq_len:, :]
        hidden_states = self.final_layer(hidden_states)
        return hidden_states


#################################################################################
#                           AR-DiT-Simple Configs                               #
#################################################################################

def AR_DiT_Simple_XL(**kwargs) -> AR_DiT_Simple:
    return AR_DiT_Simple(depth=24, hidden_size=2048, num_heads=16, intermediate_size=5440, **kwargs)

def AR_DiT_Simple_Large(**kwargs) -> AR_DiT_Simple:
    return AR_DiT_Simple(depth=24, hidden_size=1536, num_heads=12, intermediate_size=4096, **kwargs)

def AR_DiT_Simple_Medium(**kwargs) -> AR_DiT_Simple:
    return AR_DiT_Simple(depth=32, hidden_size=1024, num_heads=16, intermediate_size=2688, **kwargs)

def AR_DiT_Simple_Base(**kwargs) -> AR_DiT_Simple:
    return AR_DiT_Simple(depth=12, hidden_size=768, num_heads=12, intermediate_size=2048, **kwargs)

def AR_DiT_Simple_B(**kwargs) -> AR_DiT_Simple:
    return AR_DiT_Simple(depth=24, hidden_size=768, num_heads=12, intermediate_size=2048, **kwargs)

def AR_DiT_Simple_H(**kwargs) -> AR_DiT_Simple:
    return AR_DiT_Simple(depth=32, hidden_size=1280, num_heads=20, intermediate_size=5120, **kwargs)

AR_DiT_Simple_models = {
    "AR-DiT-Simple-XL": AR_DiT_Simple_XL,
    "AR-DiT-Simple-Large": AR_DiT_Simple_Large,
    "AR-DiT-Simple-Medium": AR_DiT_Simple_Medium,
    "AR-DiT-Simple-Base": AR_DiT_Simple_Base,
    "AR-DiT-Simple-B": AR_DiT_Simple_B,
    "AR-DiT-Simple-H": AR_DiT_Simple_H,
}


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required for AR_DiT_Simple flash-attn test")
    batch_size = 2
    seq_len = 4
    in_channels = 6
    hidden_size = 32
    num_heads = 4
    prompt_seq_len = 5
    clap_dim = 8
    t5_dim = 12
    max_t5_tokens = prompt_seq_len - 1

    model = AR_DiT_Simple(
        seq_len=seq_len,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=2,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        intermediate_size=64,
        clap_dim=clap_dim,
        t5_dim=t5_dim,
        prompt_seq_len=prompt_seq_len,
    ).to(device, dtype=torch.bfloat16)
    model.eval()

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    t5_mask = torch.zeros(batch_size, max_t5_tokens, dtype=torch.bool, device=device)
    t5_mask[0, :2] = True
    t5_mask[1, :max_t5_tokens] = True
    prompt = {
        "clap": torch.randn(batch_size, clap_dim, device=device, dtype=torch.bfloat16),
        "t5": torch.randn(batch_size, max_t5_tokens, t5_dim, device=device, dtype=torch.bfloat16),
        "t5_mask": t5_mask,
    }

    hidden_states = torch.randn(batch_size, seq_len, in_channels, device=device, dtype=torch.bfloat16)
    timesteps = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
    with autocast:
        out = model(hidden_states, timesteps, prompt)
    assert out.shape == (batch_size, seq_len, in_channels), f"Unexpected output shape: {out.shape}"

    # Verify near-zero output at init
    max_val = out.abs().max().item()
    print(f"Max output magnitude at init: {max_val:.6f}")
    assert max_val < 1e-3, f"Output should be near-zero at init, got max={max_val}"

    def sample_func(model_fn, noise):
        ts = torch.zeros(noise.shape[0], noise.shape[1], device=noise.device, dtype=torch.float32)
        return model_fn(noise, ts)

    with autocast:
        samples = model.sample_with_cfg(prompt, cfg_scale=1.0, sample_func=sample_func)
    assert samples.shape == (batch_size, seq_len, in_channels), f"Unexpected sample shape: {samples.shape}"

    print("PASS: AR_DiT_Simple forward shapes OK")
    print("PASS: AR_DiT_Simple sample_with_cfg OK")
    print("PASS: AR_DiT_Simple zero-init verified")
