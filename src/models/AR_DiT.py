import functools

import torch
import torch.nn as nn

from .modules.attention import Attention
from .modules.norms import RMSNorm
from .modules.adaln import AdaLNzero, modulate, gate, FinalLayer
from .modules.embeddings import TimestepEmbedder, PromptEmbedder
from .modules.ffn import SwiGLU


class AR_DiTBlock(nn.Module):
    """
    An autoregressive DiT block with adaLN-Zero conditioning.
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
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))
        self.scale_shift_table._no_weight_decay = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_modulation: torch.Tensor,
        inference_params=None,
    ) -> torch.Tensor:

        # time_modulation: (B, T, 6D) bf16 under autocast
        bias = self.scale_shift_table.reshape(1, 1, -1)  # (1,1,6D) bf16
        biases = time_modulation + bias                                                  # (B,T,6D) bf16
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = biases.to(dtype=hidden_states.dtype).chunk(6, dim=-1)

        residual = hidden_states
        hidden_states = self.attn(
            modulate(self.norm1(hidden_states), shift_msa, scale_msa),
            inference_params=inference_params,
        )
        hidden_states = residual + gate(hidden_states, gate_msa)

        residual = hidden_states
        hidden_states = self.mlp(
            modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        )
        hidden_states = residual + gate(hidden_states, gate_mlp)
        
        return hidden_states


class AR_DiT(nn.Module):
    """
    Diffusion model with an autoregressive Transformer backbone.
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

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.seq_len = seq_len

        if intermediate_size is None:
            intermediate_size = int(hidden_size * 7 / 3 / 64) * 64 # roughly 4x ratio in regular MLP but 2.6ish for swiglu

        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=False)
        self.time_embedder = TimestepEmbedder(hidden_size)

        self.prompt_embedder = PromptEmbedder(
            clap_dim=clap_dim,
            t5_dim=t5_dim,
            hidden_size=hidden_size,
            prompt_seq_len=prompt_seq_len,
            dropout_prob=prompt_dropout_prob,
        )
        self.prompt_seq_len = prompt_seq_len

        self.time_modulation = AdaLNzero(hidden_size=hidden_size, out_mult=6)

        self.blocks = nn.ModuleList(
            [
                AR_DiTBlock(
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
        prompt: dict,
        inference_params=None,
        *,
        prompt_drop_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of AR_DiT.
        hidden_states: (B, T, C) tensor of noisy latent tokens
        timesteps: (B, T) tensor of diffusion timesteps
        prompt: dict with clap/t5 embeddings
        """
        del kwargs
        assert timesteps.dim() == 2, "AR_DiT expects tokenwise timesteps with shape (B, T)"

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
        time_modulation = self.time_modulation(time_emb)
        t_prompt_mod = torch.zeros(
            hidden_states.size(0),
            self.prompt_seq_len,
            time_modulation.size(-1),
            device=time_modulation.device,
            dtype=time_modulation.dtype,
        )
        time_modulation = torch.cat([t_prompt_mod, time_modulation], dim=1)  # (B, T+P, 6D)
        hidden_states = torch.cat([prompt_seq, hidden_states], dim=1)  # (B, T+P, D)

        for block in self.blocks:
            hidden_states = block(hidden_states, time_modulation, inference_params=inference_params)

        # Remove conditioning token before the final layer.
        hidden_states = hidden_states[:, self.prompt_seq_len :, :]
        hidden_states = self.final_layer(hidden_states, time_emb)
        return hidden_states

    def sample_with_cfg(self, prompt: dict, cfg_scale: float, sample_func) -> torch.Tensor:
        # Build [cond, uncond] prompt batch for classifier-free guidance.
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
        """
        Forward pass of AR_DiT, batching unconditional and conditional paths for CFG.
        """
        half = hidden_states[: len(hidden_states) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, timesteps, prompt, prompt_drop_ids=prompt_drop_ids)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return torch.cat([guided_eps, guided_eps], dim=0)


#################################################################################
#                                  AR-DiT Configs                               #
#################################################################################

def AR_DiT_XL(**kwargs) -> AR_DiT:
    return AR_DiT(depth=24, hidden_size=2048, num_heads=16, intermediate_size=5440, **kwargs)

def AR_DiT_Large(**kwargs) -> AR_DiT:
    return AR_DiT(depth=24, hidden_size=1536, num_heads=12, intermediate_size=4096, **kwargs)

def AR_DiT_Medium(**kwargs) -> AR_DiT:
    return AR_DiT(depth=24, hidden_size=1024, num_heads=16, intermediate_size=2560, **kwargs)

def AR_DiT_Base(**kwargs) -> AR_DiT:
    return AR_DiT(depth=12, hidden_size=768, num_heads=12, intermediate_size=2048, **kwargs)

AR_DiT_models = {
    "AR-DiT-XL": AR_DiT_XL,
    "AR-DiT-Large": AR_DiT_Large,
    "AR-DiT-Medium": AR_DiT_Medium,
    "AR-DiT-Base": AR_DiT_Base,
}


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required for AR_DiT flash-attn test")
    batch_size = 2
    seq_len = 4
    in_channels = 6
    hidden_size = 32
    num_heads = 4
    prompt_seq_len = 5
    clap_dim = 8
    t5_dim = 12
    max_t5_tokens = prompt_seq_len - 1

    model = AR_DiT(
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

    def sample_func(model_fn, noise):
        ts = torch.zeros(noise.shape[0], noise.shape[1], device=noise.device, dtype=torch.float32)
        return model_fn(noise, ts)

    with autocast:
        samples = model.sample_with_cfg(prompt, cfg_scale=1.0, sample_func=sample_func)
    assert samples.shape == (batch_size, seq_len, in_channels), f"Unexpected sample shape: {samples.shape}"

    print("PASS: AR_DiT continuous prompt shapes OK")
    print("PASS: AR_DiT sample_with_cfg OK")
