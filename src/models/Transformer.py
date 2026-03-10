import functools

import torch
import torch.nn as nn
from flash_attn.utils.generation import InferenceParams

from .modules.attention import Attention
from .modules.norms import RMSNorm
from .modules.adaln import AdaLNzero, modulate, gate, FinalLayer
from .modules.embeddings import TimestepEmbedder, PromptEmbedder
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
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        if intermediate_size is None:
            intermediate_size = int(hidden_size * 10 / 3 / 64) * 64
        if diffusion_intermediate_size is None:
            diffusion_intermediate_size = intermediate_size

        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=False)
        self.noisy_input_embedder = nn.Linear(in_channels, hidden_size, bias=False)
        self.time_embedder = TimestepEmbedder(hidden_size)

        self.prompt_embedder = PromptEmbedder(
            clap_dim=clap_dim,
            t5_dim=t5_dim,
            hidden_size=hidden_size,
            prompt_seq_len=prompt_seq_len,
            dropout_prob=prompt_dropout_prob,
        )
        self.prompt_seq_len = prompt_seq_len

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
        prompt: dict[str, torch.Tensor],
        batch_mul: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of Transformer.
        hidden_states: (B, T, C) tensor of noisy latent tokens
        x_start: (B, T, C) tensor of clean latent tokens
        timesteps: (B, T) or (B,) tensor of diffusion timesteps
        prompt: dict with clap/t5 embeddings
        """
        del kwargs
        conditioning = self.forward_parallel(x_start, prompt)
        conditioning = conditioning.repeat_interleave(batch_mul, dim=0)
        return self.forward_diffusion(hidden_states, timesteps, conditioning)

    def forward_parallel(
        self,
        hidden_states: torch.Tensor,
        prompt: dict[str, torch.Tensor],
        prompt_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.input_embedder(hidden_states)
        prompt_seq = self.prompt_embedder(
            prompt,
            self.training,
            force_drop_ids=prompt_drop_ids,
        )
        hidden_states = torch.cat((prompt_seq, hidden_states[:, :-1]), dim=1)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.condition_layer(hidden_states)
        return hidden_states[:, self.prompt_seq_len - 1 :, :]

    def forward_recurrent(
        self,
        hidden_states: dict[str, torch.Tensor] | torch.Tensor,
        start_pos: int = 0,
        inference_params: InferenceParams | None = None,
        prompt_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        start_pos = int(start_pos)
        if start_pos == 0:
            if not isinstance(hidden_states, dict):
                raise ValueError("Prompt data is required when start_pos is 0.")
            prompt_data = hidden_states
            token_states = self.prompt_embedder(
                prompt_data,
                self.training,
                force_drop_ids=prompt_drop_ids,
            )
        else:
            if not torch.is_tensor(hidden_states):
                raise ValueError("Latent tokens are required when start_pos is not 0.")
            token_states = self.input_embedder(hidden_states)

        if inference_params is not None:
            inference_params.seqlen_offset = start_pos

        for block in self.blocks:
            token_states = block(token_states, inference_params=inference_params)

        token_states = self.condition_layer(token_states[:, -1:])
        return token_states

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

    @staticmethod
    def cfg_scale_at_token(token_idx: int, seq_len: int, cfg_scale: float) -> float:
        """
        Position-dependent standard CFG scale s_i for token index i.

        Currently kept constant to match audiontp constant CFG behavior.
        """
        del token_idx, seq_len
        return float(cfg_scale)
        # Scheduled behavior disabled for now:
        # if seq_len <= 1:
        #     return float(cfg_scale)
        # pos = float(token_idx) / float(seq_len - 1)
        # return 1.0 + (float(cfg_scale) - 1.0) * (1.0 - pos)

    def sample_with_cfg(self, prompt: dict[str, torch.Tensor], cfg_scale: float, sample_func) -> torch.Tensor:
        """
        Sample with CFG using a constant standard guidance scale.
        """
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
        prompt_input = prompt

        batch_size = prompt["clap"].shape[0]
        inference_params = InferenceParams(
            max_seqlen=self.seq_len + self.prompt_seq_len - 1,
            max_batch_size=batch_size,
        )
        prev_token = None
        samples = []
        for i in range(self.seq_len):
            noise = torch.randn(
                batch_size,
                1,
                self.in_channels,
                device=self.device,
                dtype=self.dtype,
            )
            if i == 0:
                recurrent_input = prompt_input
            else:
                if prev_token is None:
                    raise RuntimeError("Expected prev_token to be set before recurrent steps.")
                recurrent_input = torch.cat([prev_token, prev_token], dim=0)
            conditioning = self.forward_recurrent(
                recurrent_input,
                start_pos=0 if i == 0 else self.prompt_seq_len - 1 + i,
                inference_params=inference_params,
                prompt_drop_ids=prompt_drop_ids,
            )
            guidance_scale = self.cfg_scale_at_token(token_idx=i, seq_len=self.seq_len, cfg_scale=cfg_scale)
            prev_token = sample_func(
                functools.partial(
                    self.forward_with_cfg,
                    conditioning=conditioning,
                    guidance_scale=guidance_scale,
                ),
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
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Forward pass of Transformer, batching unconditional and conditional paths for CFG.
        guidance_scale follows standard CFG form:
            eps = eps_u + guidance_scale * (eps_c - eps_u).
        """
        half = hidden_states[: len(hidden_states) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward_diffusion(combined, timesteps, conditioning)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
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
        intermediate_size=2688,
        diffusion_intermediate_size=2688,
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
        diffusion_depth=3,
        intermediate_size=2048, ## should be 2048 to match expanding mlps
        diffusion_intermediate_size=2048,
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


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required for Transformer flash-attn test")
    batch_size = 2
    seq_len = 3
    in_channels = 6
    hidden_size = 32
    num_heads = 4
    prompt_seq_len = 5
    clap_dim = 8
    t5_dim = 12
    max_t5_tokens = prompt_seq_len - 1

    model = Transformer(
        seq_len=seq_len,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=2,
        diffusion_depth=1,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        intermediate_size=64,
        diffusion_intermediate_size=64,
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

    x_start = torch.randn(batch_size, seq_len, in_channels, device=device, dtype=torch.bfloat16)
    with autocast:
        conditioning = model.forward_parallel(x_start, prompt)
    assert conditioning.shape == (batch_size, seq_len, hidden_size), f"Unexpected conditioning shape: {conditioning.shape}"

    # Constant CFG checks.
    cfg_scale = 3.0
    cfg_first = model.cfg_scale_at_token(token_idx=0, seq_len=seq_len, cfg_scale=cfg_scale)
    cfg_last = model.cfg_scale_at_token(token_idx=seq_len - 1, seq_len=seq_len, cfg_scale=cfg_scale)
    cfg_single = model.cfg_scale_at_token(token_idx=0, seq_len=1, cfg_scale=cfg_scale)
    assert cfg_first == cfg_scale, f"Expected cfg_first={cfg_scale}, got {cfg_first}"
    assert cfg_last == cfg_scale, f"Expected cfg_last={cfg_scale}, got {cfg_last}"
    assert cfg_single == cfg_scale, f"Expected cfg_single={cfg_scale}, got {cfg_single}"

    start_positions = []
    original_forward_recurrent = model.forward_recurrent

    def _record_forward_recurrent(hidden_states, start_pos=0, inference_params=None, prompt_drop_ids=None):
        start_positions.append(int(start_pos))
        return original_forward_recurrent(
            hidden_states,
            start_pos=start_pos,
            inference_params=inference_params,
            prompt_drop_ids=prompt_drop_ids,
        )

    model.forward_recurrent = _record_forward_recurrent

    def sample_func(model_fn, noise):
        ts = torch.zeros(noise.shape[0], device=noise.device, dtype=torch.float32)
        return model_fn(noise, ts)

    with autocast:
        samples = model.sample_with_cfg(prompt, cfg_scale=1.0, sample_func=sample_func)
    expected_positions = [0] + [model.prompt_seq_len - 1 + i for i in range(1, seq_len)]
    assert start_positions == expected_positions, f"Unexpected start_pos trace: {start_positions}"
    assert samples.shape == (batch_size, seq_len, in_channels), f"Unexpected sample shape: {samples.shape}"

    print("PASS: Transformer conditioning shapes OK")
    print("PASS: Transformer start_pos offsets OK")
    print("PASS: Transformer sample_with_cfg OK")
