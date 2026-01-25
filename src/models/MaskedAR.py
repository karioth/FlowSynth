import functools

import torch
import torch.nn as nn
from flash_attn.utils.generation import InferenceParams

from .modules.attention import Attention
from .modules.norms import RMSNorm
from .modules.adaln import AdaLNzero, modulate, gate, FinalLayer
from .modules.embeddings import TimestepEmbedder, LabelEmbedder
from .modules.ffn import SwiGLU


class Block(nn.Module):
    """
    Causal transformer block for the backbone.
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


class MaskedARTransformer(nn.Module):
    """
    Masked AR Transformer with a causal backbone and diffusion head.

    Key difference from vanilla Transformer:
    - Same-position readout (position t predicts token t, not t+1)
    - Uses [MASK] token replacement instead of shifting to prevent leakage
    - Loss computed only at masked positions (handled by scheduler)
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
        self.prompt_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

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
        # mask_token is already zeros from initialization

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        x_start: torch.Tensor,
        prompt: torch.Tensor,
        mask: torch.Tensor,
        flat_mask_indices: torch.Tensor,
        batch_mul: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of MaskedARTransformer (optimized).

        Args:
            hidden_states: (num_masked * batch_mul, C) noisy tokens at masked positions
            timesteps: (num_masked * batch_mul,) diffusion timesteps
            x_start: (B, T, C) clean latent tokens (full sequences for backbone)
            prompt: (B,) class prompts
            mask: (B, T) boolean tensor, True = masked position
            flat_mask_indices: (num_masked,) indices into flattened B*T
            batch_mul: batch multiplier for multiple timestep samples
        """
        del kwargs
        bsz, seq_len, _ = x_start.shape

        # 1. Run backbone on full sequences
        conditioning = self.forward_backbone(x_start, prompt, mask)  # (B, T, hidden)

        # 2. Gather conditioning at masked positions
        cond_flat = conditioning.reshape(bsz * seq_len, -1)  # (B*T, hidden)
        cond_masked = cond_flat[flat_mask_indices]  # (num_masked, hidden)

        # 3. Repeat for batch_mul
        cond_masked = cond_masked.repeat_interleave(batch_mul, dim=0)  # (num_masked * batch_mul, hidden)

        # 4. Run diffusion head on packed tokens (seq_len=1 per token)
        return self.forward_diffusion(
            hidden_states.unsqueeze(1),  # (N, 1, C)
            timesteps.unsqueeze(1),       # (N, 1)
            cond_masked.unsqueeze(1),     # (N, 1, hidden)
        ).squeeze(1)  # (N, C)

    def forward_backbone(
        self,
        hidden_states: torch.Tensor,
        prompt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Causal backbone with masked input.

        Unlike vanilla Transformer which shifts tokens, we replace masked positions
        with the [MASK] token. Position t outputs conditioning for predicting token t.
        """
        hidden_states = self.input_embedder(hidden_states)

        # Replace masked positions with [MASK] token
        mask_expanded = mask.unsqueeze(-1).expand_as(hidden_states)
        hidden_states = torch.where(
            mask_expanded,
            self.mask_token.expand_as(hidden_states),
            hidden_states,
        )

        # Prepend prompt embedding (no shift needed - same position readout)
        label_emb = self.prompt_embedder(prompt, self.training)
        hidden_states = torch.cat((label_emb.unsqueeze(1), hidden_states), dim=1)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        # Remove conditioning token
        hidden_states = hidden_states[:, 1:, :]
        return hidden_states

    def forward_recurrent(
        self,
        hidden_states: torch.Tensor | None,
        start_pos: int = 0,
        inference_params: InferenceParams | None = None,
        append_mask: bool = False,
    ) -> torch.Tensor:
        """
        Recurrent forward for inference with KV caching.

        Args:
            hidden_states: Either prompt ids (at start_pos=0), predicted tokens, or None to use mask_token
            start_pos: Current position in sequence
            inference_params: KV cache container
            append_mask: If True, append a [MASK] token to query the next position in the same pass
        """
        start_pos = int(start_pos)
        if start_pos == 0:
            # First position: embed the prompt
            hidden_states = self.prompt_embedder(hidden_states, self.training).unsqueeze(1)
        elif hidden_states is None:
            # Query with mask token (matches training)
            batch_size = inference_params.max_batch_size
            hidden_states = self.mask_token.expand(batch_size, 1, -1)
        else:
            # Subsequent positions: embed the predicted token
            if hidden_states.dim() == 2:
                # It's a raw latent token (B, C) -> (B, 1, C)
                hidden_states = hidden_states.unsqueeze(1)
            hidden_states = self.input_embedder(hidden_states)

        if append_mask:
            # Append [MASK] in hidden space to get the next-position readout in one pass.
            batch_size = hidden_states.shape[0]
            mask_token = self.mask_token.expand(batch_size, 1, -1)
            hidden_states = torch.cat((hidden_states, mask_token), dim=1)

        if inference_params is not None:
            inference_params.seqlen_offset = start_pos

        for block in self.blocks:
            hidden_states = block(hidden_states, inference_params=inference_params)

        return hidden_states

    def forward_diffusion(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Diffusion head for denoising.
        Same as vanilla Transformer.
        """
        bsz, seq_len = timesteps.shape if timesteps.dim() > 1 else (timesteps.shape[0], 1)
        time_emb = self.time_embedder(timesteps.view(-1)).view(bsz, seq_len, -1)
        conditioning = conditioning + time_emb
        hidden_states = self.noisy_input_embedder(hidden_states)

        for block in self.diffusion_blocks:
            hidden_states = block(hidden_states, conditioning)

        hidden_states = self.final_layer(hidden_states, conditioning)
        return hidden_states

    def sample_with_cfg(
        self,
        prompt: torch.Tensor,
        cfg_scale: float,
        sample_func,
    ) -> torch.Tensor:
        """
        AR sampling with classifier-free guidance.

        For each position:
          1) Pass [MASK] through backbone to get conditioning.
          2) Denoise via diffusion head.
          3) Cache the predicted token for the next iteration.

        TODO: Currently, code writes the mask into KV at `i+1` and overwrites it on the next step.
        To avoid that extra cache write, add a only cache first token passed in
        `Attention._update_kv_cache` (e.g., a `cache_write_len` or boolean mask) so only
        the generated token updates KV while the MASK tokens are only used for queries.
        """
        if not torch.is_tensor(prompt):
            prompt = torch.tensor(prompt, device=self.device, dtype=torch.long)
        else:
            prompt = prompt.to(device=self.device, dtype=torch.long)
        # Build [cond, uncond] prompt batch for classifier-free guidance.
        y_null = torch.full_like(prompt, self.prompt_embedder.num_classes, device=self.device)
        prompt = torch.cat([prompt, y_null], dim=0)

        batch_size = prompt.shape[0]
        inference_params = InferenceParams(max_seqlen=self.seq_len + 1, max_batch_size=batch_size)

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
                # First iteration: pass prompts for both conditional and unconditional batches.
                recurrent_input = prompt
            else:
                # Cache the previous predicted token (CFG: duplicate batch).
                recurrent_input = torch.cat([prev_token, prev_token], dim=0)

            # Single pass: write token_i to KV and append [MASK] at i+1; the mask KV
            # is overwritten by the generated token on the next step.
            conditioning = self.forward_recurrent(
                recurrent_input,
                start_pos=i,
                inference_params=inference_params,
                append_mask=True,
            )
            conditioning = conditioning[:, -1:]  # Mask position readout.

            # Denoise
            prev_token = sample_func(
                functools.partial(
                    self.forward_with_cfg,
                    conditioning=conditioning,
                    cfg_scale=cfg_scale,
                ),
                noise,
            )
            prev_token, _ = prev_token.chunk(2, dim=0)
            samples.append(prev_token)

        return torch.cat(samples, dim=1)

    def forward_with_cfg(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.
        """
        half = hidden_states[: len(hidden_states) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward_diffusion(combined, timesteps, conditioning)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return torch.cat([guided_eps, guided_eps], dim=0)


#################################################################################
#                           MaskedAR Transformer Configs                        #
#################################################################################


def MaskedAR_XL(**kwargs) -> MaskedARTransformer:
    return MaskedARTransformer(
        depth=24,
        hidden_size=2048,
        num_heads=16,
        intermediate_size=8192,
        diffusion_intermediate_size=8192,
        **kwargs,
    )


def MaskedAR_Large(**kwargs) -> MaskedARTransformer:
    return MaskedARTransformer(
        depth=24,
        hidden_size=1536,
        num_heads=12,
        intermediate_size=6144,
        diffusion_intermediate_size=6144,
        **kwargs,
    )


def MaskedAR_Medium(**kwargs) -> MaskedARTransformer:
    return MaskedARTransformer(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        intermediate_size=3456,
        diffusion_intermediate_size=3456,
        **kwargs,
    )


def MaskedAR_Base(**kwargs) -> MaskedARTransformer:
    return MaskedARTransformer(
        depth=12,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        diffusion_intermediate_size=3072,
        **kwargs,
    )


def MaskedAR_H(**kwargs) -> MaskedARTransformer:
    return MaskedARTransformer(
        depth=40,
        hidden_size=1280,
        num_heads=20,
        diffusion_depth=12,
        intermediate_size=5120,
        diffusion_intermediate_size=5120,
        **kwargs,
    )


def MaskedAR_L(**kwargs) -> MaskedARTransformer:
    return MaskedARTransformer(
        depth=32,
        hidden_size=1024,
        num_heads=16,
        diffusion_depth=8,
        intermediate_size=4096,
        diffusion_intermediate_size=4096,
        **kwargs,
    )


def MaskedAR_B(**kwargs) -> MaskedARTransformer:
    return MaskedARTransformer(
        depth=24,
        hidden_size=768,
        num_heads=12,
        diffusion_depth=6,
        intermediate_size=3072,
        diffusion_intermediate_size=3072,
        **kwargs,
    )


MaskedAR_models = {
    "MaskedAR-XL": MaskedAR_XL,
    "MaskedAR-Large": MaskedAR_Large,
    "MaskedAR-Medium": MaskedAR_Medium,
    "MaskedAR-Base": MaskedAR_Base,
    "MaskedAR-H": MaskedAR_H,
    "MaskedAR-L": MaskedAR_L,
    "MaskedAR-B": MaskedAR_B,
}
