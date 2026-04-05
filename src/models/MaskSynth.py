import functools

import torch
import torch.nn as nn
from flash_attn.utils.generation import InferenceParams

from .modules.adaln import AdaLNFinalLayer
from .modules.embeddings import TimestepEmbedder, PromptEmbedder
from .modules.layers import AdaLNMLPBlock, FinalLayer, TransformerBlock


class MaskSynth(nn.Module):
    """
    MaskSynth with a causal backbone and diffusion head.

    Key difference from ShiftSynth:
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

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    intermediate_size=intermediate_size,
                    layer_idx=idx,
                    is_causal=True,
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
                AdaLNMLPBlock(hidden_size, diffusion_intermediate_size)
                for _ in range(diffusion_depth)
            ]
        )
        self.condition_layer = FinalLayer(hidden_size, hidden_size)
        self.final_layer = AdaLNFinalLayer(hidden_size, self.out_channels)

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
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        x_start: torch.Tensor,
        prompt: dict,
        mask: torch.Tensor,
        flat_mask_indices: torch.Tensor,
        batch_mul: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of MaskSynth (optimized).

        Args:
            hidden_states: (num_masked * batch_mul, C) noisy tokens at masked positions
            timesteps: (num_masked * batch_mul,) diffusion timesteps
            x_start: (B, T, C) clean latent tokens (full sequences for backbone)
        prompt: dict with clap/t5 embeddings
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
        prompt: dict,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Causal backbone with masked input.

        Unlike ShiftSynth, which shifts tokens, we replace masked positions
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
        prompt_seq = self.prompt_embedder(prompt, self.training)
        hidden_states = torch.cat((prompt_seq, hidden_states), dim=1)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states[:, self.prompt_seq_len :, :]
        return self.condition_layer(hidden_states)

    def forward_recurrent(
        self,
        hidden_states: dict | torch.Tensor | None,
        start_pos: int = 0,
        inference_params: InferenceParams | None = None,
        append_mask: bool = False,
        prompt_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Recurrent forward for inference with KV caching.

        Args:
            hidden_states: Either prompt ids/embeddings (at start_pos=0), predicted tokens, or None to use mask_token
            start_pos: Current position in sequence
            inference_params: KV cache container
            append_mask: If True, append a [MASK] token to query the next position in the same pass
        """
        start_pos = int(start_pos)
        if start_pos == 0:
            # First position: embed the prompt
            hidden_states = self.prompt_embedder(
                hidden_states,
                self.training,
                force_drop_ids=prompt_drop_ids,
            )
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

        return self.condition_layer(hidden_states)

    def forward_diffusion(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Diffusion head for denoising.
        Same as ShiftSynth.
        """
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
        Standard constant CFG scale.
        """
        del token_idx, seq_len
        return float(cfg_scale)

    def sample_with_cfg(
        self,
        prompt: dict,
        cfg_scale: float,
        sample_func,
    ) -> torch.Tensor:
        """
        AR sampling with classifier-free guidance.

        For each position:
          1) Pass [MASK] through backbone to get conditioning.
          2) Denoise via diffusion head.
          3) Cache the predicted token for the next iteration.
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
            max_seqlen=self.seq_len + self.prompt_seq_len,
            max_batch_size=batch_size,
        )

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
                recurrent_input = prompt_input
            else:
                # Cache the previous predicted token (CFG: duplicate batch).
                recurrent_input = torch.cat([prev_token, prev_token], dim=0)

            # Single pass: write token_i to KV and append [MASK] at i+1; the mask KV
            # is overwritten by the generated token on the next step.
            conditioning = self.forward_recurrent(
                recurrent_input,
                start_pos=0 if i == 0 else self.prompt_seq_len + i - 1,
                inference_params=inference_params,
                append_mask=True,
                prompt_drop_ids=prompt_drop_ids,
            )
            conditioning = conditioning[:, -1:]  # Mask position readout.
            guidance_scale = self.cfg_scale_at_token(
                token_idx=i,
                seq_len=self.seq_len,
                cfg_scale=cfg_scale,
            )

            # Denoise
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

        return torch.cat(samples, dim=1)

    def forward_with_cfg(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.
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
#                               MaskSynth Configs                               #
#################################################################################


def MaskSynth_XL(**kwargs) -> MaskSynth:
    return MaskSynth(
        depth=24,
        hidden_size=2048,
        num_heads=16,
        intermediate_size=8192,
        diffusion_intermediate_size=8192,
        **kwargs,
    )


def MaskSynth_Large(**kwargs) -> MaskSynth:
    return MaskSynth(
        depth=24,
        hidden_size=1536,
        num_heads=12,
        intermediate_size=6144,
        diffusion_intermediate_size=6144,
        **kwargs,
    )


def MaskSynth_Medium(**kwargs) -> MaskSynth:
    return MaskSynth(
        depth=32,
        hidden_size=1024,
        num_heads=16,
        diffusion_depth=4,
        intermediate_size=2688,
        diffusion_intermediate_size=2688,
        **kwargs,
    )


def MaskSynth_Base(**kwargs) -> MaskSynth:
    return MaskSynth(
        depth=12,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        diffusion_intermediate_size=3072,
        **kwargs,
    )


def MaskSynth_H(**kwargs) -> MaskSynth:
    return MaskSynth(
        depth=40,
        hidden_size=1280,
        num_heads=20,
        diffusion_depth=12,
        intermediate_size=5120,
        diffusion_intermediate_size=5120,
        **kwargs,
    )


def MaskSynth_L(**kwargs) -> MaskSynth:
    return MaskSynth(
        depth=32,
        hidden_size=1024,
        num_heads=16,
        diffusion_depth=8,
        intermediate_size=4096,
        diffusion_intermediate_size=4096,
        **kwargs,
    )


def MaskSynth_B(**kwargs) -> MaskSynth:
    return MaskSynth(
        depth=24,
        hidden_size=768,
        num_heads=12,
        diffusion_depth=3,
        intermediate_size=2048,
        diffusion_intermediate_size=2048,
        **kwargs,
    )


def MaskSynth_H2(**kwargs) -> MaskSynth:
    return MaskSynth(
        depth=32,
        hidden_size=1280,
        num_heads=20,
        diffusion_depth=4,
        intermediate_size=5120,
        diffusion_intermediate_size=5120,
        **kwargs,
    )


MaskSynth_models = {
    "MaskSynth-XL": MaskSynth_XL,
    "MaskSynth-Large": MaskSynth_Large,
    "MaskSynth-Medium": MaskSynth_Medium,
    "MaskSynth-Base": MaskSynth_Base,
    "MaskSynth-H": MaskSynth_H,
    "MaskSynth-H2": MaskSynth_H2,
    "MaskSynth-L": MaskSynth_L,
    "MaskSynth-B": MaskSynth_B,
}


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required for MaskSynth flash-attn test")
    batch_size = 2
    seq_len = 3
    in_channels = 6
    hidden_size = 32
    num_heads = 4
    prompt_seq_len = 5
    clap_dim = 8
    t5_dim = 12
    max_t5_tokens = prompt_seq_len - 1

    model = MaskSynth(
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

    hidden_states = torch.randn(batch_size, seq_len, in_channels, device=device, dtype=torch.bfloat16)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[0, 0] = True
    with autocast:
        conditioning = model.forward_backbone(hidden_states, prompt, mask)
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

    def _record_forward_recurrent(hidden_states, start_pos=0, inference_params=None, append_mask=False, prompt_drop_ids=None):
        start_positions.append(int(start_pos))
        return original_forward_recurrent(
            hidden_states,
            start_pos=start_pos,
            inference_params=inference_params,
            append_mask=append_mask,
            prompt_drop_ids=prompt_drop_ids,
        )

    model.forward_recurrent = _record_forward_recurrent

    def sample_func(model_fn, noise):
        ts = torch.zeros(noise.shape[0], device=noise.device, dtype=torch.float32)
        return model_fn(noise, ts)

    with autocast:
        samples = model.sample_with_cfg(prompt, cfg_scale=1.0, sample_func=sample_func)
    expected_positions = [0] + [model.prompt_seq_len + i - 1 for i in range(1, seq_len)]
    assert start_positions == expected_positions, f"Unexpected start_pos trace: {start_positions}"
    assert samples.shape == (batch_size, seq_len, in_channels), f"Unexpected sample shape: {samples.shape}"

    print("PASS: MaskSynth conditioning shapes OK")
    print("PASS: MaskSynth start_pos offsets OK")
    print("PASS: MaskSynth sample_with_cfg OK")
