import functools
from dataclasses import dataclass

import torch
import torch.nn as nn
from flash_attn.utils.generation import InferenceParams

from .modules.embeddings import PromptEmbedder, TimestepEmbedder
from .modules.layers import FinalLayer, TransformerBlock


@dataclass
class DriftSynthInferenceState:
    inference_params: InferenceParams
    prompt_cached: bool = False
    cached_frames: int = 0


class DriftSynth(nn.Module):
    """
    Simplified autoregressive DiT without AdaLN modulation.
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
                    is_causal=True,
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

    @staticmethod
    def _prefix_length(mask: torch.Tensor, *, name: str) -> int:
        if mask.dim() != 1:
            raise ValueError(f"{name} must be rank-1, got shape {tuple(mask.shape)}.")
        prefix_len = int(mask.sum().item())
        if prefix_len > 0 and not torch.all(mask[:prefix_len]):
            raise ValueError(f"{name} must be a contiguous left prefix.")
        if prefix_len < mask.numel() and torch.any(mask[prefix_len:]):
            raise ValueError(f"{name} must be a contiguous left prefix.")
        return prefix_len

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
                f"DriftSynth forward_recurrent expects tokenwise timesteps with shape (B, T), got {tuple(timesteps.shape)}."
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
        token_states = token_states + self.time_bias_proj(time_emb)

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
        del kwargs
        assert timesteps.dim() == 2, "DriftSynth expects tokenwise timesteps with shape (B, T)"

        hidden_states = self.input_embedder(hidden_states)
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
        )
        hidden_states = hidden_states + self.time_bias_proj(time_emb)
        hidden_states = torch.cat([prompt_seq, hidden_states], dim=1)

        for block in self.blocks:
            hidden_states = block(hidden_states, inference_params=inference_params)

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
        inference_state = DriftSynthInferenceState(
            inference_params=InferenceParams(
                max_seqlen=self.prompt_seq_len + self.seq_len,
                max_batch_size=batch_size,
            )
        )
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
                inference_state=inference_state,
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
        inference_state: DriftSynthInferenceState | None = None,
        frame_update_mask_bool: torch.Tensor | None = None,
        step_index_by_frame: torch.Tensor | None = None,
    ) -> torch.Tensor:
        half = hidden_states[: len(hidden_states) // 2]
        combined = torch.cat([half, half], dim=0)

        use_recurrent = (
            inference_state is not None
            and frame_update_mask_bool is not None
            and step_index_by_frame is not None
        )
        if not use_recurrent:
            eps = self.forward(combined, timesteps, prompt, prompt_drop_ids=prompt_drop_ids)
        else:
            if timesteps.dim() != 2:
                raise ValueError(
                    f"DriftSynth inference expects tokenwise timesteps with shape (B, T), got {tuple(timesteps.shape)}."
                )
            if timesteps.shape != combined.shape[:2]:
                raise ValueError(
                    "timesteps and hidden_states must align on (B, T), "
                    f"got {tuple(timesteps.shape)} and {tuple(combined.shape[:2])}."
                )

            frame_update_mask_bool = frame_update_mask_bool.to(device=timesteps.device, dtype=torch.bool)
            step_index_by_frame = step_index_by_frame.to(device=timesteps.device, dtype=torch.long)
            if frame_update_mask_bool.shape != step_index_by_frame.shape:
                raise ValueError(
                    "frame_update_mask_bool and step_index_by_frame must have the same shape, "
                    f"got {tuple(frame_update_mask_bool.shape)} and {tuple(step_index_by_frame.shape)}."
                )
            if frame_update_mask_bool.numel() != combined.shape[1]:
                raise ValueError(
                    "Schedule frame metadata does not match latent sequence length: "
                    f"{frame_update_mask_bool.numel()} vs {combined.shape[1]}."
                )

            started_len = self._prefix_length(step_index_by_frame > 0, name="AD started mask")
            active_len = int(frame_update_mask_bool.sum().item())
            if active_len > 0:
                active_positions = torch.nonzero(frame_update_mask_bool, as_tuple=False).squeeze(-1)
                active_start = int(active_positions[0].item())
                active_end = int(active_positions[-1].item()) + 1
                if not torch.all(frame_update_mask_bool[active_start:active_end]):
                    raise ValueError("AD update mask must be a contiguous middle block.")
                if active_end > started_len:
                    raise ValueError(
                        "AD active block must be contained in started prefix, "
                        f"got active_end={active_end}, started_len={started_len}."
                    )

            chunk_start = int(inference_state.cached_frames)
            if chunk_start < 0:
                raise ValueError(f"cached_frames must be >= 0, got {chunk_start}.")
            if chunk_start > started_len:
                raise ValueError(
                    "Cached prefix is ahead of started frames: "
                    f"cached_frames={chunk_start}, started_len={started_len}."
                )
            chunk_end = started_len

            if not inference_state.prompt_cached:
                self._cache_prompt_tokens(
                    prompt,
                    inference_params=inference_state.inference_params,
                    prompt_drop_ids=prompt_drop_ids,
                )
                inference_state.prompt_cached = True

            eps = combined.clone()
            if chunk_end > chunk_start:
                eps[:, chunk_start:chunk_end] = self.forward_recurrent(
                    combined[:, chunk_start:chunk_end],
                    timesteps[:, chunk_start:chunk_end],
                    start_pos=chunk_start,
                    inference_params=inference_state.inference_params,
                )

                chunk_timesteps = timesteps[0, chunk_start:chunk_end]
                finished_cacheable = (
                    ~frame_update_mask_bool[chunk_start:chunk_end]
                ) & (chunk_timesteps <= 0)
                finished_prefix_len = self._prefix_length(
                    finished_cacheable,
                    name="AD finished cache mask",
                )
                inference_state.cached_frames = chunk_start + finished_prefix_len

        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return torch.cat([guided_eps, guided_eps], dim=0)


#################################################################################
#                                DriftSynth Configs                             #
#################################################################################


def DriftSynth_XL(**kwargs) -> DriftSynth:
    return DriftSynth(depth=24, hidden_size=2048, num_heads=16, intermediate_size=5440, **kwargs)


def DriftSynth_Large(**kwargs) -> DriftSynth:
    return DriftSynth(depth=24, hidden_size=1536, num_heads=12, intermediate_size=4096, **kwargs)


def DriftSynth_Medium(**kwargs) -> DriftSynth:
    return DriftSynth(depth=32, hidden_size=1024, num_heads=16, intermediate_size=2688, **kwargs)


def DriftSynth_Base(**kwargs) -> DriftSynth:
    return DriftSynth(depth=12, hidden_size=768, num_heads=12, intermediate_size=2048, **kwargs)


def DriftSynth_B(**kwargs) -> DriftSynth:
    return DriftSynth(depth=24, hidden_size=768, num_heads=12, intermediate_size=2048, **kwargs)


def DriftSynth_H(**kwargs) -> DriftSynth:
    return DriftSynth(depth=32, hidden_size=1280, num_heads=20, intermediate_size=5120, **kwargs)


DriftSynth_models = {
    "DriftSynth-XL": DriftSynth_XL,
    "DriftSynth-Large": DriftSynth_Large,
    "DriftSynth-Medium": DriftSynth_Medium,
    "DriftSynth-Base": DriftSynth_Base,
    "DriftSynth-B": DriftSynth_B,
    "DriftSynth-H": DriftSynth_H,
    "DriftSynth-Simple-XL": DriftSynth_XL,
    "DriftSynth-Simple-Large": DriftSynth_Large,
    "DriftSynth-Simple-Medium": DriftSynth_Medium,
    "DriftSynth-Simple-Base": DriftSynth_Base,
    "DriftSynth-Simple-B": DriftSynth_B,
    "DriftSynth-Simple-H": DriftSynth_H,
}
