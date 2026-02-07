# Copyright 2024 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowMatchingSchedulerOutput:
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor`):
            The sample at the previous timestep.
        pred_original_sample (`torch.Tensor`, optional):
            The predicted x0 based on the current sample and model output.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class FlowMatchingBase(nn.Module):
    """
    Rectified flow scheduler with a linear path:
        x_t = (1 - t) * x0 + t * x1, where x1 ~ N(0, I).

    Shared base: common training utilities + basic ODE sampler.
    """

    order = 1

    def __init__(
        self,
        prediction_type: str = "flow",
        t_m: float = 0.0,
        t_s: float = 1.0,
    ):
        super().__init__()
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = None
        self.prediction_type = prediction_type
        self.t_m = t_m
        self.t_s = t_s

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[torch.Tensor] = None,
    ):
        """
        Sets descending timesteps in [1, 0) for Euler integration.
        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps`.")

        if timesteps is None:
            if num_inference_steps is None:
                raise ValueError("`num_inference_steps` must be provided when `timesteps` is None.")
            self.num_inference_steps = num_inference_steps
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)[:-1]
        else:
            timesteps = torch.tensor(timesteps, dtype=torch.float32, device=device)
            self.num_inference_steps = len(timesteps)

        self.timesteps = timesteps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear path interpolation for rectified flow.
        """
        t = timesteps.to(device=original_samples.device, dtype=original_samples.dtype)
        while t.dim() < original_samples.dim():
            t = t.unsqueeze(-1)
        return (1 - t) * original_samples + t * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rectified flow target velocity (x1 - x0).
        """
        return noise - sample

    def sample_timesteps(
        self,
        shape,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Sample timesteps with a logit-normal distribution.

        Reference: https://arxiv.org/pdf/2403.03206
        """
        u = torch.randn(shape, device=device, dtype=dtype) * self.t_s + self.t_m
        return torch.sigmoid(u)

    def configure_sampling(self, **kwargs) -> None:
        """
        Optional hook for sampling-time overrides (no-op by default).
        """
        return None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[FlowMatchingSchedulerOutput, Tuple[torch.Tensor]]:
        """
        Single Euler step: x_{t_prev} = x_t + (t_prev - t) * v_theta(x_t, t).
        """
        t = timestep
        prev_t = self.previous_timestep(t)

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=sample.device, dtype=sample.dtype)
        else:
            t = t.to(device=sample.device, dtype=sample.dtype)

        if not torch.is_tensor(prev_t):
            prev_t = torch.tensor(prev_t, device=sample.device, dtype=sample.dtype)
        else:
            prev_t = prev_t.to(device=sample.device, dtype=sample.dtype)

        dt = prev_t - t
        pred_prev_sample = sample + dt * model_output
        pred_original_sample = sample - t * model_output

        if not return_dict:
            return (pred_prev_sample,)

        return FlowMatchingSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def previous_timestep(self, timestep: Union[torch.Tensor, float, int]) -> torch.Tensor:
        timesteps = self.timesteps
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=timesteps.device, dtype=timesteps.dtype)
        index = (timesteps == timestep).nonzero(as_tuple=True)[0][0]
        if index == timesteps.shape[0] - 1:
            return torch.tensor(0.0, device=timesteps.device, dtype=timesteps.dtype)
        return timesteps[index + 1]

    def forward(self, model_fn, sample: torch.Tensor) -> torch.Tensor:
        """
        ODE sampler entry point. Expects `set_timesteps(...)` to be called first.
        """
        if self.timesteps is None:
            raise RuntimeError("set_timesteps(...) must be called before sampling.")

        timesteps = self.timesteps

        for t in timesteps:
            t_in = t.repeat(sample.shape[0]).to(sample)
            model_output = model_fn(sample, t_in)
            sample = self.step(model_output, t, sample).prev_sample
        return sample


class FlowMatchingSchedulerDiT(FlowMatchingBase):
    def get_losses(self, model, x0_seq, prompt) -> torch.Tensor:
        bsz = x0_seq.shape[0]
        noise = torch.randn_like(x0_seq)
        timesteps = self.sample_timesteps(bsz, device=x0_seq.device, dtype=x0_seq.dtype)

        x_noisy = self.add_noise(x0_seq, noise, timesteps)
        velocity = self.get_velocity(x0_seq, noise, timesteps)
        model_output = model(x_noisy, timesteps, prompt=prompt)
        return F.mse_loss(model_output.float(), velocity.float())


class FlowMatchingSchedulerTransformer(FlowMatchingBase):
    def __init__(self, *args, batch_mul: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_mul = batch_mul

    def get_losses(self, model, x0_seq, prompt) -> torch.Tensor:
        bsz, seq_len, latent_size = x0_seq.shape
        noise = torch.randn(
            (bsz * self.batch_mul * seq_len, latent_size),
            device=x0_seq.device,
            dtype=x0_seq.dtype,
        )
        timesteps = self.sample_timesteps(
            bsz * self.batch_mul * seq_len,
            device=x0_seq.device,
            dtype=x0_seq.dtype,
        )

        x0_rep = x0_seq.repeat_interleave(self.batch_mul, dim=0).reshape(-1, latent_size)
        x_noisy = self.add_noise(x0_rep, noise, timesteps)
        velocity = self.get_velocity(x0_rep, noise, timesteps)

        x_noisy, noise, velocity = [
            x.reshape(bsz * self.batch_mul, seq_len, latent_size)
            for x in (x_noisy, noise, velocity)
        ]
        timesteps = timesteps.reshape(bsz * self.batch_mul, seq_len)
        model_output = model(
            x_noisy,
            timesteps,
            x_start=x0_seq,
            prompt=prompt,
            batch_mul=self.batch_mul,
        )
        return F.mse_loss(model_output.float(), velocity.float())


class FlowMatchingSchedulerARDiff(FlowMatchingBase):
    """
    AR-Diffusion scheduler with AD + FIFO sampling.

    AD builds a per-frame timestep schedule and an update mask; FIFO limits the
    active window for efficiency without changing the schedule.
    """

    def __init__(
        self,
        *args,
        ardiff_step: int = 0,
        base_num_frames: Optional[int] = None,
        t_start: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ardiff_step = ardiff_step
        self.base_num_frames = base_num_frames
        self.t_start = float(t_start)

        self.step_template = None
        self.step_template_full = None
        self.timestep_matrix = None
        self.step_index = None
        self.update_masks = None
        self.valid_intervals = None
        self._schedule_frames = None

    def configure_sampling(
        self,
        ardiff_step: Optional[int] = None,
        base_num_frames: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> None:
        if ardiff_step is not None:
            self.ardiff_step = ardiff_step
        if base_num_frames is not None:
            self.base_num_frames = base_num_frames
        if t_start is not None:
            self.t_start = float(t_start)

    @torch.compile
    def sample_monotone_anchor_times(
        self,
        B: int,
        L: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Continuous monotone timestep sampler (anchor + logit-normal), batched.

        Returns:
          u: (B, L) tensor with 0 <= u[...,0] <= ... <= u[...,L-1] <= 1

        Main idea:
          1) Pick an anchor index k uniformly for each batch element.
          2) Pick an anchor time a from logit-normal.
          3) Sample exactly k values in [0, a] and L-1-k values in [a, 1].
          4) Sort the (L-1) values and insert the anchor at position k.
        """
        # 1) Anchor positions
        k = torch.randint(0, L, (B,), device=device)

        # 2) Anchor values (logit-normal)
        a = torch.sigmoid(torch.randn((B,), device=device, dtype=dtype) * self.t_s + self.t_m)

        Lm1 = L - 1
        j = torch.arange(Lm1, device=device)[None, :]
        prefix = j < k[:, None]

        # 3) Prefix in [0, a], suffix in [a, 1]
        r = torch.rand((B, Lm1), device=device, dtype=dtype)
        z = torch.where(
            prefix,
            r * a[:, None],
            a[:, None] + r * (1.0 - a)[:, None],
        )

        # 4) Sort and insert anchor to make the sequence monotone
        z_sorted, _ = z.sort(dim=1)

        u = torch.empty((B, L), device=device, dtype=dtype)
        pos = torch.arange(L, device=device)[None, :].expand(B, -1)
        mask = pos != k[:, None]
        u[mask] = z_sorted.reshape(-1)
        u.scatter_(1, k[:, None], a[:, None])
        
        return u

    def get_losses(self, model, x0_seq, prompt) -> torch.Tensor:
        bsz, seq_len = x0_seq.shape[:2]
        noise = torch.randn_like(x0_seq)
        t_vec = self.sample_monotone_anchor_times(
            bsz, seq_len, device=x0_seq.device, dtype=x0_seq.dtype
        )

        x_noisy = self.add_noise(x0_seq, noise, t_vec)
        velocity = self.get_velocity(x0_seq, noise, t_vec)
        model_output = model(x_noisy, t_vec, prompt=prompt)
        return F.mse_loss(model_output.float(), velocity.float())

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[torch.Tensor] = None,
    ):
        """
        Build the base step template and reset the cached AD/FIFO schedule.
        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps`.")

        if timesteps is None:
            if num_inference_steps is None:
                raise ValueError("`num_inference_steps` must be provided when `timesteps` is None.")
            self.num_inference_steps = num_inference_steps
            step_template = torch.linspace(
                1.0,
                0.0,
                num_inference_steps + 1,
                device=device,
                dtype=torch.float32,
            )[1:]
        else:
            step_template = torch.tensor(timesteps, dtype=torch.float32, device=device)
            self.num_inference_steps = len(step_template)

        self.step_template = step_template
        self.timesteps = step_template

        self.step_template_full = None
        self.timestep_matrix = None
        self.step_index = None
        self.update_masks = None
        self.valid_intervals = None
        self._schedule_frames = None

    def _build_ad_schedule(
        self,
        total_num_frames: int,
        base_num_frames: int,
        ardiff_step: int,
        step_template: torch.Tensor,
    ):
        """
        Port of AR-Diffusion's generate_timestep_matrix.

        Returns:
          - step_matrix: per-iteration per-frame timesteps
          - step_index: integer step indices per iteration
          - step_update_mask: frames whose step index changed this iteration
          - valid_interval: FIFO window (start, end) per iteration
          - step_template_full: [t_start] + step_template
        """
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template)

        t_start = torch.tensor([self.t_start], device=step_template.device, dtype=step_template.dtype)
        step_template_full = torch.cat([t_start, step_template], dim=0)
        pre_row = torch.zeros(total_num_frames, dtype=torch.long, device=step_template.device)

        while torch.all(pre_row == num_iterations) == False:
            new_row = torch.zeros(total_num_frames, dtype=torch.long, device=step_template.device)
            for i in range(total_num_frames):
                if i == 0 or pre_row[i - 1] == num_iterations:
                    # First frame or previous frame has finished denoising.
                    new_row[i] = pre_row[i] + 1
                else:
                    # Otherwise lag behind by ardiff_step.
                    new_row[i] = new_row[i - 1] - ardiff_step
            new_row = new_row.clamp(0, num_iterations)

            # Update only frames whose step index changed.
            update_mask.append(new_row != pre_row)
            step_index.append(new_row)
            step_matrix.append(step_template_full[new_row])
            pre_row = new_row

        terminal_flag = base_num_frames
        for i in range(0, len(update_mask)):
            if terminal_flag < total_num_frames and update_mask[i][terminal_flag].item():
                # When the next frame becomes active, shift the FIFO window.
                terminal_flag += 1
            valid_interval.append((terminal_flag - base_num_frames, terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)
        return step_matrix, step_index, step_update_mask, valid_interval, step_template_full

    def forward(self, model_fn, sample: torch.Tensor) -> torch.Tensor:
        """
        AD/FIFO sampling loop with per-frame timesteps and update masks.
        """
        if self.step_template is None:
            raise RuntimeError("set_timesteps(...) must be called before sampling.")
        if self.ardiff_step < 0:
            raise ValueError("ardiff_step must be >= 0.")

        total_num_frames = sample.shape[1]
        base_num_frames = self.base_num_frames or total_num_frames
        if base_num_frames > total_num_frames:
            raise ValueError("base_num_frames must be <= total_num_frames.")

        if self.timestep_matrix is None or self._schedule_frames != total_num_frames:
            (
                self.timestep_matrix,
                self.step_index,
                self.update_masks,
                self.valid_intervals,
                self.step_template_full,
            ) = self._build_ad_schedule(
                total_num_frames,
                base_num_frames,
                self.ardiff_step,
                self.step_template,
            )
            self._schedule_frames = total_num_frames

        for i in range(self.timestep_matrix.shape[0]):
            valid_s, valid_e = self.valid_intervals[i]
            x_slice = sample[:, valid_s:valid_e, ...]

            t_cur_f = self.timestep_matrix[i][valid_s:valid_e]
            step_index = self.step_index[i][valid_s:valid_e]
            update_mask = self.update_masks[i][valid_s:valid_e]

            update_mask = update_mask.to(dtype=x_slice.dtype)
            update_mask = update_mask.view(
                (1, update_mask.shape[0]) + (1,) * (x_slice.dim() - 2)
            )

            # Map step indices to actual timesteps; newly-activated frames step from t_start.
            prev_index = torch.clamp(step_index - 1, min=0)
            t_prev_f = self.step_template_full[prev_index]

            t_prev = t_prev_f.unsqueeze(0).expand(x_slice.shape[0], -1)
            model_output = model_fn(x_slice, t_prev)

            # Freeze frames whose timestep did not change.
            model_output = model_output * update_mask + x_slice * (1 - update_mask)

            dt = (t_cur_f - t_prev_f).to(dtype=x_slice.dtype)
            dt = dt.view((1, dt.shape[0]) + (1,) * (x_slice.dim() - 2))
            dt = dt * update_mask
            x_slice = x_slice + dt * model_output

            sample[:, valid_s:valid_e, ...] = x_slice

        return sample


class FlowMatchingSchedulerMaskedAR(FlowMatchingBase):
    """
    Scheduler for MaskedARTransformer.

    Samples per-sequence masking logits from a logistic-normal prior,
    then selects an exact global-K mask with Gumbel-TopK.
    Computes loss only at masked positions.
    """

    def __init__(
        self,
        *args,
        mask_prob: float = 0.75,
        batch_mul: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_prob = mask_prob
        self.batch_mul = batch_mul
        # Per-sequence logistic-normal prior on row logits.
        # Under global exact Top-K, global logit shifts are rank-invariant,
        # so mask_logit_std is the main shape knob.
        p = float(mask_prob)
        if 0.0 < p < 1.0:
            self.mask_logit_mean = math.log(p / (1.0 - p))
        else:
            self.mask_logit_mean = 0.0
        self.mask_logit_std = 1.0

    def _sample_sequence_logits(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")

        mean = float(self.mask_logit_mean)
        std = float(self.mask_logit_std)
        if not math.isfinite(mean):
            raise ValueError(
                f"mask_logit_mean must be finite, got {mean}."
            )
        if (not math.isfinite(std)) or std <= 0.0:
            raise ValueError(
                f"mask_logit_std must be finite and > 0, got {std}."
            )

        return torch.randn(batch_size, device=device, dtype=torch.float32) * std + mean

    def _sample_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}.")

        total_tokens = batch_size * seq_len
        p = float(self.mask_prob)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"mask_prob must be in [0, 1], got {p}.")

        num_masked = round(p * total_tokens)
        if num_masked <= 0:
            raise ValueError(
                "round(mask_prob * total_tokens) must be > 0 for MaskedAR training. "
                f"Got mask_prob={p}, total_tokens={total_tokens}."
            )

        row_logits = self._sample_sequence_logits(batch_size=batch_size, device=device)
        eps = torch.finfo(row_logits.dtype).eps

        uniform = torch.rand((batch_size, seq_len), device=device, dtype=row_logits.dtype)
        uniform = uniform.clamp(min=eps, max=1.0 - eps)
        gumbel = -torch.log(-torch.log(uniform))

        scores = row_logits.unsqueeze(1) + gumbel
        flat_scores = scores.reshape(-1)

        if num_masked >= total_tokens:
            flat_mask_indices = torch.arange(total_tokens, device=device)
        else:
            flat_mask_indices = torch.topk(
                flat_scores,
                k=num_masked,
                largest=True,
                sorted=False,
            ).indices

        mask_flat = torch.zeros(total_tokens, dtype=torch.bool, device=device)
        mask_flat[flat_mask_indices] = True
        mask = mask_flat.view(batch_size, seq_len)

        if int(flat_mask_indices.numel()) != num_masked:
            raise RuntimeError(
                "Mask sampler produced wrong number of masked tokens. "
                f"Expected {num_masked}, got {int(flat_mask_indices.numel())}."
            )

        return mask, flat_mask_indices

    def get_losses(self, model, x0_seq, prompt) -> torch.Tensor:
        bsz, seq_len, latent_size = x0_seq.shape
        total_tokens = bsz * seq_len

        # 1. Sample per-sequence logits and select exact global-K tokens via Gumbel-TopK.
        mask, flat_mask_indices = self._sample_mask(
            batch_size=bsz,
            seq_len=seq_len,
            device=x0_seq.device,
        )
        num_masked = int(flat_mask_indices.numel())

        # 4. Gather x0 at masked positions only
        x0_flat = x0_seq.reshape(total_tokens, latent_size)  # (B*L, C)
        x0_masked = x0_flat[flat_mask_indices]  # (num_masked, C)

        # 5. Create noise and timesteps only for num_masked * batch_mul
        noise = torch.randn(
            num_masked * self.batch_mul,
            latent_size,
            device=x0_seq.device,
            dtype=x0_seq.dtype,
        )
        timesteps = self.sample_timesteps(
            num_masked * self.batch_mul,
            device=x0_seq.device,
            dtype=x0_seq.dtype,
        )

        # 6. Create noisy tokens and velocity targets
        x0_rep = x0_masked.repeat_interleave(self.batch_mul, dim=0)  # (num_masked * batch_mul, C)
        x_noisy = self.add_noise(x0_rep, noise, timesteps)
        velocity = self.get_velocity(x0_rep, noise, timesteps)

        # 7. Forward pass - model returns (num_masked * batch_mul, C)
        model_output = model(
            x_noisy,
            timesteps,
            x_start=x0_seq,
            prompt=prompt,
            mask=mask,
            flat_mask_indices=flat_mask_indices,
            batch_mul=self.batch_mul,
        )

        # 8. Loss on all outputs (all are masked positions)
        return F.mse_loss(model_output.float(), velocity.float())
