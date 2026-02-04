import math
from typing import Optional

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        time_factor: float = 1000.0,
    ) -> None:
        super().__init__()
        if frequency_embedding_size % 2 != 0:
            raise ValueError("frequency_embedding_size must be even.")

        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.time_factor = time_factor

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        # cache freqs to avoid recomputing them every time
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs, persistent=False)  # (half,) fp32
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match DiT-style init for the timestep MLP.
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (N,) or (N,1) tensor, int or float, can be fractional.
        returns: (N, frequency_embedding_size) in t.dtype
        """
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t = t.reshape(-1)

        t = t * self.time_factor # scale 0-1 to original ddpm discrete scales as in Flux
        args = t.float()[:, None] * self.freqs[None, :]              # (N, half) fp32
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (N, 2*half)

        return emb.to(dtype=t.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t)
        return self.mlp(t_freq)


class PromptEmbedder(nn.Module):
    """
    Embeds pooled CLAP + T5 hidden states into a fixed-length prompt sequence.

    Output: [B, prompt_seq_len, hidden_size] where prompt_seq_len = 1 + max_t5_tokens.
    Expects prompt_data to include a boolean t5_mask for valid tokens.
    """

    def __init__(
        self,
        clap_dim: int,
        t5_dim: int,
        hidden_size: int,
        prompt_seq_len: int = 69,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.clap_dim = clap_dim
        self.t5_dim = t5_dim
        self.hidden_size = hidden_size
        self.prompt_seq_len = prompt_seq_len
        self.max_t5_tokens = prompt_seq_len - 1
        self.dropout_prob = dropout_prob

        self.clap_proj = nn.Linear(clap_dim, hidden_size, bias=False)
        self.t5_proj = nn.Linear(t5_dim, hidden_size, bias=False)
        self.null_embeddings = nn.Parameter(torch.empty(prompt_seq_len, hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.clap_proj.weight, std=0.02)
        nn.init.normal_(self.t5_proj.weight, std=0.02)
        nn.init.normal_(self.null_embeddings, std=0.02)

    def forward(
        self,
        prompt_data: dict,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        clap = prompt_data["clap"]
        t5 = prompt_data["t5"]
        t5_mask = prompt_data["t5_mask"]

        batch_size = clap.shape[0]
        device = clap.device

        clap_emb = self.clap_proj(clap)
        t5_emb = self.t5_proj(t5)

        t5_mask = t5_mask.to(device=device, dtype=torch.bool)

        null_t5 = self.null_embeddings[1:].unsqueeze(0).expand(batch_size, -1, -1)
        t5_emb = torch.where(t5_mask.unsqueeze(-1), t5_emb, null_t5)

        prompt_seq = torch.cat([clap_emb.unsqueeze(1), t5_emb], dim=1)

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            if force_drop_ids is None:
                drop_ids = torch.rand(batch_size, device=device) < self.dropout_prob
            else:
                drop_ids = force_drop_ids == 1

            null_seq = self.null_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            prompt_seq = torch.where(drop_ids.view(-1, 1, 1), null_seq, prompt_seq)

        if torch.is_autocast_enabled():
            prompt_seq = prompt_seq.to(torch.get_autocast_gpu_dtype())
        return prompt_seq


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size = 2
    hidden_size = 768

        embedder = PromptEmbedder(
        clap_dim=512,
        t5_dim=1024,
        hidden_size=hidden_size,
        prompt_seq_len=69,
        dropout_prob=0.1,
    )

    t5_mask = torch.zeros(batch_size, 68, dtype=torch.bool)
    t5_mask[0, :5] = True
    t5_mask[1, :68] = True
    prompt_data = {
        "clap": torch.randn(batch_size, 512),
        "t5": torch.randn(batch_size, 68, 1024),
        "t5_mask": t5_mask,
    }

    out = embedder(prompt_data, train=False)
    assert out.shape == (batch_size, 69, hidden_size), f"Unexpected output shape: {out.shape}"

    valid_len = int(prompt_data["t5_mask"][0].sum().item())
    padded_start = 1 + valid_len
    expected_padding = embedder.null_embeddings[padded_start:]
    actual_padding = out[0, padded_start:, :]
    assert torch.allclose(
        actual_padding,
        expected_padding,
        atol=0,
        rtol=0,
    ), "Padding does not match null embeddings"

    force_drop_ids = torch.ones(batch_size, dtype=torch.long)
    out_dropped = embedder(prompt_data, train=False, force_drop_ids=force_drop_ids)
    expected_full = embedder.null_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    assert torch.allclose(
        out_dropped,
        expected_full,
        atol=0,
        rtol=0,
    ), "CFG drop does not match null embeddings"

    embedder.zero_grad(set_to_none=True)
    out_train = embedder(prompt_data, train=True)
    out_train.sum().backward()
    assert embedder.clap_proj.weight.grad is not None, "Missing clap_proj grads"
    assert embedder.t5_proj.weight.grad is not None, "Missing t5_proj grads"
    assert embedder.null_embeddings.grad is not None, "Missing null embedding grads"
    assert not torch.isnan(embedder.clap_proj.weight.grad).any(), "NaN grads in clap_proj"
    assert not torch.isnan(embedder.t5_proj.weight.grad).any(), "NaN grads in t5_proj"
    assert not torch.isnan(embedder.null_embeddings.grad).any(), "NaN grads in null_embeddings"

    prompt_single = {
        "clap": prompt_data["clap"][:1],
        "t5": prompt_data["t5"][:1],
        "t5_mask": prompt_data["t5_mask"][:1],
    }
    trials = 1000
    drops = 0
    null_seq = embedder.null_embeddings.unsqueeze(0)
    for _ in range(trials):
        out = embedder(prompt_single, train=True)
        if torch.allclose(out, null_seq, atol=0, rtol=0):
            drops += 1
    drop_rate = drops / trials
    assert 0.05 < drop_rate < 0.15, f"Unexpected drop rate: {drop_rate:.3f}"

    print("PASS: sequence prompt embedder shapes OK")
    print("PASS: padding uses null embeddings")
    print("PASS: CFG drop uses full null sequence")
    print("PASS: gradient flow OK")
    print("PASS: dropout rate OK")
