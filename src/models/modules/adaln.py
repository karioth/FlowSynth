import torch
import torch.nn as nn

from .norms import RMSNorm


def modulate(hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return hidden_states * (1 + scale) + shift


def gate(hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return hidden_states * gate


class AdaLNzero(nn.Module):
    """
    AdaLNzero modulation MLP that outputs a configurable multiple of hidden_size.
    """

    def __init__(self, hidden_size: int, out_mult: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.proj = nn.Linear(hidden_size, out_mult * hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Zero-init to keep adaLN modulation initially inactive.
        nn.init.constant_(self.proj.weight, 0)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.proj(self.act(conditioning))


class AdaLNFinalLayer(nn.Module):
    """
    The final layer of AdaLNzero modules.
    """

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = AdaLNzero(hidden_size, out_mult=2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keep final projection zeroed at init to match DiT behavior.
        nn.init.constant_(self.out_proj.weight, 0)
        self.adaLN_modulation.reset_parameters()

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        hidden_states = modulate(self.norm_final(hidden_states), shift, scale)
        return self.out_proj(hidden_states)
