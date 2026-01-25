import numpy as np
import torch


class SequenceDiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            zeros = torch.zeros_like(self.mean)
            self.std = zeros
            self.var = zeros

    def sample(self) -> torch.Tensor:
        noise = torch.randn_like(self.mean)
        return self.mean + self.std * noise

    def kl(self, other=None, dims=None) -> torch.Tensor:
        if self.deterministic:
            return torch.tensor(0.0, device=self.parameters.device)

        if dims is None:
            dims = tuple(range(1, self.mean.dim()))

        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=dims,
            )

        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=dims,
        )

    def nll(self, sample: torch.Tensor, dims=None) -> torch.Tensor:
        if self.deterministic:
            return torch.tensor(0.0, device=self.parameters.device)
        if dims is None:
            dims = tuple(range(1, self.mean.dim()))
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean
