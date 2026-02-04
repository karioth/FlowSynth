import torch


def sample_posterior(posterior_params: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    mean, logvar = torch.chunk(posterior_params, 2, dim=-1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    if deterministic:
        return mean
    std = torch.exp(0.5 * logvar)
    return mean + std * torch.randn_like(mean)
