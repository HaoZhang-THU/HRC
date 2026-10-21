from __future__ import annotations

import torch
from torch import Tensor, nn


class TeamContextCritic(nn.Module):
    def __init__(self, obs_dim_per: int, n_agents: int, context_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.obs_dim_per, self.n_agents = obs_dim_per, n_agents
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim_per, context_dim), nn.Tanh(), nn.Linear(context_dim, context_dim))
        self.value = nn.Sequential(
            nn.Linear(obs_dim_per * n_agents + context_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def team_context(self, obs: Tensor) -> Tensor:
        d = self.obs_dim_per
        z = None
        for i in range(self.n_agents):
            e = self.encoder(obs[:, i * d:(i + 1) * d])
            z = e if z is None else z + e
        return z / float(self.n_agents)

    def forward(self, obs: Tensor) -> Tensor:
        return self.value(torch.cat((obs, self.team_context(obs)), dim=-1)).squeeze(-1)


class JointCritic(nn.Module):
    def __init__(self, obs_dim_per: int, n_agents: int, hidden_sizes=(256, 256, 128)):
        super().__init__()
        self.obs_dim_per, self.n_agents = obs_dim_per, n_agents
        d = obs_dim_per * n_agents
        gain = nn.init.calculate_gain("relu")
        layers: list[nn.Module] = [nn.LayerNorm(d)]
        for h in hidden_sizes:
            lin = nn.Linear(d, h)
            nn.init.orthogonal_(lin.weight, gain=gain)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.ReLU(), nn.LayerNorm(h)]
            d = h
        out = nn.Linear(d, 1)
        nn.init.orthogonal_(out.weight, gain=1.0)
        nn.init.zeros_(out.bias)
        self.value = nn.Sequential(*layers, out)

    def forward(self, obs: Tensor) -> Tensor:
        return self.value(obs).squeeze(-1)


class ValueNorm(nn.Module):
    def __init__(self, beta: float = 0.99999, epsilon: float = 1e-5):
        super().__init__()
        self.beta, self.epsilon = beta, epsilon
        self.register_buffer("running_mean", torch.zeros(()))
        self.register_buffer("running_mean_sq", torch.zeros(()))
        self.register_buffer("debiasing_term", torch.zeros(()))

    def mean_var(self) -> tuple[Tensor, Tensor]:
        d = self.debiasing_term.clamp(min=self.epsilon)
        mean, mean_sq = self.running_mean / d, self.running_mean_sq / d
        return mean, (mean_sq - mean ** 2).clamp(min=1e-2)

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        w = self.beta
        self.running_mean.mul_(w).add_(x.mean() * (1.0 - w))
        self.running_mean_sq.mul_(w).add_((x ** 2).mean() * (1.0 - w))
        self.debiasing_term.mul_(w).add_(1.0 - w)

    def normalize(self, x: Tensor) -> Tensor:
        mean, var = self.mean_var()
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x: Tensor) -> Tensor:
        mean, var = self.mean_var()
        return x * torch.sqrt(var) + mean


def huber_loss(e: Tensor, d: float) -> Tensor:
    a = (e.abs() <= d).float()
    return a * e ** 2 / 2 + (1 - a) * d * (e.abs() - d / 2)
