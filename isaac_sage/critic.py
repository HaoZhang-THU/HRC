from __future__ import annotations

import torch
from torch import Tensor, nn


class TeamContextCritic(nn.Module):
    def __init__(self, obs_dim_per: int, num_agents: int, context_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.obs_dim_per, self.num_agents = obs_dim_per, num_agents
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim_per, context_dim), nn.Tanh(), nn.Linear(context_dim, context_dim))
        self.value = nn.Sequential(
            nn.Linear(obs_dim_per * num_agents + context_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def team_context(self, obs: Tensor) -> Tensor:
        dim = self.obs_dim_per
        context = None
        for i in range(self.num_agents):
            feature = self.encoder(obs[:, i * dim:(i + 1) * dim])
            context = feature if context is None else context + feature
        return context / float(self.num_agents)

    def forward(self, obs: Tensor) -> Tensor:
        return self.value(torch.cat((obs, self.team_context(obs)), dim=-1)).squeeze(-1)


class JointCritic(nn.Module):
    def __init__(self, obs_dim_per: int, num_agents: int, hidden_sizes=(256, 256, 128)):
        super().__init__()
        self.obs_dim_per, self.num_agents = obs_dim_per, num_agents
        dim = obs_dim_per * num_agents
        gain = nn.init.calculate_gain("relu")
        layers: list[nn.Module] = [nn.LayerNorm(dim)]
        for size in hidden_sizes:
            linear = nn.Linear(dim, size)
            nn.init.orthogonal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.ReLU(), nn.LayerNorm(size)]
            dim = size
        out = nn.Linear(dim, 1)
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
        debias = self.debiasing_term.clamp(min=self.epsilon)
        mean, mean_sq = self.running_mean / debias, self.running_mean_sq / debias
        return mean, (mean_sq - mean ** 2).clamp(min=1e-2)

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        weight = self.beta
        self.running_mean.mul_(weight).add_(x.mean() * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_((x ** 2).mean() * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 - weight)

    def normalize(self, x: Tensor) -> Tensor:
        mean, var = self.mean_var()
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x: Tensor) -> Tensor:
        mean, var = self.mean_var()
        return x * torch.sqrt(var) + mean


def huber_loss(error: Tensor, delta: float) -> Tensor:
    small = (error.abs() <= delta).float()
    return small * error ** 2 / 2 + (1 - small) * delta * (error.abs() - delta / 2)
