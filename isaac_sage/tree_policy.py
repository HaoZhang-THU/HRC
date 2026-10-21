from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ATANH_CLAMP = 0.995
LOG_STD_MIN, LOG_STD_MAX = -3.0, 0.0


class HardObliqueTree(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, depth: int = 4):
        super().__init__()
        self.obs_dim, self.act_dim, self.depth = obs_dim, act_dim, depth
        self.num_inner = 2 ** depth - 1
        self.num_leaves = 2 ** depth
        self.split_w = nn.Parameter(torch.randn(self.num_inner, obs_dim) / math.sqrt(obs_dim))
        self.split_b = nn.Parameter(torch.zeros(self.num_inner))
        self.leaf_K = nn.Parameter(0.01 * torch.randn(self.num_leaves, act_dim, obs_dim))
        self.leaf_beta = nn.Parameter(0.01 * torch.randn(self.num_leaves, act_dim))

    def unit_splits(self) -> Tensor:
        return F.normalize(self.split_w, dim=-1, eps=1e-8)

    def gates(self, obs: Tensor, tau: float, hard: bool = True) -> Tensor:
        scores = obs @ self.unit_splits().t() - self.split_b
        soft = torch.sigmoid(scores / tau)
        if not hard:
            return soft
        return (scores > 0).to(obs.dtype) + soft - soft.detach()

    def st_gates(self, obs: Tensor, tau: float) -> Tensor:
        return self.gates(obs, tau, hard=True)

    def path_indicator(self, gates: Tensor) -> Tensor:
        batch = gates.shape[0]
        probs = torch.ones(batch, 1, dtype=gates.dtype, device=gates.device)
        for level in range(self.depth):
            start = 2 ** level - 1
            gate = gates[:, start:start + 2 ** level]
            probs = torch.stack((probs * (1.0 - gate), probs * gate), dim=-1).reshape(batch, -1)
        return probs

    def leaf_raw(self, obs: Tensor) -> Tensor:
        return torch.einsum("bd,lad->bla", obs, self.leaf_K) + self.leaf_beta

    def forward(self, obs: Tensor, tau: float, hard: bool = True) -> tuple[Tensor, dict]:
        gates = self.gates(obs, tau, hard)
        probs = self.path_indicator(gates)
        raw_action = (probs.unsqueeze(-1) * self.leaf_raw(obs)).sum(dim=1)
        return torch.tanh(raw_action), {"gates": gates, "leaf_probs": probs, "leaf": probs.argmax(dim=1)}

    def regularizer(self, info: dict, eps: float = 1e-8) -> dict[str, Tensor]:
        weights = self.unit_splits().abs()
        weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)
        feature_entropy = -(weights * torch.log(weights + eps)).sum(dim=-1).mean()
        gates = info["gates"]
        routing_entropy = -(gates * torch.log(gates + eps) + (1.0 - gates) * torch.log(1.0 - gates + eps)).mean()
        usage = info["leaf_probs"].mean(dim=0)
        leaf_balance = (usage * torch.log(usage * self.num_leaves + eps)).sum()
        return {"feature_entropy": feature_entropy, "routing_entropy": routing_entropy,
                "leaf_balance": leaf_balance}

    @torch.no_grad()
    def route(self, obs: Tensor) -> Tensor:
        scores = obs @ self.unit_splits().t()
        node = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        for _ in range(self.depth):
            score = scores.gather(1, node.unsqueeze(1)).squeeze(1)
            right = (score > self.split_b[node]).long()
            node = 2 * node + 1 + right
        return node - self.num_inner

    @torch.no_grad()
    def nominal(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        leaf = self.route(obs)
        raw_action = torch.bmm(self.leaf_K[leaf], obs.unsqueeze(-1)).squeeze(-1) + self.leaf_beta[leaf]
        return torch.tanh(raw_action), leaf

    @torch.no_grad()
    def export_rules(self, feature_names: Optional[tuple[str, ...]] = None, min_coef: float = 0.05) -> dict:
        names = feature_names or tuple(f"x{i}" for i in range(self.obs_dim))
        splits = self.unit_splits()
        nodes = []
        for k in range(self.num_inner):
            coeffs = {names[i]: round(float(splits[k, i]), 4) for i in range(self.obs_dim)
                      if abs(float(splits[k, i])) >= min_coef}
            nodes.append({"node": k, "w": coeffs, "b": round(float(self.split_b[k]), 4),
                          "right_if": "w^T o > b"})
        leaves = [{"leaf": leaf,
                   "K": [[round(float(v), 4) for v in row] for row in self.leaf_K[leaf].tolist()],
                   "beta": [round(float(v), 4) for v in self.leaf_beta[leaf].tolist()]}
                  for leaf in range(self.num_leaves)]
        return {"depth": self.depth, "nodes": nodes, "leaves": leaves,
                "action": "tanh(K_l o + beta_l), normalised to [-1, 1]"}


class TanhNormal:
    def __init__(self, pre_mean: Tensor, std: Tensor):
        self.base = torch.distributions.Normal(pre_mean, std)

    def sample(self) -> Tensor:
        return torch.tanh(self.base.sample())

    def log_prob(self, action: Tensor) -> Tensor:
        action = action.clamp(-ATANH_CLAMP, ATANH_CLAMP)
        pre_action = torch.atanh(action)
        return self.base.log_prob(pre_action).sum(-1) - torch.log(1.0 - action * action + 1e-6).sum(-1)

    def entropy(self) -> Tensor:
        return self.base.entropy().sum(-1)


class CarrierActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, depth: int = 4, log_std_init: float = -0.5):
        super().__init__()
        self.tree = HardObliqueTree(obs_dim, act_dim, depth)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def std(self) -> Tensor:
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def proposal(self, mean: Tensor) -> TanhNormal:
        pre_mean = torch.atanh(mean.clamp(-ATANH_CLAMP, ATANH_CLAMP))
        return TanhNormal(pre_mean, self.std().expand_as(pre_mean))


class TeamActor(nn.Module):
    def __init__(self, obs_dim_per: int, act_dim_per: int, num_agents: int, depth: int = 4,
                 log_std_init: float = -0.5):
        super().__init__()
        self.obs_dim_per, self.act_dim_per, self.num_agents = obs_dim_per, act_dim_per, num_agents
        self.actors = nn.ModuleList(
            [CarrierActor(obs_dim_per, act_dim_per, depth, log_std_init) for _ in range(num_agents)])

    def agent_obs(self, obs: Tensor, i: int) -> Tensor:
        dim = self.obs_dim_per
        return obs[:, i * dim:(i + 1) * dim]

    def agent_action(self, actions: Tensor, i: int) -> Tensor:
        dim = self.act_dim_per
        return actions[:, i * dim:(i + 1) * dim]

    def nominal_st(self, obs: Tensor, tau: float, detach_except: Optional[int] = None,
                   hard: bool = True, return_reg: bool = False):
        out, regs = [], []
        for i, actor in enumerate(self.actors):
            action, info = actor.tree(self.agent_obs(obs, i), tau, hard)
            if detach_except is not None and i != detach_except:
                action = action.detach()
            out.append(action)
            if return_reg:
                regs.append(sum(actor.tree.regularizer(info).values()))
        actions = torch.cat(out, dim=-1)
        return (actions, regs) if return_reg else actions

    def sample_proposals(self, proposal_mean: Tensor) -> tuple[Tensor, Tensor]:
        acts, log_probs = [], []
        for i, actor in enumerate(self.actors):
            dist = actor.proposal(self.agent_action(proposal_mean, i))
            action = dist.sample()
            acts.append(action)
            log_probs.append(dist.log_prob(action))
        return torch.cat(acts, -1), torch.stack(log_probs, -1)

    def agent_log_prob(self, i: int, proposal_mean: Tensor, proposals: Tensor) -> Tensor:
        return self.actors[i].proposal(self.agent_action(proposal_mean, i)).log_prob(self.agent_action(proposals, i))

    def entropy(self, proposal_mean: Tensor) -> Tensor:
        return sum(self.actors[i].proposal(self.agent_action(proposal_mean, i)).entropy().mean()
                   for i in range(self.num_agents)) / self.num_agents

    @torch.no_grad()
    def nominal(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        acts, leaves = [], []
        for i, actor in enumerate(self.actors):
            action, leaf = actor.tree.nominal(self.agent_obs(obs, i))
            acts.append(action)
            leaves.append(leaf)
        return torch.cat(acts, -1), torch.stack(leaves, -1)

    def block_parameters(self, i: int) -> list[nn.Parameter]:
        return list(self.actors[i].parameters())
