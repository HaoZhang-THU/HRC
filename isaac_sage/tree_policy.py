"""Hard oblique decision-tree nominal actors and bounded proposal policies
(paper Sec. III-B and Eq. 30-31).

Nominal actor  mu_theta  (Eq. 23-25)
    internal node k:  g_k(o) = 1[w_k^T o > b_k],  ||w_k||_2 = 1
    leaf l:           a_raw_l(o) = K_l o + beta_l
    a_nom(o) = c_a + s_a * tanh(a_raw_{l*(o)}(o)),  here c_a = 0, s_a = 1
    (actions are normalised to [-1, 1]^3; the environment adapter applies the
    physical scale).

Routing during training  (Eq. 26 and the tau_n schedule of Alg. 1)
    soft phase: g_k = sigmoid((w_k^T o - b_k) / tau), the nominal action is the
        path-probability mixture of the leaf controllers;
    hard phase (final fraction of training): straight-through gates -- the
        forward pass uses the hard gate, the backward pass the same sigmoid
        surrogate, so the rollout nominal action follows the deployed
        hard-routing semantics.
    The path indicator is the product of g_k on right branches and 1 - g_k on
    left branches in both phases.

Structure regulariser  R(theta_i)
    split-weight entropy (sparse, near-axis-aligned splits) + gate entropy
    (decisive routing) + leaf-usage KL to uniform (no dead leaves); weighted by
    lambda_tree and subtracted from every surrogate of the carrier's block.

Proposal policy  pi_theta_i  (Eq. 30)
    tanh-squashed Gaussian centred on the (SAIL-adjusted) proposal centre
    a_bar_i: fixed support [-1, 1]^3 and positive density in its interior, so
    the PPO ratio at stored proposals is well defined.  The exploration scale
    (log_std) belongs to the actor parameter block theta_i.
"""
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
        """Soft gates sigmoid((w^T o - b)/tau), or g_k^ST (Eq. 26) when ``hard``:
        hard value forward, the same sigmoid backward."""
        v = obs @ self.unit_splits().t() - self.split_b
        soft = torch.sigmoid(v / tau)
        if not hard:
            return soft
        return (v > 0).to(obs.dtype) + soft - soft.detach()

    def st_gates(self, obs: Tensor, tau: float) -> Tensor:
        return self.gates(obs, tau, hard=True)

    def path_indicator(self, gates: Tensor) -> Tensor:
        """[B, L] product of g^ST (right) / 1 - g^ST (left) along each root-to-leaf path."""
        B = gates.shape[0]
        ind = torch.ones(B, 1, dtype=gates.dtype, device=gates.device)
        for level in range(self.depth):
            start = 2 ** level - 1
            g = gates[:, start:start + 2 ** level]
            ind = torch.stack((ind * (1.0 - g), ind * g), dim=-1).reshape(B, -1)
        return ind

    def leaf_raw(self, obs: Tensor) -> Tensor:
        """a_raw_l(o) for every leaf: [B, L, da] (Eq. 24)."""
        return torch.einsum("bd,lad->bla", obs, self.leaf_K) + self.leaf_beta

    def forward(self, obs: Tensor, tau: float, hard: bool = True) -> tuple[Tensor, dict]:
        """Training-time nominal action in [-1, 1]^da: soft path mixture, or the
        hard-forward ST action mu^ST_{theta,tau}(o) when ``hard`` (one-hot forward)."""
        g = self.gates(obs, tau, hard)
        ind = self.path_indicator(g)
        a_raw = (ind.unsqueeze(-1) * self.leaf_raw(obs)).sum(dim=1)
        return torch.tanh(a_raw), {"gates": g, "leaf_probs": ind, "leaf": ind.argmax(dim=1)}

    def regularizer(self, info: dict, eps: float = 1e-8) -> dict[str, Tensor]:
        """R(theta_i) terms on the gates / path probabilities of one forward pass."""
        w = self.unit_splits().abs()
        w = w / (w.sum(dim=-1, keepdim=True) + eps)
        feature_entropy = -(w * torch.log(w + eps)).sum(dim=-1).mean()
        g = info["gates"]
        routing_entropy = -(g * torch.log(g + eps) + (1.0 - g) * torch.log(1.0 - g + eps)).mean()
        usage = info["leaf_probs"].mean(dim=0)
        leaf_balance = (usage * torch.log(usage * self.num_leaves + eps)).sum()
        return {"feature_entropy": feature_entropy, "routing_entropy": routing_entropy,
                "leaf_balance": leaf_balance}

    @torch.no_grad()
    def route(self, obs: Tensor) -> Tensor:
        """Deployed hard routing (Eq. 23): leaf index per observation."""
        v = obs @ self.unit_splits().t()
        node = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        for _ in range(self.depth):
            val = v.gather(1, node.unsqueeze(1)).squeeze(1)
            right = (val > self.split_b[node]).long()
            node = 2 * node + 1 + right
        return node - self.num_inner

    @torch.no_grad()
    def nominal(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Deployed deterministic nominal action (Eq. 25) and leaf index."""
        leaf = self.route(obs)
        a_raw = torch.bmm(self.leaf_K[leaf], obs.unsqueeze(-1)).squeeze(-1) + self.leaf_beta[leaf]
        return torch.tanh(a_raw), leaf

    @torch.no_grad()
    def export_rules(self, feature_names: Optional[tuple[str, ...]] = None, min_coef: float = 0.05) -> dict:
        names = feature_names or tuple(f"x{i}" for i in range(self.obs_dim))
        W = self.unit_splits()
        nodes = []
        for k in range(self.num_inner):
            coeffs = {names[i]: round(float(W[k, i]), 4) for i in range(self.obs_dim)
                      if abs(float(W[k, i])) >= min_coef}
            nodes.append({"node": k, "w": coeffs, "b": round(float(self.split_b[k]), 4),
                          "right_if": "w^T o > b"})
        leaves = [{"leaf": l,
                   "K": [[round(float(v), 4) for v in row] for row in self.leaf_K[l].tolist()],
                   "beta": [round(float(v), 4) for v in self.leaf_beta[l].tolist()]}
                  for l in range(self.num_leaves)]
        return {"depth": self.depth, "nodes": nodes, "leaves": leaves,
                "action": "tanh(K_l o + beta_l), normalised to [-1, 1]"}


class TanhNormal:
    """Gaussian in pre-tanh space squashed to (-1, 1): the bounded proposal family of Eq. 30."""

    def __init__(self, pre_mean: Tensor, std: Tensor):
        self.base = torch.distributions.Normal(pre_mean, std)

    def sample(self) -> Tensor:
        return torch.tanh(self.base.sample())

    def log_prob(self, a: Tensor) -> Tensor:
        a = a.clamp(-ATANH_CLAMP, ATANH_CLAMP)
        z = torch.atanh(a)
        return self.base.log_prob(z).sum(-1) - torch.log(1.0 - a * a + 1e-6).sum(-1)

    def entropy(self) -> Tensor:
        return self.base.entropy().sum(-1)


class CarrierActor(nn.Module):
    """One carrier's parameter block theta_i: hard tree + exploration log-std."""

    def __init__(self, obs_dim: int, act_dim: int, depth: int = 4, log_std_init: float = -0.5):
        super().__init__()
        self.tree = HardObliqueTree(obs_dim, act_dim, depth)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def std(self) -> Tensor:
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def proposal(self, a_bar_i: Tensor) -> TanhNormal:
        """pi_theta_i( . | x, o ; a_bar_i): centred on the proposal centre block."""
        pre = torch.atanh(a_bar_i.clamp(-ATANH_CLAMP, ATANH_CLAMP))
        return TanhNormal(pre, self.std().expand_as(pre))


class TeamActor(nn.Module):
    """N independently parameterised carrier actors; observations / actions are
    concatenated over carriers (no parameter sharing, Eq. 2)."""

    def __init__(self, obs_dim_per: int, act_dim_per: int, n_agents: int, depth: int = 4,
                 log_std_init: float = -0.5):
        super().__init__()
        self.obs_dim_per, self.act_dim_per, self.n_agents = obs_dim_per, act_dim_per, n_agents
        self.actors = nn.ModuleList(
            [CarrierActor(obs_dim_per, act_dim_per, depth, log_std_init) for _ in range(n_agents)])

    def obs_i(self, obs: Tensor, i: int) -> Tensor:
        d = self.obs_dim_per
        return obs[:, i * d:(i + 1) * d]

    def act_i(self, a: Tensor, i: int) -> Tensor:
        d = self.act_dim_per
        return a[:, i * d:(i + 1) * d]

    def nominal_st(self, obs: Tensor, tau: float, detach_except: Optional[int] = None,
                   hard: bool = True, return_reg: bool = False):
        """Stacked training-time nominal a^ST (Sec. III-C).  With ``detach_except=i``
        every partner branch is detached before the SAIL map, as required by the
        independent surrogate L_i (Eq. 37).  ``return_reg`` also returns the
        per-carrier regulariser R(theta_i) (sum of its terms) as a list."""
        out, regs = [], []
        for i, actor in enumerate(self.actors):
            a_i, info = actor.tree(self.obs_i(obs, i), tau, hard)
            if detach_except is not None and i != detach_except:
                a_i = a_i.detach()
            out.append(a_i)
            if return_reg:
                regs.append(sum(actor.tree.regularizer(info).values()))
        a = torch.cat(out, dim=-1)
        return (a, regs) if return_reg else a

    def sample_proposals(self, a_bar: Tensor) -> tuple[Tensor, Tensor]:
        """a_prop ~ prod_i pi_theta_i (Eq. 31); returns [B, da] and per-carrier log-probs [B, N]."""
        acts, lps = [], []
        for i, actor in enumerate(self.actors):
            dist = actor.proposal(self.act_i(a_bar, i))
            a_i = dist.sample()
            acts.append(a_i)
            lps.append(dist.log_prob(a_i))
        return torch.cat(acts, -1), torch.stack(lps, -1)

    def log_prob_i(self, i: int, a_bar: Tensor, a_prop: Tensor) -> Tensor:
        return self.actors[i].proposal(self.act_i(a_bar, i)).log_prob(self.act_i(a_prop, i))

    def entropy(self, a_bar: Tensor) -> Tensor:
        return sum(self.actors[i].proposal(self.act_i(a_bar, i)).entropy().mean()
                   for i in range(self.n_agents)) / self.n_agents

    @torch.no_grad()
    def nominal(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Deployed stacked a_nom(o) (Eq. 21/25) and per-carrier leaf indices [B, N]."""
        acts, leaves = [], []
        for i, actor in enumerate(self.actors):
            a_i, l_i = actor.tree.nominal(self.obs_i(obs, i))
            acts.append(a_i)
            leaves.append(l_i)
        return torch.cat(acts, -1), torch.stack(leaves, -1)

    def block_parameters(self, i: int) -> list[nn.Parameter]:
        return list(self.actors[i].parameters())
