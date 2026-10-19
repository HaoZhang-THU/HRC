"""SAGE training loop (paper Algorithm 1) and deployment controller (Algorithm 2).

Rollout (Alg. 1, lines 3-9)        a_ST --Pi_kappa--> a_bar --pi_theta--> a_prop --Q_CBF--> a_exec   (Eq. 32)
    4  a_ST_i  = mu^ST_{theta_i,tau}(o_i)            tree nominal: soft routing, ST hard routing
                                                     (Eq. 26) in the final hard_finetune_fraction
    5  a_bar   = Pi_kappa(x, [a_ST_1; ...; a_ST_N])  SAIL penalty map (Eq. 27)
    6  a_prop_i ~ pi_theta_i(. | x, o; a_bar_i)      bounded proposal (Eq. 30)
    7  a_exec  = Q_CBF(x, a_prop)                    exact joint QP (Eq. 18); env stepped with a_exec
    8  store (o, x-rows, a_prop, a_exec, r, log pi_old_i)

Update (Alg. 1, lines 10-17)
    10 executed-transition GAE advantages (Eq. 34) from the team-context critic (Eq. 33)
    12 rebuild a_bar, rho_i, rho_team at the stored proposals (Eq. 35)
    13 L_i (partner branches detached before SAIL, Eq. 37) and L_team (Eq. 38)
    14 g_ind, g_team, W_gap (Eq. 39-40)
    15 g_TALO (Eq. 44)
    16 theta <- theta + eta g_TALO; critic regression on executed returns

Variants (same exact QP in every shielded arm, Sec. IV-A):
    tree       nominal tree, no filter, no SAIL, no TALO
    tree_cbf   + exact CBF-QP at execution
    sail       tree_cbf + SAIL
    talo       tree_cbf + TALO
    sage       tree_cbf + SAIL + TALO

The environment is reached through a small adapter object (see
``isaac_env.IsaacCoopTransportAdapter`` and the dummy adapter in
``smoke_test.py``) exposing ``reset``, ``step``, ``safety_state`` and the
dimensions; everything else is environment-agnostic torch.
"""
from __future__ import annotations

import faulthandler
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from .critic import JointCritic, TeamContextCritic, ValueNorm, huber_loss
from .safety import FormationSafetyModel, RowSet, SafetyConfig, SafetyState
from .sail import sail_penalty_map
from .talo import talo_update_field
from .tree_policy import TeamActor

VARIANTS = {
    #           use_cbf, use_sail, use_talo
    "tree":     (False, False, False),
    "tree_cbf": (True,  False, False),
    "sail":     (True,  True,  False),
    "talo":     (True,  False, True),
    "sage":     (True,  True,  True),
}
# critic per arm under critic="auto": the team-context critic (Eq. 33) belongs to the
# SAIL / TALO arms, the plain tree baselines use a joint-observation critic
JOINT_CRITIC_VARIANTS = ("tree", "tree_cbf")


@dataclass
class SAGEConfig:
    variant: str = "sage"
    # nominal tree (Sec. III-B)
    tree_depth: int = 4
    log_std_init: float = -0.5
    # gate temperature schedule tau_n (Alg. 1), fractions of total env-steps
    tau_init: float = 2.0
    tau_final: float = 0.12
    tau_anneal_start: float = 0.1
    tau_anneal_end: float = 0.8
    # soft routing until 1 - hard_finetune_fraction, ST hard routing afterwards
    # (1.0 = hard from the first iteration)
    hard_finetune_fraction: float = 0.15
    lambda_tree: float = 0.01
    # on the soft -> hard switch: freeze split hyperplanes/thresholds (leaf-only
    # finetune) and scale the actor lr once
    freeze_routing_on_hard: bool = True
    hard_lr_scale: float = 0.5
    # SAIL schedule kappa_n and unrolled solver (Eq. 27)
    kappa_init: float = 8.0
    kappa_final: float = 32.0
    kappa_anneal_start: float = 0.0
    kappa_anneal_end: float = 1.0
    sail_newton_steps: int = 8
    sail_damping: float = 1.0
    # TALO (Eq. 41-44)
    talo_sigma: float = 0.1
    talo_eps: float = 1e-8
    # PPO / critic
    gamma: float = 0.99
    gae_lambda: float = 0.9
    clip_eps: float = 0.2
    ppo_epochs: int = 15
    num_minibatches: int = 15
    actor_lr: float = 1e-4
    critic_lr: float = 5e-5
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.0
    num_steps: int = 60           # rollout length per iteration (high-level steps)
    critic: str = "auto"          # "auto" | "team" (Eq. 33) | "joint" (plain HAPPO critic)
    # actor update of the arms without TALO: "happo" (sequential per-carrier updates in a
    # random order, advantages weighted by the ratios of the carriers already updated) or
    # "joint" (one clipped joint-ratio step, the field g_team); TALO arms always use g_TALO
    actor_update: str = "auto"
    # joint critic value loss (HAPPO): ValueNorm targets, Huber, clipped at clip_eps
    value_huber_delta: float = 10.0
    # a warm-started actor paired with a freshly initialised critic: train only the
    # critic for this many iterations before the actor resumes updating
    critic_warmup_iters: int = 5
    context_dim: int = 64
    critic_hidden: int = 256
    # safety model
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    # diagnostics: ||Q_CBF(x, a_ST) - a_ST|| each rollout step (one extra QP)
    log_nominal_correction: bool = True

    def flags(self) -> tuple[bool, bool, bool]:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {self.variant!r}")
        return VARIANTS[self.variant]

    def critic_kind(self) -> str:
        if self.critic == "auto":
            return "joint" if self.variant in JOINT_CRITIC_VARIANTS else "team"
        if self.critic not in ("team", "joint"):
            raise ValueError(f"critic must be auto, team or joint, got {self.critic!r}")
        return self.critic

    def actor_update_rule(self) -> str:
        if self.flags()[2]:
            if self.actor_update not in ("auto", "talo"):
                raise ValueError(f"TALO arms update with g_TALO, got actor_update={self.actor_update!r}")
            return "talo"
        rule = "happo" if self.actor_update == "auto" else self.actor_update
        if rule not in ("happo", "joint"):
            raise ValueError(f"actor_update must be auto, happo or joint, got {self.actor_update!r}")
        return rule


def _linear(frac: float, v0: float, v1: float, s: float, e: float) -> float:
    a = min(1.0, max(0.0, (frac - s) / max(e - s, 1e-9)))
    return v0 + a * (v1 - v0)


class RunningMeanStd(torch.nn.Module):
    """Per-dimension running mean / variance (per-dimension counts, so a subset of
    channels can be re-estimated with ``reset_dims`` without touching the rest)."""

    def __init__(self, dim: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.full((dim,), eps))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "count"
        if key in state_dict and state_dict[key].dim() == 0:
            state_dict[key] = state_dict[key].expand_as(self.count).clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def reset_dims(self, idx) -> None:
        self.mean[idx], self.var[idx], self.count[idx] = 0.0, 1.0, self.eps

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), float(x.shape[0])
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a, m_b = self.var * self.count, bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def normalize(self, x: Tensor) -> Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10.0, 10.0)


class Rollout:
    def __init__(self, T: int, B: int, obs_dim: int, da: int, K: int, N: int, device):
        z = lambda *s, **k: torch.zeros(*s, device=device, **k)  # noqa: E731
        self.obs, self.next_obs = z(T, B, obs_dim), z(T, B, obs_dim)
        self.C, self.c, self.mask = z(T, B, K, da), z(T, B, K), z(T, B, K)
        self.a_prop, self.a_exec = z(T, B, da), z(T, B, da)
        self.logp = z(T, B, N)
        self.values, self.rewards = z(T, B), z(T, B)
        self.vpred = z(T, B)      # raw critic output (ValueNorm units for the joint critic)
        self.dones = z(T, B, dtype=torch.bool)
        self.T, self.B = T, B

    def rows(self, idx: Tensor) -> RowSet:
        C, c, m = (x.reshape(self.T * self.B, *x.shape[2:])[idx] for x in (self.C, self.c, self.mask))
        return RowSet(C, c, m, torch.zeros_like(c), torch.zeros_like(c, dtype=torch.long))


def gae(rewards: Tensor, values: Tensor, dones: Tensor, last_v: Tensor, gamma: float, lam: float):
    """Eq. 34, truncated at the rollout boundary, zeroed across episode ends."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    run = torch.zeros_like(last_v)
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else values[t + 1]
        nt = (~dones[t]).float()
        delta = rewards[t] + gamma * nv * nt - values[t]
        run = delta + gamma * lam * nt * run
        adv[t] = run
    return adv, adv + values


class SAGETrainer:
    def __init__(self, adapter, cfg: SAGEConfig, device: torch.device):
        self.env, self.cfg, self.device = adapter, cfg, device
        self.use_cbf, self.use_sail, self.use_talo = cfg.flags()
        self.N = adapter.n_agents
        self.do, self.da_i = adapter.obs_dim_per, adapter.act_dim_per
        self.obs_dim, self.da = self.do * self.N, self.da_i * self.N
        self.B = adapter.num_envs

        self.actor = TeamActor(self.do, self.da_i, self.N, cfg.tree_depth, cfg.log_std_init).to(device)
        self.critic_kind = cfg.critic_kind()
        self.actor_rule = cfg.actor_update_rule()
        if self.critic_kind == "team":
            self.critic = TeamContextCritic(self.do, self.N, cfg.context_dim, cfg.critic_hidden).to(device)
        else:
            self.critic = JointCritic(self.do, self.N).to(device)
        self.value_norm = ValueNorm().to(device) if self.critic_kind == "joint" else None
        self._critic_warmup_left = 0
        self.opt_a = Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_c = Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.obs_norm = RunningMeanStd(self.obs_dim).to(device)
        self.safety = FormationSafetyModel(cfg.safety)
        self.K = cfg.safety.max_rows
        self.buf = Rollout(cfg.num_steps, self.B, self.obs_dim, self.da, self.K, self.N, device)

        self.total_steps = 0
        self.best_return = float("-inf")
        self.tau, self.kappa = cfg.tau_init, cfg.kappa_init
        self.hard = cfg.hard_finetune_fraction >= 1.0
        # hard from the first iteration has no soft -> hard switch, so nothing is frozen
        self._hard_switched = self.hard
        self._obs: Optional[Tensor] = None
        self._ep_ret = torch.zeros(self.B, device=device)
        self.last_env_log: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def set_progress(self, frac: float) -> None:
        c = self.cfg
        self.tau = _linear(frac, c.tau_init, c.tau_final, c.tau_anneal_start, c.tau_anneal_end)
        self.kappa = _linear(frac, c.kappa_init, c.kappa_final, c.kappa_anneal_start, c.kappa_anneal_end)
        self.hard = c.hard_finetune_fraction > 0.0 and frac >= 1.0 - c.hard_finetune_fraction
        if self.hard and not self._hard_switched:
            self._switch_to_hard(scale_lr=True)

    def _switch_to_hard(self, scale_lr: bool) -> None:
        if self.cfg.freeze_routing_on_hard:
            for a in self.actor.actors:
                a.tree.split_w.requires_grad_(False)
                a.tree.split_b.requires_grad_(False)
        if scale_lr and self.cfg.hard_lr_scale != 1.0:
            for g in self.opt_a.param_groups:
                g["lr"] *= self.cfg.hard_lr_scale
        self._hard_switched = True
        print(f"[SAGE] hard routing on: routing frozen={self.cfg.freeze_routing_on_hard} "
              f"actor lr={self.opt_a.param_groups[0]['lr']:.2e}", flush=True)

    def _value(self, v_raw: Tensor) -> Tensor:
        return v_raw if self.value_norm is None else self.value_norm.denormalize(v_raw)

    def _value_loss(self, v: Tensor, vpred_old: Tensor, ret: Tensor) -> Tensor:
        if self.value_norm is None:
            return F.mse_loss(v, ret)
        self.value_norm.update(ret)
        target = self.value_norm.normalize(ret)
        v_clipped = vpred_old + (v - vpred_old).clamp(-self.cfg.clip_eps, self.cfg.clip_eps)
        d = self.cfg.value_huber_delta
        return torch.max(huber_loss(target - v, d), huber_loss(target - v_clipped, d)).mean()

    def _proposal_centre(self, a_st: Tensor, rows: RowSet) -> Tensor:
        """Alg. 1 line 5: SAIL map, or the identity when SAIL is disabled (Sec. III-C)."""
        if self.use_sail and self.use_cbf:
            return sail_penalty_map(a_st, rows, self.kappa, self.cfg.sail_newton_steps, self.cfg.sail_damping)
        return a_st

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def collect(self) -> dict[str, float]:
        cfg, buf = self.cfg, self.buf
        if self._obs is None:
            raw = self.env.reset()
            self.obs_norm.update(raw)
            self._obs = self.obs_norm.normalize(raw)
        obs = self._obs
        acc: dict[str, float] = {}
        ep_returns: list[float] = []

        def _acc(k, v):
            acc[k] = acc.get(k, 0.0) + float(v)

        for t in range(cfg.num_steps):
            state: SafetyState = self.env.safety_state()
            rows = self.safety.rows(state)
            a_st = self.actor.nominal_st(obs, self.tau, hard=self.hard)     # line 4
            a_bar = self._proposal_centre(a_st, rows)                       # line 5
            a_prop, logp = self.actor.sample_proposals(a_bar)               # line 6
            if self.use_cbf:
                a_exec, qp = self.safety.exact_projection(a_prop, rows)     # line 7
                _acc("qp_infeasible", (~qp["feasible"]).float().mean())
                _acc("correction", qp["correction"].mean())
                _acc("rows_active", qp["active"].float().sum(1).mean())
                if cfg.log_nominal_correction:
                    a_nom_exec, _ = self.safety.exact_projection(a_st, rows)
                    _acc("nominal_correction", (a_nom_exec - a_st).norm(dim=-1).mean())
            else:
                a_exec = a_prop.clamp(-1.0, 1.0)
            finite = torch.isfinite(a_exec).all(dim=-1)
            if not bool(finite.all()):
                # clamp() keeps NaN, and a NaN command would propagate into the simulator
                a_exec = torch.where(finite.unsqueeze(-1), a_exec, torch.zeros_like(a_exec))
            _acc("nonfinite_actions", (~finite).float().mean())
            _acc("pviol_nominal", rows.violated(a_st).float().mean())
            _acc("pviol_proposal", rows.violated(a_prop).float().mean())
            _acc("rows_present", rows.mask.sum(1).mean())
            _acc("min_h", rows.h.masked_fill(rows.mask <= 0, float("inf")).min(1).values.clamp(max=5.0).mean())
            v_raw = self.critic(obs)
            v = self._value(v_raw)
            buf.vpred[t] = v_raw

            next_raw, r, done, log = self.env.step(a_exec)
            self.obs_norm.update(next_raw)
            next_obs = self.obs_norm.normalize(next_raw)

            buf.obs[t], buf.next_obs[t] = obs, next_obs
            buf.C[t], buf.c[t], buf.mask[t] = rows.C, rows.c, rows.mask
            buf.a_prop[t], buf.a_exec[t], buf.logp[t] = a_prop, a_exec, logp
            buf.values[t], buf.rewards[t], buf.dones[t] = v, r, done

            self._ep_ret += r
            if done.any():
                ep_returns.extend(self._ep_ret[done].tolist())
                self._ep_ret[done] = 0.0
            for k, val in (log or {}).items():
                try:
                    _acc("env/" + k, float(val))
                except (TypeError, ValueError):
                    pass
            obs = next_obs

        self._obs = obs
        last_v = self._value(self.critic(obs))
        self.adv, self.ret = gae(buf.rewards, buf.values, buf.dones, last_v, cfg.gamma, cfg.gae_lambda)
        self.total_steps += cfg.num_steps * self.B
        out = {k: v / cfg.num_steps for k, v in acc.items()}
        out["episode_return"] = float(np.mean(ep_returns)) if ep_returns else float("nan")
        out["num_episodes"] = float(len(ep_returns))
        self.last_env_log = out
        return out

    # ------------------------------------------------------------------ #
    def _surrogates(self, o: Tensor, rows: RowSet, a_prop: Tensor, logp_old: Tensor, adv: Tensor):
        """L_i (Eq. 37) for every carrier and L_team (Eq. 38) on one minibatch."""
        cfg = self.cfg

        def clipped(ratio: Tensor) -> Tensor:
            return torch.minimum(ratio * adv, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()

        L_ind = []
        for i in range(self.N):
            a_st, regs = self.actor.nominal_st(o, self.tau, detach_except=i, hard=self.hard, return_reg=True)
            a_bar = self._proposal_centre(a_st, rows)
            lp = self.actor.log_prob_i(i, a_bar, a_prop)
            L_i = clipped(torch.exp(lp - logp_old[:, i])) - cfg.lambda_tree * regs[i]
            if cfg.entropy_coef > 0:
                L_i = L_i + cfg.entropy_coef * self.actor.actors[i].proposal(
                    self.actor.act_i(a_bar, i)).entropy().mean()
            L_ind.append(L_i)
        a_st, regs = self.actor.nominal_st(o, self.tau, hard=self.hard, return_reg=True)
        a_bar = self._proposal_centre(a_st, rows)
        lp_team = sum(self.actor.log_prob_i(i, a_bar, a_prop) for i in range(self.N))
        tree_reg = sum(regs)
        L_team = clipped(torch.exp(lp_team - logp_old.sum(1))) - cfg.lambda_tree * tree_reg
        return L_ind, L_team, tree_reg.detach()

    @torch.no_grad()
    def _logp_carrier(self, i: int, obs: Tensor, rows: RowSet, a_prop: Tensor) -> Tensor:
        """log pi_i(a_prop_i | o) at the current parameters of every carrier."""
        a_bar = self._proposal_centre(self.actor.nominal_st(obs, self.tau, hard=self.hard), rows)
        return self.actor.log_prob_i(i, a_bar, a_prop)

    def _update_happo(self, obs: Tensor, a_prop: Tensor, logp_old: Tensor, adv: Tensor,
                      ret: Tensor, vpred_old: Tensor, actor_frozen: bool) -> dict[str, float]:
        """HAPPO (Kuba et al., ICLR 2022): carriers are updated one
        after another in a random order, each for the full epochs x minibatches with its own
        clipped ratio and its own gradient clipping; the advantage of carrier i_m is weighted by
        the (unclipped) factor prod_{j before i_m} pi_j^new / pi_j^old over the carriers already
        updated in this iteration.  The critic is trained afterwards.  With SAIL the proposal
        centre of a carrier also depends on its partners, which are held at their current
        parameters while it is updated."""
        cfg, buf = self.cfg, self.buf
        n = obs.shape[0]
        mb = max(n // cfg.num_minibatches, 1)
        rows_all = buf.rows(torch.arange(n, device=self.device))
        stats: dict[str, float] = {}
        counts: dict[str, int] = {}

        def _acc(d: dict):
            for k, v in d.items():
                stats[k] = stats.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1

        if not actor_frozen:
            factor = torch.ones(n, device=self.device)
            order = torch.randperm(self.N).tolist()
            for m, i in enumerate(order):
                params_i = [p for p in self.actor.actors[i].parameters() if p.requires_grad]
                lp_before = self._logp_carrier(i, obs, rows_all, a_prop)
                for _ in range(cfg.ppo_epochs):
                    perm = torch.randperm(n, device=self.device)
                    for s in range(0, n, mb):
                        idx = perm[s:s + mb]
                        rows = buf.rows(idx)
                        if m == 0:
                            # W_gap / cos(g_ind, g_team) at the current parameters, logged as for
                            # the other arms (diagnostic only, not applied)
                            L_ind, L_team, tree_reg = self._surrogates(
                                obs[idx], rows, a_prop[idx], logp_old[idx], adv[idx])
                            _, ts = talo_update_field(L_ind, L_team, list(self.actor.parameters()),
                                                      cfg.talo_sigma, cfg.talo_eps, enabled=False)
                            _acc({"W_gap": ts["W_gap"], "cos_align": ts["cos_align"],
                                  "g_ind_norm": ts["g_ind_norm"], "g_team_norm": ts["g_team_norm"],
                                  "L_ind": float(sum(L_ind).detach()), "L_team": float(L_team.detach()),
                                  "tree_reg": float(tree_reg)})
                        a_st, regs = self.actor.nominal_st(obs[idx], self.tau, detach_except=i,
                                                           hard=self.hard, return_reg=True)
                        a_bar = self._proposal_centre(a_st, rows)
                        lp = self.actor.log_prob_i(i, a_bar, a_prop[idx])
                        ratio = torch.exp(lp - logp_old[idx, i])
                        surr = torch.minimum(ratio * adv[idx],
                                             ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[idx])
                        obj = (factor[idx] * surr).mean() - cfg.lambda_tree * regs[i]
                        if cfg.entropy_coef > 0:
                            obj = obj + cfg.entropy_coef * self.actor.actors[i].proposal(
                                self.actor.act_i(a_bar, i)).entropy().mean()
                        self.opt_a.zero_grad(set_to_none=True)
                        (-obj).backward()
                        step_norm = torch.nn.utils.clip_grad_norm_(params_i, cfg.max_grad_norm)
                        self.opt_a.step()
                        _acc({"step_norm": float(step_norm),
                              "step_clipped": float(step_norm > cfg.max_grad_norm),
                              "happo_ratio": float(ratio.detach().mean())})
                lp_after = self._logp_carrier(i, obs, rows_all, a_prop)
                factor = (factor * torch.exp(lp_after - lp_before)).detach()
            stats["happo_factor_mean"], counts["happo_factor_mean"] = float(factor.mean()), 1
            stats["happo_factor_std"], counts["happo_factor_std"] = float(factor.std()), 1

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                idx = perm[s:s + mb]
                lv = self._value_loss(self.critic(obs[idx]), vpred_old[idx], ret[idx])
                self.opt_c.zero_grad(set_to_none=True)
                lv.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.opt_c.step()
                _acc({"value_loss": float(lv.detach())})
        return {k: v / max(counts[k], 1) for k, v in stats.items()}

    def update(self) -> dict[str, float]:
        cfg, buf = self.cfg, self.buf
        T, B = cfg.num_steps, self.B
        n = T * B
        obs = buf.obs.reshape(n, self.obs_dim)
        a_prop = buf.a_prop.reshape(n, self.da)
        logp_old = buf.logp.reshape(n, self.N)
        adv = self.adv.reshape(n)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = self.ret.reshape(n)
        vpred_old = buf.vpred.reshape(n)
        params = [p for p in self.actor.parameters() if p.requires_grad]
        actor_frozen = self._critic_warmup_left > 0

        if self.actor_rule == "happo":
            out = self._update_happo(obs, a_prop, logp_old, adv, ret, vpred_old, actor_frozen)
            return self._finish_update(out, actor_frozen)

        mb = max(n // cfg.num_minibatches, 1)
        stats: dict[str, float] = {}
        count = 0

        def _acc(d: dict):
            for k, v in d.items():
                stats[k] = stats.get(k, 0.0) + float(v)

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, mb):
                idx = perm[s:s + mb]
                if not actor_frozen:
                    rows = buf.rows(idx)
                    L_ind, L_team, tree_reg = self._surrogates(obs[idx], rows, a_prop[idx], logp_old[idx], adv[idx])
                    field, ts = talo_update_field(L_ind, L_team, params, cfg.talo_sigma, cfg.talo_eps,
                                                  enabled=self.use_talo)
                    self.opt_a.zero_grad(set_to_none=True)
                    for p, g in zip(params, field):
                        p.grad = -g            # ascent field -> descent gradient for the optimiser
                    step_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    self.opt_a.step()
                    ts["step_norm"] = float(step_norm)
                    ts["step_clipped"] = float(step_norm > cfg.max_grad_norm)
                    _acc(ts)
                    _acc({"L_ind": float(sum(L_ind).detach()), "L_team": float(L_team.detach()),
                          "tree_reg": float(tree_reg)})

                lv = self._value_loss(self.critic(obs[idx]), vpred_old[idx], ret[idx])
                self.opt_c.zero_grad(set_to_none=True)
                lv.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.opt_c.step()

                _acc({"value_loss": float(lv.detach())})
                count += 1
        out = {k: v / max(count, 1) for k, v in stats.items()}
        return self._finish_update(out, actor_frozen)

    def _finish_update(self, out: dict[str, float], actor_frozen: bool) -> dict[str, float]:
        out.update({"tau": self.tau, "kappa": self.kappa, "hard_routing": float(self.hard),
                    "log_std": float(torch.stack([a.log_std for a in self.actor.actors]).mean()),
                    "critic_warmup": float(actor_frozen)})
        if actor_frozen:
            self._critic_warmup_left -= 1
            if self._critic_warmup_left == 0:
                print("[SAGE] critic warm-up done, actor updates resume", flush=True)
        return out

    # ------------------------------------------------------------------ #
    def checkpoint(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "obs_norm": self.obs_norm.state_dict(), "opt_a": self.opt_a.state_dict(),
                "opt_c": self.opt_c.state_dict(), "cfg": asdict(self.cfg),
                "total_steps": self.total_steps, "tau": self.tau, "kappa": self.kappa,
                "best_return": self.best_return, "hard_switched": self._hard_switched, "n_agents": self.N, "obs_dim_per": self.do, "act_dim_per": self.da_i,
                "critic_kind": self.critic_kind, "critic_warmup_left": self._critic_warmup_left,
                "value_norm": None if self.value_norm is None else self.value_norm.state_dict()}

    def load(self, path, load_critic: bool = True, load_obs_norm: bool = True,
             restore_progress: bool = False) -> None:
        """Warm-start from a checkpoint.  ``restore_progress`` additionally restores
        the env-step counter, both optimisers and tau/kappa, i.e. continues the
        same run instead of starting a new schedule."""
        ck = torch.load(str(path), map_location=self.device, weights_only=False)
        for k, cur in (("n_agents", self.N), ("obs_dim_per", self.do), ("act_dim_per", self.da_i)):
            if k in ck and int(ck[k]) != int(cur):
                raise ValueError(f"checkpoint {k}={ck[k]} != env {k}={cur}")
        self.actor.load_state_dict(ck["actor"])
        ck_kind = ck.get("critic_kind") or (
            "team" if any(k.startswith("encoder.") for k in ck.get("critic", {})) else "joint")
        critic_loaded = load_critic and "critic" in ck and ck_kind == self.critic_kind
        if critic_loaded:
            self.critic.load_state_dict(ck["critic"])
            if self.value_norm is not None and ck.get("value_norm") is not None:
                self.value_norm.load_state_dict(ck["value_norm"])
        elif "critic" in ck:
            print(f"[SAGE] checkpoint critic ({ck_kind}) not loaded; {self.critic_kind} critic "
                  f"starts from scratch", flush=True)
        if load_obs_norm and "obs_norm" in ck:
            self.obs_norm.load_state_dict(ck["obs_norm"])
        if restore_progress:
            self._critic_warmup_left = int(ck.get("critic_warmup_left", 0))
        elif not critic_loaded:
            self._critic_warmup_left = self.cfg.critic_warmup_iters
        if self._critic_warmup_left > 0:
            print(f"[SAGE] critic warm-up: actor frozen for {self._critic_warmup_left} iterations", flush=True)
        if restore_progress:
            self.opt_a.load_state_dict(ck["opt_a"])
            if critic_loaded:
                self.opt_c.load_state_dict(ck["opt_c"])
            self.total_steps = int(ck["total_steps"])
            self.tau, self.kappa = float(ck["tau"]), float(ck["kappa"])
            self.best_return = float(ck.get("best_return", float("-inf")))
            if ck.get("hard_switched") and not self._hard_switched:
                self._switch_to_hard(scale_lr=False)    # the scaled lr comes back with opt_a

    @torch.no_grad()
    def activate_inputs(self, dims: list[int]) -> None:
        """Warm start from a policy trained with these per-carrier observation channels
        held at zero: their weights never received a gradient, so they are zeroed in
        the actor (split hyperplanes, leaf gains) and in a loaded team-context critic,
        and the channels' normalisation statistics are re-estimated from new data.
        The warm-started policy is unchanged as long as the channels stay at zero."""
        idx = torch.as_tensor(dims, dtype=torch.long, device=self.device)
        for a in self.actor.actors:
            t = a.tree
            n_full = t.split_w.norm(dim=-1)
            t.split_w[:, idx] = 0.0
            n_rest = t.split_w.norm(dim=-1)
            # splits are unit-normalised: rescaling the threshold by the same factor keeps
            # sign(w^T o - b), i.e. the hard routing, identical
            t.split_b.mul_(torch.where(n_rest > 1e-8, n_full / n_rest.clamp_min(1e-8), torch.ones_like(n_rest)))
            t.leaf_K[:, :, idx] = 0.0
        if isinstance(self.critic, TeamContextCritic):
            self.critic.encoder[0].weight[:, idx] = 0.0
            for i in range(self.N):
                self.critic.value[0].weight[:, i * self.do + idx] = 0.0
        self.obs_norm.reset_dims(torch.cat([i * self.do + idx for i in range(self.N)]))
        print(f"[SAGE] activated {len(dims)} observation channels per carrier "
              f"(weights zeroed, normalisation re-estimated)", flush=True)

    def export_rules(self, feature_names=None) -> dict:
        return {f"carrier_{i}": a.tree.export_rules(feature_names) for i, a in enumerate(self.actor.actors)}


# ---------------------------------------------------------------------- #
# Algorithm 2: deployment controller (per control step)
# ---------------------------------------------------------------------- #
@torch.no_grad()
def deploy_action(actor: TeamActor, safety: FormationSafetyModel, obs_norm: RunningMeanStd,
                  obs_raw: Tensor, state: SafetyState, human_slot: Optional[int] = None,
                  a_human: Optional[Tensor] = None) -> tuple[Tensor, dict]:
    """a_exec = Q_CBF(x, a_nom(o)) (Eq. 22); with a human in slot i_H the measured
    block a_human ([B, 3], normalised) is held fixed and only the robot blocks are
    optimised (Eq. 19)."""
    obs = obs_norm.normalize(obs_raw)
    a_nom, leaves = actor.nominal(obs)
    rows = safety.rows(state)
    fixed = None
    if human_slot is not None:
        a_nom = a_nom.clone()
        a_nom[:, 3 * human_slot:3 * human_slot + 3] = a_human
        fixed = torch.zeros(a_nom.shape[1], dtype=torch.bool, device=a_nom.device)
        fixed[3 * human_slot:3 * human_slot + 3] = True
    a_exec, qp = safety.exact_projection(a_nom, rows, fixed=fixed)
    return a_exec, {"a_nom": a_nom, "leaves": leaves, "rows": rows, "violated": rows.violated(a_nom), **qp}


# ---------------------------------------------------------------------- #
def train(adapter, cfg: SAGEConfig, total_steps: int, log_dir: Path, device: torch.device,
          seed: int = 1, checkpoint_interval: int = 25, resume: Optional[str] = None,
          resume_load_critic: bool = True, resume_load_obs_norm: bool = True,
          resume_continue: bool = False, stall_timeout_s: float = 900.0,
          writer=None, print_every: int = 1, activate_dims: Optional[list[int]] = None) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainer = SAGETrainer(adapter, cfg, device)
    if resume:
        trainer.load(resume, resume_load_critic, resume_load_obs_norm, restore_progress=resume_continue)
        print(f"[{'continue' if resume_continue else 'warm-start'}] loaded {resume} "
              f"(steps={trainer.total_steps})", flush=True)
        if activate_dims and not resume_continue:
            trainer.activate_inputs(activate_dims)
    if stall_timeout_s and stall_timeout_s > 0:
        # fatal signals dump the Python stack; an iteration that exceeds the timeout
        # dumps every thread's stack and exits, so a supervisor can resume the run
        faulthandler.enable()
    log_dir = Path(log_dir)
    (log_dir / "models").mkdir(parents=True, exist_ok=True)
    if writer is None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(log_dir / "tensorboard"))
    best_path = log_dir / "best_model.pt"
    per_iter = cfg.num_steps * adapter.num_envs
    n_iter = (total_steps + per_iter - 1) // per_iter
    it0 = trainer.total_steps // per_iter
    print(f"[SAGE] variant={cfg.variant} cbf={trainer.use_cbf} sail={trainer.use_sail} talo={trainer.use_talo} "
          f"N={trainer.N} obs/agent={trainer.do} envs={adapter.num_envs} steps/iter={per_iter} "
          f"iters={n_iter} start_iter={it0}", flush=True)

    for it in range(it0, n_iter):
        t0 = time.time()
        if stall_timeout_s and stall_timeout_s > 0:
            faulthandler.dump_traceback_later(stall_timeout_s, exit=True)
        trainer.set_progress(trainer.total_steps / max(total_steps, 1))
        if hasattr(adapter, "set_train_progress"):
            adapter.set_train_progress(trainer.total_steps / max(total_steps, 1))
        m = trainer.collect()
        u = trainer.update()
        g = trainer.total_steps
        for k, v in m.items():
            if k.startswith("env/"):
                writer.add_scalar(k, v, g)
            elif k in ("episode_return", "num_episodes"):
                if not (isinstance(v, float) and np.isnan(v)):
                    writer.add_scalar("train/" + k, v, g)
            else:
                writer.add_scalar("safety/" + k, v, g)
        for k, v in u.items():
            writer.add_scalar("train/" + k, v, g)
        ret = m["episode_return"]
        if m["num_episodes"] > 0 and ret > trainer.best_return:
            trainer.best_return = ret
            torch.save(trainer.checkpoint(), best_path)
        torch.save(trainer.checkpoint(), log_dir / "models" / "latest.pt")
        if (it + 1) % checkpoint_interval == 0:
            torch.save(trainer.checkpoint(), log_dir / "models" / f"iter_{it + 1}.pt")
        if print_every and (it + 1) % print_every == 0:
            print(f"[{cfg.variant}] it {it + 1}/{n_iter} steps={g} ret={ret:.2f}(n={int(m['num_episodes'])}) "
                  f"pviol={m.get('pviol_proposal', float('nan')):.3f} corr={m.get('correction', float('nan')):.3f} "
                  f"infeas={m.get('qp_infeasible', float('nan')):.3f} Wgap={u.get('W_gap', float('nan')):.3g} "
                  f"cos={u.get('cos_align', float('nan')):.3f} tau={u['tau']:.2f}{'(hard)' if trainer.hard else ''} kappa={u['kappa']:.1f} "
                  f"{time.time() - t0:.1f}s", flush=True)
    faulthandler.cancel_dump_traceback_later()
    writer.close()
    (log_dir / "DONE").write_text(f"{trainer.total_steps}\n")
    return best_path
