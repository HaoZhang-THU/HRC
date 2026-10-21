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
    "tree":     (False, False, False),
    "tree_cbf": (True,  False, False),
    "sail":     (True,  True,  False),
    "talo":     (True,  False, True),
    "sage":     (True,  True,  True),
}
JOINT_CRITIC_VARIANTS = ("tree", "tree_cbf")


@dataclass
class SAGEConfig:
    variant: str = "sage"
    tree_depth: int = 4
    log_std_init: float = -0.5
    tau_init: float = 2.0
    tau_final: float = 0.12
    tau_anneal_start: float = 0.1
    tau_anneal_end: float = 0.8
    hard_finetune_fraction: float = 0.15
    lambda_tree: float = 0.01
    freeze_routing_on_hard: bool = True
    hard_lr_scale: float = 0.5
    kappa_init: float = 8.0
    kappa_final: float = 32.0
    kappa_anneal_start: float = 0.0
    kappa_anneal_end: float = 1.0
    sail_newton_steps: int = 8
    sail_damping: float = 1.0
    talo_sigma: float = 0.1
    talo_eps: float = 1e-8
    gamma: float = 0.99
    gae_lambda: float = 0.9
    clip_eps: float = 0.2
    ppo_epochs: int = 15
    num_minibatches: int = 15
    actor_lr: float = 1e-4
    critic_lr: float = 5e-5
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.0
    num_steps: int = 60
    critic: str = "auto"
    actor_update: str = "auto"
    value_huber_delta: float = 10.0
    critic_warmup_iters: int = 5
    context_dim: int = 64
    critic_hidden: int = 256
    safety: SafetyConfig = field(default_factory=SafetyConfig)
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


def _schedule(progress: float, start_value: float, end_value: float, start: float, end: float) -> float:
    weight = min(1.0, max(0.0, (progress - start) / max(end - start, 1e-9)))
    return start_value + weight * (end_value - start_value)


class RunningMeanStd(torch.nn.Module):
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
    def reset_dims(self, dims) -> None:
        self.mean[dims], self.var[dims], self.count[dims] = 0.0, 1.0, self.eps

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0, unbiased=False), float(x.shape[0])
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        old_m2, batch_m2 = self.var * self.count, batch_var * batch_count
        self.var = (old_m2 + batch_m2 + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: Tensor) -> Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10.0, 10.0)


class RolloutStorage:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        num_rows: int,
        num_agents: int,
        device,
    ):
        zeros = lambda *shape, **kwargs: torch.zeros(*shape, device=device, **kwargs)
        self.obs, self.next_obs = zeros(num_steps, num_envs, obs_dim), zeros(num_steps, num_envs, obs_dim)
        self.row_coef, self.row_bound, self.row_mask = (
            zeros(num_steps, num_envs, num_rows, act_dim),
            zeros(num_steps, num_envs, num_rows),
            zeros(num_steps, num_envs, num_rows),
        )
        self.proposals, self.actions = zeros(num_steps, num_envs, act_dim), zeros(num_steps, num_envs, act_dim)
        self.log_probs = zeros(num_steps, num_envs, num_agents)
        self.values, self.rewards = zeros(num_steps, num_envs), zeros(num_steps, num_envs)
        self.value_preds = zeros(num_steps, num_envs)
        self.dones = zeros(num_steps, num_envs, dtype=torch.bool)
        self.num_steps, self.num_envs = num_steps, num_envs

    def rows(self, index: Tensor) -> RowSet:
        coef, bound, mask = (
            x.reshape(self.num_steps * self.num_envs, *x.shape[2:])[index]
            for x in (self.row_coef, self.row_bound, self.row_mask)
        )
        return RowSet(coef, bound, mask, torch.zeros_like(bound), torch.zeros_like(bound, dtype=torch.long))


def compute_gae(rewards: Tensor, values: Tensor, dones: Tensor, last_values: Tensor, gamma: float, lam: float):
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_values)
    for step in reversed(range(num_steps)):
        next_values = last_values if step == num_steps - 1 else values[step + 1]
        not_done = (~dones[step]).float()
        delta = rewards[step] + gamma * next_values * not_done - values[step]
        running = delta + gamma * lam * not_done * running
        advantages[step] = running
    return advantages, advantages + values


class SAGETrainer:
    def __init__(self, adapter, cfg: SAGEConfig, device: torch.device):
        self.env, self.cfg, self.device = adapter, cfg, device
        self.use_cbf, self.use_sail, self.use_talo = cfg.flags()
        self.num_agents = adapter.num_agents
        self.obs_dim_per, self.act_dim_per = adapter.obs_dim_per, adapter.act_dim_per
        self.obs_dim, self.act_dim = self.obs_dim_per * self.num_agents, self.act_dim_per * self.num_agents
        self.num_envs = adapter.num_envs

        self.actor = TeamActor(
            self.obs_dim_per, self.act_dim_per, self.num_agents, cfg.tree_depth, cfg.log_std_init
        ).to(device)
        self.critic_kind = cfg.critic_kind()
        self.actor_update = cfg.actor_update_rule()
        if self.critic_kind == "team":
            self.critic = TeamContextCritic(
                self.obs_dim_per, self.num_agents, cfg.context_dim, cfg.critic_hidden
            ).to(device)
        else:
            self.critic = JointCritic(self.obs_dim_per, self.num_agents).to(device)
        self.value_norm = ValueNorm().to(device) if self.critic_kind == "joint" else None
        self._critic_warmup_left = 0
        self.actor_optimizer = Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.obs_norm = RunningMeanStd(self.obs_dim).to(device)
        self.safety = FormationSafetyModel(cfg.safety)
        self.num_rows = cfg.safety.max_rows
        self.storage = RolloutStorage(
            cfg.num_steps, self.num_envs, self.obs_dim, self.act_dim, self.num_rows, self.num_agents, device
        )

        self.total_steps = 0
        self.best_return = float("-inf")
        self.tau, self.kappa = cfg.tau_init, cfg.kappa_init
        self.hard_routing = cfg.hard_finetune_fraction >= 1.0
        self._hard_switched = self.hard_routing
        self._obs: Optional[Tensor] = None
        self._episode_returns = torch.zeros(self.num_envs, device=device)
        self.env_log: dict[str, float] = {}

    def set_progress(self, progress: float) -> None:
        cfg = self.cfg
        self.tau = _schedule(progress, cfg.tau_init, cfg.tau_final, cfg.tau_anneal_start, cfg.tau_anneal_end)
        self.kappa = _schedule(progress, cfg.kappa_init, cfg.kappa_final, cfg.kappa_anneal_start, cfg.kappa_anneal_end)
        self.hard_routing = cfg.hard_finetune_fraction > 0.0 and progress >= 1.0 - cfg.hard_finetune_fraction
        if self.hard_routing and not self._hard_switched:
            self._switch_to_hard(scale_lr=True)

    def _switch_to_hard(self, scale_lr: bool) -> None:
        if self.cfg.freeze_routing_on_hard:
            for actor in self.actor.actors:
                actor.tree.split_w.requires_grad_(False)
                actor.tree.split_b.requires_grad_(False)
        if scale_lr and self.cfg.hard_lr_scale != 1.0:
            for group in self.actor_optimizer.param_groups:
                group["lr"] *= self.cfg.hard_lr_scale
        self._hard_switched = True
        print(f"[SAGE] hard routing on: routing frozen={self.cfg.freeze_routing_on_hard} "
              f"actor lr={self.actor_optimizer.param_groups[0]['lr']:.2e}", flush=True)

    def _denormalize_value(self, raw_values: Tensor) -> Tensor:
        return raw_values if self.value_norm is None else self.value_norm.denormalize(raw_values)

    def _value_loss(self, values: Tensor, old_values: Tensor, returns: Tensor) -> Tensor:
        if self.value_norm is None:
            return F.mse_loss(values, returns)
        self.value_norm.update(returns)
        target = self.value_norm.normalize(returns)
        clipped_values = old_values + (values - old_values).clamp(-self.cfg.clip_eps, self.cfg.clip_eps)
        delta = self.cfg.value_huber_delta
        return torch.max(huber_loss(target - values, delta), huber_loss(target - clipped_values, delta)).mean()

    def _proposal_mean(self, nominal_actions: Tensor, rows: RowSet) -> Tensor:
        if self.use_sail and self.use_cbf:
            return sail_penalty_map(
                nominal_actions, rows, self.kappa, self.cfg.sail_newton_steps, self.cfg.sail_damping
            )
        return nominal_actions

    @torch.no_grad()
    def collect(self) -> dict[str, float]:
        cfg, storage = self.cfg, self.storage
        if self._obs is None:
            raw_obs = self.env.reset()
            self.obs_norm.update(raw_obs)
            self._obs = self.obs_norm.normalize(raw_obs)
        obs = self._obs
        sums: dict[str, float] = {}
        episode_returns: list[float] = []

        def _accumulate(key, value):
            sums[key] = sums.get(key, 0.0) + float(value)

        for step in range(cfg.num_steps):
            state: SafetyState = self.env.safety_state()
            rows = self.safety.rows(state)
            nominal_actions = self.actor.nominal_st(obs, self.tau, hard=self.hard_routing)
            proposal_mean = self._proposal_mean(nominal_actions, rows)
            proposals, log_probs = self.actor.sample_proposals(proposal_mean)
            if self.use_cbf:
                actions, qp_info = self.safety.exact_projection(proposals, rows)
                _accumulate("qp_infeasible", (~qp_info["feasible"]).float().mean())
                _accumulate("correction", qp_info["correction"].mean())
                _accumulate("rows_active", qp_info["active"].float().sum(1).mean())
                if cfg.log_nominal_correction:
                    projected_nominal, _ = self.safety.exact_projection(nominal_actions, rows)
                    _accumulate("nominal_correction", (projected_nominal - nominal_actions).norm(dim=-1).mean())
            else:
                actions = proposals.clamp(-1.0, 1.0)
            finite = torch.isfinite(actions).all(dim=-1)
            if not bool(finite.all()):
                actions = torch.where(finite.unsqueeze(-1), actions, torch.zeros_like(actions))
            _accumulate("nonfinite_actions", (~finite).float().mean())
            _accumulate("pviol_nominal", rows.violated(nominal_actions).float().mean())
            _accumulate("pviol_proposal", rows.violated(proposals).float().mean())
            _accumulate("rows_present", rows.mask.sum(1).mean())
            _accumulate(
                "min_h",
                rows.barrier.masked_fill(rows.mask <= 0, float("inf")).min(1).values.clamp(max=5.0).mean(),
            )
            raw_values = self.critic(obs)
            values = self._denormalize_value(raw_values)
            storage.value_preds[step] = raw_values

            next_raw_obs, rewards, dones, env_log = self.env.step(actions)
            self.obs_norm.update(next_raw_obs)
            next_obs = self.obs_norm.normalize(next_raw_obs)

            storage.obs[step], storage.next_obs[step] = obs, next_obs
            storage.row_coef[step], storage.row_bound[step], storage.row_mask[step] = rows.coef, rows.bound, rows.mask
            storage.proposals[step], storage.actions[step], storage.log_probs[step] = proposals, actions, log_probs
            storage.values[step], storage.rewards[step], storage.dones[step] = values, rewards, dones

            self._episode_returns += rewards
            if dones.any():
                episode_returns.extend(self._episode_returns[dones].tolist())
                self._episode_returns[dones] = 0.0
            for key, value in (env_log or {}).items():
                try:
                    _accumulate("env/" + key, float(value))
                except (TypeError, ValueError):
                    pass
            obs = next_obs

        self._obs = obs
        last_values = self._denormalize_value(self.critic(obs))
        self.advantages, self.returns = compute_gae(
            storage.rewards, storage.values, storage.dones, last_values, cfg.gamma, cfg.gae_lambda
        )
        self.total_steps += cfg.num_steps * self.num_envs
        stats = {key: values / cfg.num_steps for key, values in sums.items()}
        stats["episode_return"] = float(np.mean(episode_returns)) if episode_returns else float("nan")
        stats["num_episodes"] = float(len(episode_returns))
        self.env_log = stats
        return stats

    def _objectives(self, obs: Tensor, rows: RowSet, proposals: Tensor, old_log_probs: Tensor, advantages: Tensor):
        cfg = self.cfg

        def clipped(ratio: Tensor) -> Tensor:
            return torch.minimum(
                ratio * advantages, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
            ).mean()

        agent_objectives = []
        for i in range(self.num_agents):
            nominal_actions, regs = self.actor.nominal_st(
                obs, self.tau, detach_except=i, hard=self.hard_routing, return_reg=True
            )
            proposal_mean = self._proposal_mean(nominal_actions, rows)
            log_prob = self.actor.agent_log_prob(i, proposal_mean, proposals)
            objective = clipped(torch.exp(log_prob - old_log_probs[:, i])) - cfg.lambda_tree * regs[i]
            if cfg.entropy_coef > 0:
                objective = objective + cfg.entropy_coef * self.actor.actors[i].proposal(
                    self.actor.agent_action(proposal_mean, i)).entropy().mean()
            agent_objectives.append(objective)
        nominal_actions, regs = self.actor.nominal_st(obs, self.tau, hard=self.hard_routing, return_reg=True)
        proposal_mean = self._proposal_mean(nominal_actions, rows)
        team_log_prob = sum(self.actor.agent_log_prob(i, proposal_mean, proposals) for i in range(self.num_agents))
        tree_reg = sum(regs)
        team_objective = clipped(torch.exp(team_log_prob - old_log_probs.sum(1))) - cfg.lambda_tree * tree_reg
        return agent_objectives, team_objective, tree_reg.detach()

    @torch.no_grad()
    def _agent_log_prob(self, i: int, obs: Tensor, rows: RowSet, proposals: Tensor) -> Tensor:
        proposal_mean = self._proposal_mean(self.actor.nominal_st(obs, self.tau, hard=self.hard_routing), rows)
        return self.actor.agent_log_prob(i, proposal_mean, proposals)

    def _update_happo(self, obs: Tensor, proposals: Tensor, old_log_probs: Tensor, advantages: Tensor,
                      returns: Tensor, old_values: Tensor, actor_frozen: bool) -> dict[str, float]:
        cfg, storage = self.cfg, self.storage
        num_samples = obs.shape[0]
        batch_size = max(num_samples // cfg.num_minibatches, 1)
        all_rows = storage.rows(torch.arange(num_samples, device=self.device))
        stats: dict[str, float] = {}
        counts: dict[str, int] = {}

        def _accumulate(values: dict):
            for key, value in values.items():
                stats[key] = stats.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1

        if not actor_frozen:
            factor = torch.ones(num_samples, device=self.device)
            order = torch.randperm(self.num_agents).tolist()
            for position, i in enumerate(order):
                agent_params = [p for p in self.actor.actors[i].parameters() if p.requires_grad]
                old_log_prob = self._agent_log_prob(i, obs, all_rows, proposals)
                for _ in range(cfg.ppo_epochs):
                    perm = torch.randperm(num_samples, device=self.device)
                    for start in range(0, num_samples, batch_size):
                        idx = perm[start:start + batch_size]
                        rows = storage.rows(idx)
                        if position == 0:
                            agent_objectives, team_objective, tree_reg = self._objectives(
                                obs[idx], rows, proposals[idx], old_log_probs[idx], advantages[idx])
                            _, talo_stats = talo_update_field(
                                agent_objectives,
                                team_objective,
                                list(self.actor.parameters()),
                                cfg.talo_sigma,
                                cfg.talo_eps,
                                enabled=False,
                            )
                            _accumulate(
                                {
                                    "W_gap": talo_stats["W_gap"],
                                    "cos_align": talo_stats["cos_align"],
                                    "g_ind_norm": talo_stats["g_ind_norm"],
                                    "g_team_norm": talo_stats["g_team_norm"],
                                    "L_ind": float(sum(agent_objectives).detach()),
                                    "L_team": float(team_objective.detach()),
                                    "tree_reg": float(tree_reg),
                                }
                            )
                        nominal_actions, regs = self.actor.nominal_st(obs[idx], self.tau, detach_except=i,
                                                           hard=self.hard_routing, return_reg=True)
                        proposal_mean = self._proposal_mean(nominal_actions, rows)
                        log_prob = self.actor.agent_log_prob(i, proposal_mean, proposals[idx])
                        ratio = torch.exp(log_prob - old_log_probs[idx, i])
                        surrogate = torch.minimum(ratio * advantages[idx],
                                             ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages[idx])
                        objective = (factor[idx] * surrogate).mean() - cfg.lambda_tree * regs[i]
                        if cfg.entropy_coef > 0:
                            objective = objective + cfg.entropy_coef * self.actor.actors[i].proposal(
                                self.actor.agent_action(proposal_mean, i)).entropy().mean()
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        (-objective).backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(agent_params, cfg.max_grad_norm)
                        self.actor_optimizer.step()
                        _accumulate({"step_norm": float(grad_norm),
                              "step_clipped": float(grad_norm > cfg.max_grad_norm),
                              "happo_ratio": float(ratio.detach().mean())})
                new_log_prob = self._agent_log_prob(i, obs, all_rows, proposals)
                factor = (factor * torch.exp(new_log_prob - old_log_prob)).detach()
            stats["happo_factor_mean"], counts["happo_factor_mean"] = float(factor.mean()), 1
            stats["happo_factor_std"], counts["happo_factor_std"] = float(factor.std()), 1

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                idx = perm[start:start + batch_size]
                value_loss = self._value_loss(self.critic(obs[idx]), old_values[idx], returns[idx])
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()
                _accumulate({"value_loss": float(value_loss.detach())})
        return {k: v / max(counts[k], 1) for k, v in stats.items()}

    def update(self) -> dict[str, float]:
        cfg, storage = self.cfg, self.storage
        num_steps, num_envs = cfg.num_steps, self.num_envs
        num_samples = num_steps * num_envs
        obs = storage.obs.reshape(num_samples, self.obs_dim)
        proposals = storage.proposals.reshape(num_samples, self.act_dim)
        old_log_probs = storage.log_probs.reshape(num_samples, self.num_agents)
        advantages = self.advantages.reshape(num_samples)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = self.returns.reshape(num_samples)
        old_values = storage.value_preds.reshape(num_samples)
        params = [p for p in self.actor.parameters() if p.requires_grad]
        actor_frozen = self._critic_warmup_left > 0

        if self.actor_update == "happo":
            out = self._update_happo(obs, proposals, old_log_probs, advantages, returns, old_values, actor_frozen)
            return self._finish_update(out, actor_frozen)

        batch_size = max(num_samples // cfg.num_minibatches, 1)
        stats: dict[str, float] = {}
        count = 0

        def _accumulate(values: dict):
            for key, value in values.items():
                stats[key] = stats.get(key, 0.0) + float(value)

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                idx = perm[start:start + batch_size]
                if not actor_frozen:
                    rows = storage.rows(idx)
                    agent_objectives, team_objective, tree_reg = self._objectives(
                        obs[idx], rows, proposals[idx], old_log_probs[idx], advantages[idx]
                    )
                    field, talo_stats = talo_update_field(
                        agent_objectives,
                        team_objective,
                        params,
                        cfg.talo_sigma,
                        cfg.talo_eps,
                        enabled=self.use_talo,
                    )
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    for p, grad in zip(params, field):
                        p.grad = -grad
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    self.actor_optimizer.step()
                    talo_stats["step_norm"] = float(grad_norm)
                    talo_stats["step_clipped"] = float(grad_norm > cfg.max_grad_norm)
                    _accumulate(talo_stats)
                    _accumulate(
                        {
                            "L_ind": float(sum(agent_objectives).detach()),
                            "L_team": float(team_objective.detach()),
                            "tree_reg": float(tree_reg),
                        }
                    )

                value_loss = self._value_loss(self.critic(obs[idx]), old_values[idx], returns[idx])
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()

                _accumulate({"value_loss": float(value_loss.detach())})
                count += 1
        out = {k: v / max(count, 1) for k, v in stats.items()}
        return self._finish_update(out, actor_frozen)

    def _finish_update(self, out: dict[str, float], actor_frozen: bool) -> dict[str, float]:
        out.update({"tau": self.tau, "kappa": self.kappa, "hard_routing": float(self.hard_routing),
                    "log_std": float(torch.stack([actor.log_std for actor in self.actor.actors]).mean()),
                    "critic_warmup": float(actor_frozen)})
        if actor_frozen:
            self._critic_warmup_left -= 1
            if self._critic_warmup_left == 0:
                print("[SAGE] critic warm-up done, actor updates resume", flush=True)
        return out

    def checkpoint(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "obs_norm": self.obs_norm.state_dict(),
            "opt_a": self.actor_optimizer.state_dict(),
            "opt_c": self.critic_optimizer.state_dict(),
            "cfg": asdict(self.cfg),
            "total_steps": self.total_steps,
            "tau": self.tau,
            "kappa": self.kappa,
            "best_return": self.best_return,
            "hard_switched": self._hard_switched,
            "n_agents": self.num_agents,
            "obs_dim_per": self.obs_dim_per,
            "act_dim_per": self.act_dim_per,
            "critic_kind": self.critic_kind,
            "critic_warmup_left": self._critic_warmup_left,
            "value_norm": None if self.value_norm is None else self.value_norm.state_dict(),
        }

    def load(self, path, load_critic: bool = True, load_obs_norm: bool = True,
             restore_progress: bool = False) -> None:
        loaded_dict = torch.load(str(path), map_location=self.device, weights_only=False)
        for key, current in (
            ("n_agents", self.num_agents),
            ("obs_dim_per", self.obs_dim_per),
            ("act_dim_per", self.act_dim_per),
        ):
            if key in loaded_dict and int(loaded_dict[key]) != int(current):
                raise ValueError(f"checkpoint {key}={loaded_dict[key]} != env {key}={current}")
        self.actor.load_state_dict(loaded_dict["actor"])
        saved_kind = loaded_dict.get("critic_kind") or (
            "team" if any(key.startswith("encoder.") for key in loaded_dict.get("critic", {})) else "joint")
        critic_loaded = load_critic and "critic" in loaded_dict and saved_kind == self.critic_kind
        if critic_loaded:
            self.critic.load_state_dict(loaded_dict["critic"])
            if self.value_norm is not None and loaded_dict.get("value_norm") is not None:
                self.value_norm.load_state_dict(loaded_dict["value_norm"])
        elif "critic" in loaded_dict:
            print(f"[SAGE] checkpoint critic ({saved_kind}) not loaded; {self.critic_kind} critic "
                  f"starts from scratch", flush=True)
        if load_obs_norm and "obs_norm" in loaded_dict:
            self.obs_norm.load_state_dict(loaded_dict["obs_norm"])
        if restore_progress:
            self._critic_warmup_left = int(loaded_dict.get("critic_warmup_left", 0))
        elif not critic_loaded:
            self._critic_warmup_left = self.cfg.critic_warmup_iters
        if self._critic_warmup_left > 0:
            print(f"[SAGE] critic warm-up: actor frozen for {self._critic_warmup_left} iterations", flush=True)
        if restore_progress:
            self.actor_optimizer.load_state_dict(loaded_dict["opt_a"])
            if critic_loaded:
                self.critic_optimizer.load_state_dict(loaded_dict["opt_c"])
            self.total_steps = int(loaded_dict["total_steps"])
            self.tau, self.kappa = float(loaded_dict["tau"]), float(loaded_dict["kappa"])
            self.best_return = float(loaded_dict.get("best_return", float("-inf")))
            if loaded_dict.get("hard_switched") and not self._hard_switched:
                self._switch_to_hard(scale_lr=False)

    @torch.no_grad()
    def activate_inputs(self, dims: list[int]) -> None:
        index = torch.as_tensor(dims, dtype=torch.long, device=self.device)
        for actor in self.actor.actors:
            tree = actor.tree
            full_norm = tree.split_w.norm(dim=-1)
            tree.split_w[:, index] = 0.0
            rest_norm = tree.split_w.norm(dim=-1)
            tree.split_b.mul_(
                torch.where(
                    rest_norm > 1e-8, full_norm / rest_norm.clamp_min(1e-8), torch.ones_like(rest_norm)
                )
            )
            tree.leaf_K[:, :, index] = 0.0
        if isinstance(self.critic, TeamContextCritic):
            self.critic.encoder[0].weight[:, index] = 0.0
            for i in range(self.num_agents):
                self.critic.value[0].weight[:, i * self.obs_dim_per + index] = 0.0
        self.obs_norm.reset_dims(torch.cat([i * self.obs_dim_per + index for i in range(self.num_agents)]))
        print(f"[SAGE] activated {len(dims)} observation channels per carrier "
              f"(weights zeroed, normalisation re-estimated)", flush=True)

    def export_rules(self, feature_names=None) -> dict:
        return {f"carrier_{i}": actor.tree.export_rules(feature_names) for i, actor in enumerate(self.actor.actors)}


@torch.no_grad()
def deploy_action(actor: TeamActor, safety: FormationSafetyModel, obs_norm: RunningMeanStd,
                  raw_obs: Tensor, state: SafetyState, human_slot: Optional[int] = None,
                  human_action: Optional[Tensor] = None) -> tuple[Tensor, dict]:
    obs = obs_norm.normalize(raw_obs)
    nominal_actions, leaves = actor.nominal(obs)
    rows = safety.rows(state)
    fixed = None
    if human_slot is not None:
        nominal_actions = nominal_actions.clone()
        nominal_actions[:, 3 * human_slot:3 * human_slot + 3] = human_action
        fixed = torch.zeros(nominal_actions.shape[1], dtype=torch.bool, device=nominal_actions.device)
        fixed[3 * human_slot:3 * human_slot + 3] = True
    actions, qp_info = safety.exact_projection(nominal_actions, rows, fixed=fixed)
    return actions, {
        "a_nom": nominal_actions,
        "leaves": leaves,
        "rows": rows,
        "violated": rows.violated(nominal_actions),
        **qp_info,
    }


def train(adapter, cfg: SAGEConfig, total_steps: int, log_dir: Path, device: torch.device,
          seed: int = 1, checkpoint_interval: int = 25, resume: Optional[str] = None,
          resume_load_critic: bool = True, resume_load_obs_norm: bool = True,
          resume_continue: bool = False, stall_timeout_s: float = 900.0,
          writer=None, print_every: int = 1, activate_dims: Optional[list[int]] = None,
          max_iters: Optional[int] = None) -> Path:
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
        faulthandler.enable()
    log_dir = Path(log_dir)
    (log_dir / "models").mkdir(parents=True, exist_ok=True)
    if writer is None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(log_dir / "tensorboard"))
    best_path = log_dir / "best_model.pt"
    steps_per_iter = cfg.num_steps * adapter.num_envs
    num_iters = (total_steps + steps_per_iter - 1) // steps_per_iter
    if max_iters:
        num_iters = min(num_iters, max_iters)
    start_iter = trainer.total_steps // steps_per_iter
    print(f"[SAGE] variant={cfg.variant} cbf={trainer.use_cbf} sail={trainer.use_sail} talo={trainer.use_talo} "
          f"N={trainer.num_agents} obs/agent={trainer.obs_dim_per} envs={adapter.num_envs} steps/iter={steps_per_iter} "
          f"iters={num_iters} start_iter={start_iter}", flush=True)

    for iteration in range(start_iter, num_iters):
        start_time = time.time()
        if stall_timeout_s and stall_timeout_s > 0:
            faulthandler.dump_traceback_later(stall_timeout_s, exit=True)
        trainer.set_progress(trainer.total_steps / max(total_steps, 1))
        if hasattr(adapter, "set_train_progress"):
            adapter.set_train_progress(trainer.total_steps / max(total_steps, 1))
        collect_stats = trainer.collect()
        update_stats = trainer.update()
        step = trainer.total_steps
        for key, value in collect_stats.items():
            if key.startswith("env/"):
                writer.add_scalar(key, value, step)
            elif key in ("episode_return", "num_episodes"):
                if not (isinstance(value, float) and np.isnan(value)):
                    writer.add_scalar("train/" + key, value, step)
            else:
                writer.add_scalar("safety/" + key, value, step)
        for key, value in update_stats.items():
            writer.add_scalar("train/" + key, value, step)
        episode_return = collect_stats["episode_return"]
        if collect_stats["num_episodes"] > 0 and episode_return > trainer.best_return:
            trainer.best_return = episode_return
            torch.save(trainer.checkpoint(), best_path)
        torch.save(trainer.checkpoint(), log_dir / "models" / "latest.pt")
        if (iteration + 1) % checkpoint_interval == 0:
            torch.save(trainer.checkpoint(), log_dir / "models" / f"iter_{iteration + 1}.pt")
        if print_every and (iteration + 1) % print_every == 0:
            print(
                f"[{cfg.variant}] it {iteration + 1}/{num_iters} steps={step} ret={episode_return:.2f}(n={int(collect_stats['num_episodes'])}) "
                f"pviol={collect_stats.get('pviol_proposal', float('nan')):.3f} corr={collect_stats.get('correction', float('nan')):.3f} "
                f"infeas={collect_stats.get('qp_infeasible', float('nan')):.3f} Wgap={update_stats.get('W_gap', float('nan')):.3g} "
                f"cos={update_stats.get('cos_align', float('nan')):.3f} tau={update_stats['tau']:.2f}{'(hard)' if trainer.hard_routing else ''} kappa={update_stats['kappa']:.1f} "
                f"{time.time() - start_time:.1f}s",
                flush=True,
            )
    faulthandler.cancel_dump_traceback_later()
    writer.close()
    (log_dir / "DONE").write_text(f"{trainer.total_steps}\n")
    return best_path
