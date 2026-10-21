#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaac_sage.safety import FormationSafetyModel, RowSet, SafetyConfig, SafetyState, rotation_2d
from isaac_sage.sail import sail_penalty_map
from isaac_sage.talo import talo_update_field
from isaac_sage.trainer import SAGEConfig, SAGETrainer
from isaac_sage.tree_policy import TeamActor

torch.set_default_dtype(torch.float32)
DEV = torch.device("cpu")


class PlanarFormationEnv:
    def __init__(self, num_envs=16, obs_dim_per=12, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.generator = generator
        self.num_envs, self.num_agents, self.obs_dim_per, self.act_dim_per = num_envs, 2, obs_dim_per, 3
        self.device = DEV
        self.action_scale = torch.tensor([0.2, 0.2, 0.3])
        self.anchors_local = torch.tensor([[1.2, 0.0], [-1.2, 0.0], [0.0, 0.7]])
        self.controlled = torch.tensor([0, 1])
        self.aabbs = torch.tensor([[-3.0, 6.0, 1.3, 1.8], [-3.0, 6.0, -1.8, -1.3], [2.5, 3.0, -0.4, 0.4]])
        self.goal = torch.tensor([5.0, 0.0])
        self.step_count = 0

    def _obs(self):
        rel = self.goal - self.board_xy
        feats = [rel, torch.stack((torch.cos(self.board_yaw), torch.sin(self.board_yaw)), -1),
                 self.board_xy]
        core = torch.cat(feats, -1)
        pad = torch.zeros(self.num_envs, self.obs_dim_per - core.shape[1])
        obs = torch.cat((core, pad), -1)
        return torch.cat((obs, obs), -1)

    def reset(self):
        self.board_xy = torch.zeros(self.num_envs, 2)
        self.board_xy[:, 1] = torch.rand(self.num_envs, generator=self.generator) * 0.6 - 0.3
        self.board_yaw = torch.zeros(self.num_envs)
        self.step_count = 0
        return self._obs()

    def _robot_pose(self):
        rot = rotation_2d(self.board_yaw)
        xy = self.board_xy[:, None] + torch.einsum("bij,rj->bri", rot, self.anchors_local)
        yaw = self.board_yaw[:, None] + torch.tensor([math.pi, 0.0, -math.pi / 2])
        return xy, yaw

    def step(self, actions):
        robot_xy, robot_yaw = self._robot_pose()
        command = actions.reshape(self.num_envs, 2, 3).clamp(-1, 1) * self.action_scale
        disp = torch.einsum(
            "bnij,bnj->bni",
            rotation_2d(robot_yaw[:, :2].reshape(-1)).reshape(self.num_envs, 2, 2, 2),
            command[..., :2],
        )
        vel = disp.mean(1)
        self.board_xy = self.board_xy + vel
        self.board_yaw = self.board_yaw + 0.3 * command[..., 2].mean(1)
        self.step_count += 1
        dist = (self.goal - self.board_xy).norm(dim=-1)
        rewards = -0.1 * dist + 2.0 * (dist < 0.5).float()
        dones = (dist < 0.5) | torch.full((self.num_envs,), self.step_count >= 40)
        if dones.any():
            self.board_xy[dones] = 0.0
            self.board_yaw[dones] = 0.0
        if self.step_count >= 40:
            self.step_count = 0
        return self._obs(), rewards, dones, {"dist": float(dist.mean())}

    def safety_state(self):
        robot_xy, robot_yaw = self._robot_pose()
        return SafetyState(board_xy=self.board_xy, board_yaw=self.board_yaw, robot_xy=robot_xy, robot_yaw=robot_yaw,
                           controlled=self.controlled, anchors_local=self.anchors_local, aabbs=self.aabbs,
                           action_scale=self.action_scale)


def check(cond: bool, msg: str):
    print(("  [ok]  " if cond else "  [FAIL]") + msg)
    if not cond:
        raise SystemExit(1)


def main():
    torch.manual_seed(0)
    env = PlanarFormationEnv()
    env.reset()
    model = FormationSafetyModel(SafetyConfig(activation_distance=3.0))
    state = env.safety_state()
    rows = model.rows(state)
    act_dim = rows.coef.shape[-1]
    print(
        f"rows: K={rows.coef.shape[1]} present/env={rows.mask.sum(1).mean():.1f} min h={rows.barrier[rows.mask > 0].min():.3f}"
    )

    zero = torch.zeros(env.num_envs, act_dim)
    check(bool((rows.margins(zero) >= -1e-6).all()), "zero action satisfies every row (h >= 0 => c <= 0)")

    a = torch.rand(env.num_envs, act_dim) * 2 - 1
    a[:, 0] = -1.0
    out, info = model.exact_projection(a, rows)
    check(bool(info["feasible"].all()) and float(rows.margins(out).min()) > -model.cfg.qp_tolerance,
          f"projected action satisfies rows (min residual {float(rows.margins(out).min()):+.4f})")
    out2, _ = model.exact_projection(out, rows)
    check(float((out2 - out).abs().max()) < 1e-3, "projection is idempotent on feasible inputs")
    check(float((out - a).norm(dim=-1).mean()) > 0, "at least one input was corrected")

    fixed = torch.zeros(act_dim, dtype=torch.bool)
    fixed[3:6] = True
    out_fixed, _ = model.exact_projection(a, rows, fixed=fixed)
    check(bool(torch.allclose(out_fixed[:, 3:6], a[:, 3:6])), "measured partner block held fixed (Eq. 19)")

    kappa = 8.0
    coef = torch.zeros(1, 1, act_dim); coef[0, 0, 0] = 1.0
    bound = torch.full((1, 1), 0.5)
    one_row = RowSet(coef, bound, torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.long))
    nominal_actions = torch.zeros(1, act_dim, requires_grad=True)
    proposal_mean = sail_penalty_map(nominal_actions, one_row, kappa, newton_steps=8, damping=1.0)
    (jac,) = torch.autograd.grad(proposal_mean[0, 0], nominal_actions)
    expect = 1.0 / (1.0 + kappa)
    check(
        abs(float(jac[0, 0]) - expect) < 1e-4,
        f"SAIL normal gain {float(jac[0, 0]):.4f} == 1/(1+kappa) = {expect:.4f}",
    )
    check(abs(float(proposal_mean[0, 0]) - kappa * 0.5 / (1 + kappa)) < 1e-4, "SAIL minimiser matches closed form")
    proj0, _ = model.exact_projection(torch.zeros(1, act_dim), one_row)
    proj1, _ = model.exact_projection(torch.zeros(1, act_dim).index_fill(1, torch.tensor([0]), 0.1), one_row)
    check(abs(float(proj1[0, 0] - proj0[0, 0])) < 1e-3, "exact QP normal gain == 0 (projection annihilates it)")
    stiff_actions = sail_penalty_map(torch.zeros(1, act_dim), one_row, 1e4, newton_steps=8, damping=1.0)
    check(abs(float(stiff_actions[0, 0]) - 0.5) < 1e-3, "kappa -> inf recovers the exact projection")

    actor = TeamActor(env.obs_dim_per, 3, 2, depth=3)
    obs = torch.randn(32, 2 * env.obs_dim_per)
    hard_actions, _ = actor.nominal(obs)
    st_actions = actor.nominal_st(obs, tau=0.5)
    check(bool(torch.allclose(hard_actions, st_actions.detach(), atol=1e-6)), "ST forward == deployed hard routing")
    st_actions.sum().backward()
    check(actor.actors[0].tree.split_w.grad is not None and float(actor.actors[0].tree.split_w.grad.abs().sum()) > 0,
          "split hyperplanes receive straight-through gradients")
    detached_actions = actor.nominal_st(obs, tau=0.5, detach_except=1)
    g = torch.autograd.grad(detached_actions.sum(), list(actor.actors[0].parameters()), allow_unused=True)
    check(all(x is None for x in g), "partner branch detached (detach_except) carries no gradient")
    soft_actions, regs = actor.nominal_st(obs, tau=2.0, hard=False, return_reg=True)
    tree0 = actor.actors[0].tree
    probs = tree0(actor.agent_obs(obs, 0), 2.0, hard=False)[1]["leaf_probs"]
    check(bool(torch.allclose(probs.sum(1), torch.ones(32), atol=1e-5)) and float(probs.max(1).values.min()) < 0.999,
          "soft routing: path probabilities sum to 1 and mix leaves")
    check(
        not torch.allclose(soft_actions, hard_actions, atol=1e-4),
        "soft-phase nominal differs from deployed hard routing",
    )
    actor.zero_grad()
    regs[0].backward()
    check(float(tree0.split_w.grad.abs().sum()) > 0 and float(tree0.split_b.grad.abs().sum()) > 0
          and all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in actor.actors[1].parameters()),
          f"structure regulariser R={float(regs[0]):.3f} reaches own splits only")
    sched_trainer = SAGETrainer(PlanarFormationEnv(num_envs=4), SAGEConfig(tree_depth=3), DEV)
    phases = []
    for f in (0.0, 0.5, 0.84, 0.86, 1.0):
        sched_trainer.set_progress(f)
        phases.append((sched_trainer.hard_routing, round(sched_trainer.tau, 3), round(sched_trainer.kappa, 1)))
    check([p[0] for p in phases] == [False, False, False, True, True] and phases[0][1] == 2.0
          and phases[-1][1] == 0.12 and phases[0][2] == 8.0 and phases[-1][2] == 32.0,
          f"schedule (hard, tau, kappa): {phases}")
    switch_trainer = SAGETrainer(PlanarFormationEnv(num_envs=8), SAGEConfig(variant="sage", tree_depth=3, num_steps=10,
                                                                 ppo_epochs=1, num_minibatches=2), DEV)
    switch_trainer.set_progress(0.5); switch_trainer.collect(); switch_trainer.update()
    lr0 = switch_trainer.actor_optimizer.param_groups[0]["lr"]
    switch_trainer.set_progress(0.9)
    switch_tree = switch_trainer.actor.actors[0].tree
    before = (
        switch_tree.split_w.detach().clone(),
        switch_tree.split_b.detach().clone(),
        switch_tree.leaf_K.detach().clone(),
    )
    switch_trainer.collect(); switch_trainer.update()
    check(
        torch.equal(before[0], switch_tree.split_w)
        and torch.equal(before[1], switch_tree.split_b)
        and not torch.equal(before[2], switch_tree.leaf_K)
        and abs(switch_trainer.actor_optimizer.param_groups[0]["lr"] - 0.5 * lr0) < 1e-12,
        "soft -> hard switch freezes routing, leaves keep training, actor lr halved once",
    )
    torch.save(switch_trainer.checkpoint(), "/tmp/_sage_hard_ck.pt")
    resume_trainer = SAGETrainer(
        PlanarFormationEnv(num_envs=8), SAGEConfig(variant="sage", tree_depth=3, num_steps=10), DEV
    )
    resume_trainer.load("/tmp/_sage_hard_ck.pt", restore_progress=True)
    resume_trainer.set_progress(0.95)
    check(not resume_trainer.actor.actors[1].tree.split_w.requires_grad
          and abs(resume_trainer.actor_optimizer.param_groups[0]["lr"] - 0.5 * lr0) < 1e-12,
          "resume in hard phase re-freezes routing without scaling lr twice")

    p = [torch.nn.Parameter(torch.randn(5)), torch.nn.Parameter(torch.randn(3))]
    agent_objectives = [(p[0] ** 2).sum() + torch.sin(p[0]).sum(), (p[1] ** 3).sum()]
    team_objective = (p[0] ** 2).sum() * 1.3 + (p[1] ** 2).sum() + (p[0].sum() * p[1].sum())
    field, stats = talo_update_field(agent_objectives, team_objective, p, sigma=0.1, eps=1e-8, enabled=True)
    gap = 0.5 * sum(((torch.autograd.grad(sum(agent_objectives), p, create_graph=True)[i]
                    - torch.autograd.grad(team_objective, p, create_graph=True)[i]) ** 2).sum() for i in range(2))
    gap_grads = torch.autograd.grad(gap, p)
    lhs = float(sum((h * f).sum() for h, f in zip(gap_grads, field)) + 0.1 * gap)
    check(lhs <= 1e-5, f"grad W^T g_TALO + sigma W = {lhs:+.3e} <= 0 (Eq. 41)")
    agent_objectives = [(p[0] ** 2).sum() + torch.sin(p[0]).sum(), (p[1] ** 3).sum()]
    team_objective = (p[0] ** 2).sum() * 1.3 + (p[1] ** 2).sum() + (p[0].sum() * p[1].sum())
    team_grads0 = torch.autograd.grad(team_objective, p, retain_graph=True)
    field0, stats0 = talo_update_field(agent_objectives, team_objective, p, sigma=0.1, eps=1e-8, enabled=False)
    check(stats0["talo_lambda"] == 0.0 and all(torch.allclose(f, g) for f, g in zip(field0, team_grads0)),
          "TALO disabled returns g_team")

    for variant in ("tree", "tree_cbf", "sail", "talo", "sage"):
        env = PlanarFormationEnv()
        cfg = SAGEConfig(variant=variant, tree_depth=3, num_steps=20, ppo_epochs=2, num_minibatches=4,
                         safety=SafetyConfig(activation_distance=3.0, max_rows=24))
        trainer = SAGETrainer(env, cfg, DEV)
        for iteration in range(2):
            trainer.set_progress(iteration / 2)
            collect_stats = trainer.collect()
            update_stats = trainer.update()
        check(
            all(math.isfinite(v) for v in update_stats.values()),
            f"variant {variant:8s}: finite update stats "
            f"(Wgap={update_stats.get('W_gap', 0):.3g}, cos={update_stats.get('cos_align', 0):+.2f}, lam={update_stats.get('talo_lambda', 0):.3g}, "
            f"pviol={collect_stats.get('pviol_proposal', 0):.2f}, corr={collect_stats.get('correction', 0):.3f})",
        )
        if variant == "sage":
            torch.save(trainer.checkpoint(), "/tmp/_sage_team_critic_ck.pt")

    from isaac_sage.critic import JointCritic
    cfg = SAGEConfig(variant="tree_cbf", tree_depth=3, num_steps=20, ppo_epochs=2, num_minibatches=4,
                     critic_warmup_iters=2, safety=SafetyConfig(activation_distance=3.0, max_rows=24))
    joint_trainer = SAGETrainer(PlanarFormationEnv(), cfg, DEV)
    joint_trainer.load("/tmp/_sage_team_critic_ck.pt")
    snap = lambda: torch.cat([p.detach().flatten().clone() for p in joint_trainer.actor.parameters()])
    moved, value_losses = [], []
    for iteration in range(3):
        before = snap()
        joint_trainer.set_progress(iteration / 3)
        joint_trainer.collect()
        update_stats = joint_trainer.update()
        moved.append(bool((snap() - before).abs().max() > 0))
        value_losses.append(round(update_stats["value_loss"], 3))
    check(
        isinstance(joint_trainer.critic, JointCritic)
        and not any(k.startswith("encoder") for k in joint_trainer.critic.state_dict())
        and moved == [False, False, True]
        and all(math.isfinite(v) for v in value_losses),
        f"HAPPO critic, no team context; actor frozen for {cfg.critic_warmup_iters} warm-up iterations "
        f"(actor moved {moved}, value loss {value_losses})",
    )
    torch.save(joint_trainer.checkpoint(), "/tmp/_tree_cbf_joint_ck.pt")
    resumed_trainer = SAGETrainer(PlanarFormationEnv(), cfg, DEV)
    resumed_trainer.load("/tmp/_tree_cbf_joint_ck.pt", restore_progress=True)
    check(
        resumed_trainer._critic_warmup_left == 0
        and torch.equal(resumed_trainer.value_norm.running_mean, joint_trainer.value_norm.running_mean),
        "resume restores the joint critic and ValueNorm without a second warm-up",
    )
    from isaac_sage.trainer import RunningMeanStd
    activate_trainer = SAGETrainer(PlanarFormationEnv(), SAGEConfig(variant="sage", tree_depth=3, num_steps=10), DEV)
    dims = [0, 1]
    obs = torch.randn(512, activate_trainer.obs_dim)
    for i in range(activate_trainer.num_agents):
        obs[:, i * activate_trainer.obs_dim_per + torch.tensor(dims)] = 0.0
    leaves0 = [
        a.tree.route(activate_trainer.actor.agent_obs(obs, i))
        for i, a in enumerate(activate_trainer.actor.actors)
    ]
    values0 = activate_trainer.critic(obs)
    activate_trainer.activate_inputs(dims)
    leaves1 = [
        a.tree.route(activate_trainer.actor.agent_obs(obs, i))
        for i, a in enumerate(activate_trainer.actor.actors)
    ]
    check(
        all(torch.equal(x, y) for x, y in zip(leaves0, leaves1))
        and torch.allclose(values0, activate_trainer.critic(obs))
        and float(activate_trainer.obs_norm.count[dims[0]]) < 1.0,
        "activate_inputs: hard routing and critic unchanged on zero channels, their normalisation reset",
    )
    rms = RunningMeanStd(4)
    rms.load_state_dict({"mean": torch.zeros(4), "var": torch.ones(4), "count": torch.tensor(7.0)})
    check(rms.count.shape == (4,) and float(rms.count[2]) == 7.0, "scalar obs-norm count loads per dimension")
    print("all checks passed")


if __name__ == "__main__":
    main()
