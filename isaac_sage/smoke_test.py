#!/usr/bin/env python3
"""CPU smoke test for the SAGE reference implementation (no simulator needed).

    python -m isaac_sage.smoke_test

Checks, each tied to a paper statement:
  1. affine rows are satisfied by the zero action whenever h >= 0 (Eq. 10, c = -eta h)
  2. exact QP output satisfies every active row and is idempotent on feasible inputs (Eq. 18)
  3. fixed (measured) block is untouched by the partner-aware QP (Eq. 19)
  4. SAIL vs exact QP sensitivity for one isolated active row:
        d a_bar / d a_ST along the row normal = 1/(1+kappa)  (Eq. 29),  exact QP = 0 (Eq. 28)
  5. hard-forward ST tree equals the deployed hard routing; gates receive gradients (Eq. 23-26)
  6. TALO field satisfies the linearised descent condition grad W^T g + sigma W <= 0 (Eq. 41)
  7. two training iterations of every variant on a planar kinematic stand-in environment
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaac_sage.safety import FormationSafetyModel, RowSet, SafetyConfig, SafetyState, rot2  # noqa: E402
from isaac_sage.sail import sail_penalty_map  # noqa: E402
from isaac_sage.talo import talo_update_field  # noqa: E402
from isaac_sage.trainer import SAGEConfig, SAGETrainer  # noqa: E402
from isaac_sage.tree_policy import TeamActor  # noqa: E402

torch.set_default_dtype(torch.float32)
DEV = torch.device("cpu")


class PlanarFormationEnv:
    """Kinematic stand-in with the Isaac task's geometry and action semantics:
    3 robots around a 2.0 x 1.0 board, 2 controlled carriers at the short edges,
    body-frame [dx, dy, dyaw] commands scaled by (0.2, 0.2, 0.3), env-local wall boxes."""

    def __init__(self, num_envs=16, obs_dim_per=12, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.g = g
        self.num_envs, self.n_agents, self.obs_dim_per, self.act_dim_per = num_envs, 2, obs_dim_per, 3
        self.device = DEV
        self.action_scale = torch.tensor([0.2, 0.2, 0.3])
        self.anchors_local = torch.tensor([[1.2, 0.0], [-1.2, 0.0], [0.0, 0.7]])
        self.controlled = torch.tensor([0, 1])
        # corridor walls y = +-1.2 .. and a pillar ahead
        self.aabbs = torch.tensor([[-3.0, 6.0, 1.3, 1.8], [-3.0, 6.0, -1.8, -1.3], [2.5, 3.0, -0.4, 0.4]])
        self.goal = torch.tensor([5.0, 0.0])
        self.t = 0

    def _obs(self):
        rel = self.goal - self.board_xy
        feats = [rel, torch.stack((torch.cos(self.board_yaw), torch.sin(self.board_yaw)), -1),
                 self.board_xy]
        core = torch.cat(feats, -1)
        pad = torch.zeros(self.num_envs, self.obs_dim_per - core.shape[1])
        o = torch.cat((core, pad), -1)
        return torch.cat((o, o), -1)

    def reset(self):
        self.board_xy = torch.zeros(self.num_envs, 2)
        self.board_xy[:, 1] = torch.rand(self.num_envs, generator=self.g) * 0.6 - 0.3
        self.board_yaw = torch.zeros(self.num_envs)
        self.t = 0
        return self._obs()

    def _robot_pose(self):
        R = rot2(self.board_yaw)
        xy = self.board_xy[:, None] + torch.einsum("bij,rj->bri", R, self.anchors_local)
        yaw = self.board_yaw[:, None] + torch.tensor([math.pi, 0.0, -math.pi / 2])
        return xy, yaw

    def step(self, a_norm):
        rxy, ryaw = self._robot_pose()
        cmd = a_norm.reshape(self.num_envs, 2, 3).clamp(-1, 1) * self.action_scale
        d = torch.einsum("bnij,bnj->bni", rot2(ryaw[:, :2].reshape(-1)).reshape(self.num_envs, 2, 2, 2), cmd[..., :2])
        v = d.mean(1)
        self.board_xy = self.board_xy + v
        self.board_yaw = self.board_yaw + 0.3 * cmd[..., 2].mean(1)
        self.t += 1
        dist = (self.goal - self.board_xy).norm(dim=-1)
        r = -0.1 * dist + 2.0 * (dist < 0.5).float()
        done = (dist < 0.5) | torch.full((self.num_envs,), self.t >= 40)
        if done.any():
            self.board_xy[done] = 0.0
            self.board_yaw[done] = 0.0
        if self.t >= 40:
            self.t = 0
        return self._obs(), r, done, {"dist": float(dist.mean())}

    def safety_state(self):
        rxy, ryaw = self._robot_pose()
        return SafetyState(board_xy=self.board_xy, board_yaw=self.board_yaw, robot_xy=rxy, robot_yaw=ryaw,
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
    st = env.safety_state()
    rows = model.rows(st)
    da = rows.C.shape[-1]
    print(f"rows: K={rows.C.shape[1]} present/env={rows.mask.sum(1).mean():.1f} min h={rows.h[rows.mask > 0].min():.3f}")

    # 1. zero action feasible when h >= 0
    zero = torch.zeros(env.num_envs, da)
    check(bool((rows.margins(zero) >= -1e-6).all()), "zero action satisfies every row (h >= 0 => c <= 0)")

    # 2. exact QP: feasibility + idempotence
    a = torch.rand(env.num_envs, da) * 2 - 1
    a[:, 0] = -1.0  # push carrier 0 hard toward the board along its body x -> grasp rows bite
    out, info = model.exact_projection(a, rows)
    check(bool(info["feasible"].all()) and float(rows.margins(out).min()) > -model.cfg.qp_tolerance,
          f"projected action satisfies rows (min residual {float(rows.margins(out).min()):+.4f})")
    out2, _ = model.exact_projection(out, rows)
    check(float((out2 - out).abs().max()) < 1e-3, "projection is idempotent on feasible inputs")
    check(float((out - a).norm(dim=-1).mean()) > 0, "at least one input was corrected")

    # 3. fixed human block
    fixed = torch.zeros(da, dtype=torch.bool)
    fixed[3:6] = True
    out_h, _ = model.exact_projection(a, rows, fixed=fixed)
    check(bool(torch.allclose(out_h[:, 3:6], a[:, 3:6])), "measured partner block held fixed (Eq. 19)")

    # 4. SAIL vs exact QP sensitivity (Eq. 28 / 29) with one isolated active row  a_0 >= 0.5
    kappa = 8.0
    C = torch.zeros(1, 1, da); C[0, 0, 0] = 1.0
    c = torch.full((1, 1), 0.5)
    one_row = RowSet(C, c, torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.long))
    a_st = torch.zeros(1, da, requires_grad=True)
    a_bar = sail_penalty_map(a_st, one_row, kappa, newton_steps=8, damping=1.0)
    (jac,) = torch.autograd.grad(a_bar[0, 0], a_st)
    expect = 1.0 / (1.0 + kappa)
    check(abs(float(jac[0, 0]) - expect) < 1e-4, f"SAIL normal gain {float(jac[0, 0]):.4f} == 1/(1+kappa) = {expect:.4f}")
    check(abs(float(a_bar[0, 0]) - kappa * 0.5 / (1 + kappa)) < 1e-4, "SAIL minimiser matches closed form")
    p0, _ = model.exact_projection(torch.zeros(1, da), one_row)
    p1, _ = model.exact_projection(torch.zeros(1, da).index_fill(1, torch.tensor([0]), 0.1), one_row)
    check(abs(float(p1[0, 0] - p0[0, 0])) < 1e-3, "exact QP normal gain == 0 (projection annihilates it)")
    a_big = sail_penalty_map(torch.zeros(1, da), one_row, 1e4, newton_steps=8, damping=1.0)
    check(abs(float(a_big[0, 0]) - 0.5) < 1e-3, "kappa -> inf recovers the exact projection")

    # 5. tree: hard forward == deployed routing, gates get gradient
    actor = TeamActor(env.obs_dim_per, 3, 2, depth=3)
    o = torch.randn(32, 2 * env.obs_dim_per)
    a_hard, _ = actor.nominal(o)
    a_st_full = actor.nominal_st(o, tau=0.5)
    check(bool(torch.allclose(a_hard, a_st_full.detach(), atol=1e-6)), "ST forward == deployed hard routing")
    a_st_full.sum().backward()
    check(actor.actors[0].tree.split_w.grad is not None and float(actor.actors[0].tree.split_w.grad.abs().sum()) > 0,
          "split hyperplanes receive straight-through gradients")
    a_det = actor.nominal_st(o, tau=0.5, detach_except=1)
    g = torch.autograd.grad(a_det.sum(), list(actor.actors[0].parameters()), allow_unused=True)
    check(all(x is None for x in g), "partner branch detached (detach_except) carries no gradient")
    a_soft, regs = actor.nominal_st(o, tau=2.0, hard=False, return_reg=True)
    tr0 = actor.actors[0].tree
    probs = tr0(actor.obs_i(o, 0), 2.0, hard=False)[1]["leaf_probs"]
    check(bool(torch.allclose(probs.sum(1), torch.ones(32), atol=1e-5)) and float(probs.max(1).values.min()) < 0.999,
          "soft routing: path probabilities sum to 1 and mix leaves")
    check(not torch.allclose(a_soft, a_hard, atol=1e-4), "soft-phase nominal differs from deployed hard routing")
    actor.zero_grad()
    regs[0].backward()
    check(float(tr0.split_w.grad.abs().sum()) > 0 and float(tr0.split_b.grad.abs().sum()) > 0
          and all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in actor.actors[1].parameters()),
          f"structure regulariser R={float(regs[0]):.3f} reaches own splits only")
    sch = SAGETrainer(PlanarFormationEnv(num_envs=4), SAGEConfig(tree_depth=3), DEV)
    phases = []
    for f in (0.0, 0.5, 0.84, 0.86, 1.0):
        sch.set_progress(f)
        phases.append((sch.hard, round(sch.tau, 3), round(sch.kappa, 1)))
    check([p[0] for p in phases] == [False, False, False, True, True] and phases[0][1] == 2.0
          and phases[-1][1] == 0.12 and phases[0][2] == 8.0 and phases[-1][2] == 32.0,
          f"schedule (hard, tau, kappa): {phases}")
    sw = SAGETrainer(PlanarFormationEnv(num_envs=8), SAGEConfig(variant="sage", tree_depth=3, num_steps=10,
                                                                 ppo_epochs=1, num_minibatches=2), DEV)
    sw.set_progress(0.5); sw.collect(); sw.update()
    lr0 = sw.opt_a.param_groups[0]["lr"]
    sw.set_progress(0.9)
    t0 = sw.actor.actors[0].tree
    before = (t0.split_w.detach().clone(), t0.split_b.detach().clone(), t0.leaf_K.detach().clone())
    sw.collect(); sw.update()
    check(torch.equal(before[0], t0.split_w) and torch.equal(before[1], t0.split_b)
          and not torch.equal(before[2], t0.leaf_K) and abs(sw.opt_a.param_groups[0]["lr"] - 0.5 * lr0) < 1e-12,
          "soft -> hard switch freezes routing, leaves keep training, actor lr halved once")
    torch.save(sw.checkpoint(), "/tmp/_sage_hard_ck.pt")
    sr = SAGETrainer(PlanarFormationEnv(num_envs=8), SAGEConfig(variant="sage", tree_depth=3, num_steps=10), DEV)
    sr.load("/tmp/_sage_hard_ck.pt", restore_progress=True)
    sr.set_progress(0.95)
    check(not sr.actor.actors[1].tree.split_w.requires_grad
          and abs(sr.opt_a.param_groups[0]["lr"] - 0.5 * lr0) < 1e-12,
          "resume in hard phase re-freezes routing without scaling lr twice")

    # 6. TALO descent condition
    p = [torch.nn.Parameter(torch.randn(5)), torch.nn.Parameter(torch.randn(3))]
    L_ind = [(p[0] ** 2).sum() + torch.sin(p[0]).sum(), (p[1] ** 3).sum()]
    L_team = (p[0] ** 2).sum() * 1.3 + (p[1] ** 2).sum() + (p[0].sum() * p[1].sum())
    field, stats = talo_update_field(L_ind, L_team, p, sigma=0.1, eps=1e-8, enabled=True)
    W = 0.5 * sum(((torch.autograd.grad(sum(L_ind), p, create_graph=True)[i]
                    - torch.autograd.grad(L_team, p, create_graph=True)[i]) ** 2).sum() for i in range(2))
    gradW = torch.autograd.grad(W, p)
    lhs = float(sum((h * f).sum() for h, f in zip(gradW, field)) + 0.1 * W)
    check(lhs <= 1e-5, f"grad W^T g_TALO + sigma W = {lhs:+.3e} <= 0 (Eq. 41)")
    L_ind = [(p[0] ** 2).sum() + torch.sin(p[0]).sum(), (p[1] ** 3).sum()]
    L_team = (p[0] ** 2).sum() * 1.3 + (p[1] ** 2).sum() + (p[0].sum() * p[1].sum())
    g_team0 = torch.autograd.grad(L_team, p, retain_graph=True)
    field0, stats0 = talo_update_field(L_ind, L_team, p, sigma=0.1, eps=1e-8, enabled=False)
    check(stats0["talo_lambda"] == 0.0 and all(torch.allclose(f, g) for f, g in zip(field0, g_team0)),
          "TALO disabled returns g_team")

    # 7. training iterations for every variant
    for variant in ("tree", "tree_cbf", "sail", "talo", "sage"):
        env = PlanarFormationEnv()
        cfg = SAGEConfig(variant=variant, tree_depth=3, num_steps=20, ppo_epochs=2, num_minibatches=4,
                         safety=SafetyConfig(activation_distance=3.0, max_rows=24))
        tr = SAGETrainer(env, cfg, DEV)
        for it in range(2):
            tr.set_progress(it / 2)
            m = tr.collect()
            u = tr.update()
        check(all(math.isfinite(v) for v in u.values()), f"variant {variant:8s}: finite update stats "
              f"(Wgap={u.get('W_gap', 0):.3g}, cos={u.get('cos_align', 0):+.2f}, lam={u.get('talo_lambda', 0):.3g}, "
              f"pviol={m.get('pviol_proposal', 0):.2f}, corr={m.get('correction', 0):.3f})")
        if variant == "sage":
            torch.save(tr.checkpoint(), "/tmp/_sage_team_critic_ck.pt")

    # 8. plain critic arm warm-started from a team-context checkpoint: actor loaded,
    #    critic fresh, actor frozen during the critic warm-up
    from isaac_sage.critic import JointCritic
    cfg = SAGEConfig(variant="tree_cbf", tree_depth=3, num_steps=20, ppo_epochs=2, num_minibatches=4,
                     critic_warmup_iters=2, safety=SafetyConfig(activation_distance=3.0, max_rows=24))
    tj = SAGETrainer(PlanarFormationEnv(), cfg, DEV)
    tj.load("/tmp/_sage_team_critic_ck.pt")
    snap = lambda: torch.cat([p.detach().flatten().clone() for p in tj.actor.parameters()])  # noqa: E731
    moved, vloss = [], []
    for it in range(3):
        before = snap()
        tj.set_progress(it / 3)
        tj.collect()
        u = tj.update()
        moved.append(bool((snap() - before).abs().max() > 0))
        vloss.append(round(u["value_loss"], 3))
    check(isinstance(tj.critic, JointCritic) and not any(k.startswith("encoder") for k in tj.critic.state_dict())
          and moved == [False, False, True] and all(math.isfinite(v) for v in vloss),
          f"HAPPO critic, no team context; actor frozen for {cfg.critic_warmup_iters} warm-up iterations "
          f"(actor moved {moved}, value loss {vloss})")
    torch.save(tj.checkpoint(), "/tmp/_tree_cbf_joint_ck.pt")
    tr2 = SAGETrainer(PlanarFormationEnv(), cfg, DEV)
    tr2.load("/tmp/_tree_cbf_joint_ck.pt", restore_progress=True)
    check(tr2._critic_warmup_left == 0 and torch.equal(tr2.value_norm.running_mean, tj.value_norm.running_mean),
          "resume restores the joint critic and ValueNorm without a second warm-up")
    # 9. activating never-trained input channels keeps the deployed policy and critic
    from isaac_sage.trainer import RunningMeanStd
    ta = SAGETrainer(PlanarFormationEnv(), SAGEConfig(variant="sage", tree_depth=3, num_steps=10), DEV)
    dims = [0, 1]
    o = torch.randn(512, ta.obs_dim)
    for i in range(ta.N):
        o[:, i * ta.do + torch.tensor(dims)] = 0.0
    leaves0 = [a.tree.route(ta.actor.obs_i(o, i)) for i, a in enumerate(ta.actor.actors)]
    v0 = ta.critic(o)
    ta.activate_inputs(dims)
    leaves1 = [a.tree.route(ta.actor.obs_i(o, i)) for i, a in enumerate(ta.actor.actors)]
    check(all(torch.equal(x, y) for x, y in zip(leaves0, leaves1)) and torch.allclose(v0, ta.critic(o))
          and float(ta.obs_norm.count[dims[0]]) < 1.0,
          "activate_inputs: hard routing and critic unchanged on zero channels, their normalisation reset")
    rms = RunningMeanStd(4)
    rms.load_state_dict({"mean": torch.zeros(4), "var": torch.ones(4), "count": torch.tensor(7.0)})
    check(rms.count.shape == (4,) and float(rms.count[2]) == 7.0, "scalar obs-norm count loads per dimension")
    print("all checks passed")


if __name__ == "__main__":
    main()
