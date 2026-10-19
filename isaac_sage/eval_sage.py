#!/usr/bin/env python3
"""Deterministic deployment evaluation of a SAGE-family checkpoint (Algorithm 2).

Each of ``--num_episodes`` parallel environments contributes its FIRST finished
episode (no episode-length bias).  At every high-level step the deployed hard
tree produces a_nom(o) (Eq. 25); with the shield on the executed command is
a_exec = Q_CBF(x, a_nom) (Eq. 22), with it off a_exec = a_nom.  In both modes
the rows are built and Q_CBF(x, a_nom) is solved, so the shield-dependence
diagnostics are comparable across every arm, including the unshielded tree.

Per episode
    task      success (run's success_mode: loose = final path point reached, tight = arrive
              and hold at the goal pose; both are reported), completion = wp / (#wp),
              collision high-level steps (payload OBB + robot discs vs walls),
              drop (payload z < board_drop_height), fallen (any robot), timeout,
              payload tilt / roll / pitch
    safety    pviol_nominal       fraction of steps where a_nom violates >= 1 row
              pviol_<family>      same, per row family (collision / grasp / rotation / workspace)
              intervention_rate   fraction of steps with ||a_exec - a_nom|| > 1e-4
              correction          mean ||a_exec - a_nom||
              nominal_correction  mean ||Q_CBF(x, a_nom) - a_nom||   (shield-dependence of the tree)
              qp_infeasible       fraction of steps whose QP residual exceeds qp_tolerance
              rows_active         mean number of active rows at the QP solution
              min_h               smallest barrier value seen in the episode
Per run
    leaf occupancy of every carrier tree (fraction of steps routed to each leaf)

Usage
    python -m isaac_sage.eval_sage --checkpoint <run>/best_model.pt --num_episodes 30 --headless
    python -m isaac_sage.eval_sage --checkpoint <ckpt> --shield off ...        # nominal tree only
    python -m isaac_sage.eval_sage --checkpoint <ckpt> --scene_json scene.json # custom scene

Environment toggles (``--mask_lidar`` etc.) are read from ``run_args.json``
written by ``train_sage.py`` next to the checkpoint, so the evaluation
environment matches the training one; explicit flags override it.

Scene JSON format (env-local metres)
    {"name": "...",
     "obstacle_specs": [[[sx, sy, sz], [cx, cy, cz]], ...],   # box size, box centre
     "path_waypoints": [[x, y], ...],
     "start_xy": [x, y],
     "env_spacing": 20.0,              # optional
     "episode_length_s": 300.0}        # optional
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from isaac_sage.isaac_env import setup_isaaclab_path  # noqa: E402

setup_isaaclab_path()

from isaaclab.app import AppLauncher  # noqa: E402

from isaac_sage.isaac_env import TASK_OPTIONS  # noqa: E402

H_CAP = 5.0   # reported min_h when no row is active (rows beyond activation_distance are dropped)
ENV_TOGGLES = ("mask_lidar", "no_obstacles", "no_fixed_path", "target_radius", "curriculum",
               "w_heading", "w_obstacle", "gait_frequency", "g1_usd_path", "low_level_policy_path") + TASK_OPTIONS

parser = argparse.ArgumentParser(description="Deployment evaluation of a SAGE checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-G1-CooperativeTransport-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=30)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--shield", type=str, default="auto", choices=["auto", "on", "off"],
                    help="auto: off for the unshielded 'tree' arm, on for every other arm.")
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="Default: scene budget if --scene_json is given, else 300 s.")
parser.add_argument("--scene_json", type=str, default=None)
parser.add_argument("--fixed_spawn", action="store_true",
                    help="Ignore the scene's spawn_pool and always start from start_xy.")
parser.add_argument("--run_args", type=str, default=None,
                    help="Training run_args.json (default: searched next to the checkpoint).")
parser.add_argument("--out", type=str, default=None, help="JSON output (summary + per-episode rows).")
parser.add_argument("--violation_tol", type=float, default=1e-4)
parser.add_argument("--soft_tau", type=float, default=None,
                    help="Diagnostic for checkpoints still in the soft-routing phase: a_nom is the soft "
                         "path-probability mixture at this gate temperature instead of hard routing.")
parser.add_argument("--mask_lidar", action="store_true", default=None)
parser.add_argument("--no_obstacles", action="store_true", default=None)
parser.add_argument("--no_fixed_path", action="store_true", default=None)
parser.add_argument("--target_radius", type=float, default=None)
parser.add_argument("--curriculum", action="store_true", default=None)
parser.add_argument("--w_heading", type=float, default=None)
parser.add_argument("--w_obstacle", type=float, default=None)
parser.add_argument("--gait_frequency", type=float, default=None)
parser.add_argument("--g1_usd_path", type=str, default=None)
parser.add_argument("--low_level_policy_path", type=str, default=None)
parser.add_argument("--projection_progress", action="store_true", default=None)
parser.add_argument("--success_mode", type=str, default=None, choices=["loose", "tight"])
parser.add_argument("--partner_noise", action="store_true", default=None)
parser.add_argument("--partner_noise_lin", type=float, default=None)
parser.add_argument("--partner_noise_yaw_deg", type=float, default=None)
parser.add_argument("--partner_gain_lin", type=float, default=None)
parser.add_argument("--partner_gain_yaw", type=float, default=None)
parser.add_argument("--success_compensation", action="store_true", default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402

print(f"[SAGE] isaaclab from {isaaclab.__file__}")
from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from isaac_sage.isaac_env import (  # noqa: E402
    IsaacCoopTransportAdapter, apply_task_options, configure_task_cfg, episode_flags, randomize_spawn,
    register_task)
from isaac_sage.safety import FAMILY_NAMES, FormationSafetyModel, SafetyConfig  # noqa: E402
from isaac_sage.trainer import RunningMeanStd, deploy_action  # noqa: E402
from isaac_sage.tree_policy import TeamActor  # noqa: E402

register_task()
from isaac_sage.task.cooperative_transport_env import (  # noqa: E402
    ALL_ROBOT_IDS, BOARD_LENGTH, BOARD_THICKNESS, BOARD_WIDTH, _transform_points)


# ---------------------------------------------------------------------- #
def _resolve_env_toggles() -> dict:
    """Training toggles from run_args.json, overridden by explicit CLI flags."""
    toggles: dict = {}
    cands = [Path(args_cli.run_args)] if args_cli.run_args else [
        Path(args_cli.checkpoint).resolve().parent / "run_args.json",
        Path(args_cli.checkpoint).resolve().parent.parent / "run_args.json"]
    for p in cands:
        if p.is_file():
            run = json.loads(p.read_text())
            toggles = {k: run.get(k) for k in ENV_TOGGLES if run.get(k) not in (None, False)}
            print(f"[eval] env toggles from {p}: {toggles}")
            break
    else:
        print("[eval] no run_args.json found next to the checkpoint; using CLI toggles only")
    for k in ENV_TOGGLES:
        v = getattr(args_cli, k)
        if v is not None:
            toggles[k] = v
    return toggles


def _configure_env_cfg(cfg, toggles: dict, scene: dict | None):
    if toggles.get("mask_lidar"):
        cfg.mask_lidar_obs = True
    if toggles.get("no_obstacles"):
        cfg.obstacle_specs = []
    if toggles.get("no_fixed_path"):
        cfg.use_fixed_path = False
    if toggles.get("target_radius") is not None:
        cfg.target_radius = toggles["target_radius"]
        if not toggles.get("curriculum"):
            cfg.use_target_radius_curriculum = False
    elif not toggles.get("curriculum") and toggles.get("no_fixed_path"):
        cfg.use_target_radius_curriculum = False
    if toggles.get("w_heading") is not None:
        cfg.nrew_w_heading = toggles["w_heading"]
    if toggles.get("w_obstacle") is not None:
        cfg.nrew_w_obstacle = toggles["w_obstacle"]
    if toggles.get("gait_frequency") is not None:
        cfg.gait_frequency = toggles["gait_frequency"]
    if toggles.get("g1_usd_path"):
        cfg.training_g1_usd_path = toggles["g1_usd_path"]
    if toggles.get("low_level_policy_path"):
        cfg.low_level_policy_path = toggles["low_level_policy_path"]
    if scene is not None:
        cfg.obstacle_specs = [(tuple(s), tuple(p)) for s, p in scene["obstacle_specs"]]
        cfg.path_waypoints = [tuple(w) for w in scene["path_waypoints"]]
        cfg.start_xy = tuple(scene.get("start_xy", scene["path_waypoints"][0]))
        cfg.use_fixed_path = True
        if "env_spacing" in scene:
            cfg.scene.env_spacing = float(scene["env_spacing"])
        if "episode_length_s" in scene:
            cfg.episode_length_s = float(scene["episode_length_s"])
    if args_cli.episode_length_s is not None:
        cfg.episode_length_s = args_cli.episode_length_s
    elif scene is None or "episode_length_s" not in scene:
        cfg.episode_length_s = 300.0
    return cfg


@torch.no_grad()
def soft_action(actor, safety, obs_norm, obs_raw, state, tau: float):
    """deploy_action with the soft-routing nominal (most probable leaf reported)."""
    obs = obs_norm.normalize(obs_raw)
    acts, leaves = [], []
    for i, a in enumerate(actor.actors):
        a_i, tinfo = a.tree(actor.obs_i(obs, i), tau, hard=False)
        acts.append(a_i)
        leaves.append(tinfo["leaf"])
    a_nom = torch.cat(acts, -1)
    rows = safety.rows(state)
    a_exec, qp = safety.exact_projection(a_nom, rows)
    return a_exec, {"a_nom": a_nom, "leaves": torch.stack(leaves, -1), "rows": rows,
                    "violated": rows.violated(a_nom), **qp}


def _load_checkpoint(path: str, adapter, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    for k, cur in (("n_agents", adapter.n_agents), ("obs_dim_per", adapter.obs_dim_per),
                   ("act_dim_per", adapter.act_dim_per)):
        if int(ck[k]) != int(cur):
            raise ValueError(f"checkpoint {k}={ck[k]} != env {k}={cur}")
    cfg = ck["cfg"]
    actor = TeamActor(adapter.obs_dim_per, adapter.act_dim_per, adapter.n_agents,
                      int(cfg["tree_depth"]), float(cfg.get("log_std_init", -0.5))).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    obs_norm = RunningMeanStd(adapter.obs_dim_per * adapter.n_agents).to(device)
    obs_norm.load_state_dict(ck["obs_norm"])
    known = {f.name for f in fields(SafetyConfig)}
    safety_cfg = SafetyConfig(**{k: v for k, v in dict(cfg.get("safety", {})).items() if k in known})
    return actor, obs_norm, FormationSafetyModel(safety_cfg), cfg


# ---------------------------------------------------------------------- #
def _board_tilt_m(env) -> np.ndarray:
    corners = torch.tensor([[sx * BOARD_LENGTH / 2, sy * BOARD_WIDTH / 2, BOARD_THICKNESS / 2]
                            for sx in (1, -1) for sy in (1, -1)], device=env.device, dtype=torch.float32)
    z = _transform_points(corners, env._board.data.root_pos_w, env._board.data.root_quat_w)[:, :, 2]
    return (z - z.mean(dim=-1, keepdim=True)).abs().sum(dim=-1).cpu().numpy()


def _board_roll_pitch_deg(env) -> tuple[np.ndarray, np.ndarray]:
    g_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    g_b = quat_rotate_inverse(env._board.data.root_quat_w, g_w)
    pitch = torch.asin(g_b[:, 0].clamp(-1.0, 1.0)) * (180.0 / math.pi)
    roll = torch.asin(g_b[:, 1].clamp(-1.0, 1.0)) * (180.0 / math.pi)
    return roll.cpu().numpy(), pitch.cpu().numpy()


def _any_fallen(env) -> np.ndarray:
    out = np.zeros(env.num_envs, dtype=bool)
    g_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    for aid in ALL_ROBOT_IDS:
        rob = env.robots[aid]
        low = rob.data.root_pos_w[:, 2] < env.cfg.termination_height
        g_b = quat_rotate_inverse(rob.data.root_quat_w, g_w)
        out |= (low | (g_b[:, 0].abs() > 0.7) | (g_b[:, 1].abs() > 0.7)).cpu().numpy()
    return out


def _fail_reason(row: dict) -> str:
    if row["success"]:
        return "success"
    if row["drop"]:
        return "drop"
    if row["fallen"]:
        return "fallen"
    if row["timeout"]:
        return "timeout"
    return "other"


class EpisodeStats:
    """Per-env accumulators for the running episode; flushed on reset."""

    PHYS = ("col_steps", "tilt_sum", "tilt_max", "roll_max", "pitch_max", "n_hl")
    SAFE = ("n_safe", "pviol", "interv", "corr", "nomcorr", "infeas", "active") + tuple(
        f"pviol_{f}" for f in FAMILY_NAMES)

    def __init__(self, n: int):
        self.n = n
        for k in self.PHYS + self.SAFE:
            setattr(self, k, np.zeros(n, dtype=np.float64))
        self.z_min = np.full(n, np.inf)
        self.min_h = np.full(n, H_CAP)
        self.dropped = np.zeros(n, dtype=bool)
        self.fallen = np.zeros(n, dtype=bool)

    def clear(self, i: int) -> None:
        for k in self.PHYS + self.SAFE:
            getattr(self, k)[i] = 0.0
        self.z_min[i] = np.inf
        self.min_h[i] = H_CAP
        self.dropped[i] = self.fallen[i] = False


# ---------------------------------------------------------------------- #
def main():
    toggles = _resolve_env_toggles()
    scene = json.loads(Path(args_cli.scene_json).read_text()) if args_cli.scene_json else None
    n_eps = args_cli.num_episodes
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n_eps,
                            use_fabric=not getattr(args_cli, "disable_fabric", False))
    env_cfg.seed = args_cli.seed
    env_cfg = _configure_env_cfg(env_cfg, toggles, scene)
    task_opts = {k: toggles.get(k) for k in TASK_OPTIONS}
    configure_task_cfg(env_cfg, task_opts)
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = gym_env.unwrapped
    spawn_pool = scene.get("spawn_pool") if scene and not args_cli.fixed_spawn else None
    if spawn_pool:
        randomize_spawn(env, spawn_pool)
    for msg in apply_task_options(env, task_opts, scene):
        print(f"[eval] {msg}")
    adapter = IsaacCoopTransportAdapter(env)
    if task_opts.get("partner_noise"):
        kw = {k: task_opts[f"partner_{k_}"] for k, k_ in (("lin", "noise_lin"), ("yaw_deg", "noise_yaw_deg"),
                                                          ("gain_lin", "gain_lin"), ("gain_yaw", "gain_yaw"))
              if task_opts.get(f"partner_{k_}") is not None}
        adapter.set_partner_perturbation(**kw)
        print(f"[eval] partner perturbation on {adapter.agent_ids[1]}: {adapter._partner}")
    device = adapter.device
    actor, obs_norm, safety, ck_cfg = _load_checkpoint(args_cli.checkpoint, adapter, device)
    variant = str(ck_cfg.get("variant", "?"))
    shield = (variant != "tree") if args_cli.shield == "auto" else (args_cli.shield == "on")
    scene_name = scene.get("name", Path(args_cli.scene_json).stem) if scene else "training_maze"
    n_path_wp = max(int(env._path.shape[0]) - 1, 1)
    max_len = int(env.max_episode_length)
    drop_h = float(env.cfg.board_drop_height)
    if args_cli.soft_tau is not None:
        print(f"[eval] soft routing at tau={args_cli.soft_tau} (not the deployed hard tree)")
    print(f"[eval] variant={variant} shield={'on' if shield else 'off'} scene={scene_name} "
          f"episodes={n_eps} waypoints={n_path_wp} budget={env_cfg.episode_length_s}s "
          f"spawn={'randomised' if spawn_pool else 'fixed'}")

    S = EpisodeStats(env.num_envs)
    records: list[dict] = []
    recording = False
    orig_reset_idx = env._reset_idx

    def acc_phys(ids=None):
        sel = np.arange(env.num_envs) if ids is None else np.asarray(ids, dtype=int)
        tilt = _board_tilt_m(env)
        roll, pitch = _board_roll_pitch_deg(env)
        z = env._board.data.root_pos_w[:, 2].cpu().numpy()
        fallen = _any_fallen(env)
        hit = env._compute_obstacle_collision().cpu().numpy()
        S.tilt_sum[sel] += tilt[sel]
        S.tilt_max[sel] = np.maximum(S.tilt_max[sel], tilt[sel])
        S.roll_max[sel] = np.maximum(S.roll_max[sel], np.abs(roll[sel]))
        S.pitch_max[sel] = np.maximum(S.pitch_max[sel], np.abs(pitch[sel]))
        S.z_min[sel] = np.minimum(S.z_min[sel], z[sel])
        S.n_hl[sel] += 1
        S.dropped[sel] |= z[sel] < drop_h
        S.fallen[sel] |= fallen[sel]
        S.col_steps[sel] += hit[sel].astype(np.float64)

    def patched_reset_idx(env_ids):
        if recording:
            ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
            acc_phys(ids)   # terminal frame is still live before the env is re-initialised
            succ_flag, loose_flag = episode_flags(env)
            for i in ids:
                steps = int(env.episode_length_buf[i].item())
                nh, ns = max(S.n_hl[i], 1.0), max(S.n_safe[i], 1.0)
                rec = {
                    "env": int(i),
                    "wp": int(env._wp_reached_count[i].item()),
                    "success": bool(succ_flag[i].item()),
                    "success_loose": bool(loose_flag[i].item()),
                    "steps": steps,
                    "timeout": steps >= max_len,
                    "collision_hl_steps": int(S.col_steps[i]),
                    "drop": bool(S.dropped[i]),
                    "fallen": bool(S.fallen[i]),
                    "tilt_mean_m": float(S.tilt_sum[i] / nh),
                    "tilt_max_m": float(S.tilt_max[i]),
                    "roll_abs_max_deg": float(S.roll_max[i]),
                    "pitch_abs_max_deg": float(S.pitch_max[i]),
                    "board_z_min": float(S.z_min[i]),
                    "pviol_nominal": float(S.pviol[i] / ns),
                    "intervention_rate": float(S.interv[i] / ns),
                    "correction": float(S.corr[i] / ns),
                    "nominal_correction": float(S.nomcorr[i] / ns),
                    "qp_infeasible": float(S.infeas[i] / ns),
                    "rows_active": float(S.active[i] / ns),
                    "min_h": float(S.min_h[i]),
                }
                for f in FAMILY_NAMES:
                    rec[f"pviol_{f}"] = float(getattr(S, f"pviol_{f}")[i] / ns)
                rec["completion"] = 1.0 if rec["success"] else min(rec["wp"] / n_path_wp, 1.0)
                rec["reason"] = _fail_reason(rec)
                records.append(rec)
                S.clear(i)
        return orig_reset_idx(env_ids)

    env._reset_idx = patched_reset_idx

    obs_raw = adapter.reset()
    recording = True
    first: dict[int, dict] = {}
    done_first = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    leaf_counts = torch.zeros(adapter.n_agents, actor.actors[0].tree.num_leaves, device=device)
    tol = args_cli.violation_tol

    for step in range(max_len + 5):
        state = adapter.safety_state()
        if args_cli.soft_tau is None:
            a_shielded, info = deploy_action(actor, safety, obs_norm, obs_raw, state)
        else:
            a_shielded, info = soft_action(actor, safety, obs_norm, obs_raw, state, args_cli.soft_tau)
        a_nom = info["a_nom"]
        a_exec = a_shielded if shield else a_nom
        with torch.no_grad():
            rows = info["rows"]
            viol_rows = rows.margins(a_nom) < -tol
            nomcorr = (a_shielded - a_nom).norm(dim=-1)
            corr = (a_exec - a_nom).norm(dim=-1)
            h_min = rows.h.masked_fill(rows.mask <= 0, H_CAP).min(dim=1).values.clamp(max=H_CAP)
            live = ~done_first
            for n in range(adapter.n_agents):
                leaf_counts[n] += torch.bincount(info["leaves"][live, n],
                                                 minlength=leaf_counts.shape[1]).float()
        S.n_safe += 1.0
        S.pviol += viol_rows.any(1).cpu().numpy()
        for fi, f in enumerate(FAMILY_NAMES):
            getattr(S, f"pviol_{f}")[:] += (viol_rows & (rows.family == fi)).any(1).cpu().numpy()
        S.interv += (corr > 1e-4).cpu().numpy()
        S.corr += corr.cpu().numpy()
        S.nomcorr += nomcorr.cpu().numpy()
        S.infeas += (~info["feasible"]).cpu().numpy()
        S.active += info["active"].float().sum(1).cpu().numpy()
        S.min_h = np.minimum(S.min_h, h_min.cpu().numpy())

        obs_raw, _r, _done, _log = adapter.step(a_exec)
        acc_phys()
        for r in records:
            if r["env"] not in first:
                first[r["env"]] = r
                done_first[r["env"]] = True
        records.clear()
        if len(first) >= n_eps:
            break
        if (step + 1) % 50 == 0:
            print(f"[eval] hl-step {step + 1}: {len(first)}/{n_eps} episodes finished", flush=True)

    rows_out = [first[k] for k in sorted(first)]
    if not rows_out:
        raise RuntimeError("no episode finished within the step budget")

    def arr(k):
        return np.array([float(r[k]) for r in rows_out], dtype=np.float64)

    def ms(k):
        a = arr(k)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}

    summary = {
        "checkpoint": str(args_cli.checkpoint), "variant": variant, "shield": shield,
        "routing": "hard" if args_cli.soft_tau is None else f"soft(tau={args_cli.soft_tau})",
        "scene": scene_name, "episodes": len(rows_out), "n_path_wp": n_path_wp,
        "seed": args_cli.seed, "env_toggles": toggles, "spawn": "randomised" if spawn_pool else "fixed",
        "success_mode": task_opts.get("success_mode") or "loose",
        "success_rate": float(arr("success").mean()),
        "success_rate_loose": float(arr("success_loose").mean()),
        "completion": ms("completion"),
        "collision_episode_rate": float((arr("collision_hl_steps") > 0).mean()),
        "collision_hl_steps": ms("collision_hl_steps"),
        "drop_episode_rate": float(arr("drop").mean()),
        "fallen_episode_rate": float(arr("fallen").mean()),
        "timeout_episode_rate": float(arr("timeout").mean()),
        "fail_split": {k: sum(1 for r in rows_out if r["reason"] == k)
                       for k in ("success", "drop", "fallen", "timeout", "other")},
        "tilt_mean_m": ms("tilt_mean_m"), "tilt_max_m": ms("tilt_max_m"),
        "steps": ms("steps"),
    }
    for k in ("pviol_nominal", "intervention_rate", "correction", "nominal_correction",
              "qp_infeasible", "rows_active", "min_h") + tuple(f"pviol_{f}" for f in FAMILY_NAMES):
        summary[k] = ms(k)
    occ = leaf_counts / leaf_counts.sum(dim=1, keepdim=True).clamp(min=1.0)
    summary["leaf_occupancy"] = {f"carrier_{n}": [round(float(v), 4) for v in occ[n]]
                                 for n in range(adapter.n_agents)}
    summary["leaves_used"] = {f"carrier_{n}": int((occ[n] > 0).sum()) for n in range(adapter.n_agents)}

    print("\n===== SAGE deployment evaluation =====")
    print(f"checkpoint: {args_cli.checkpoint}")
    print(f"variant: {variant} | shield: {'on' if shield else 'off'} | scene: {scene_name} | "
          f"episodes: {len(rows_out)} | seed: {args_cli.seed}")
    print(f"success rate ({summary['success_mode']}): {summary['success_rate']:.3f}   "
          f"loose: {summary['success_rate_loose']:.3f}   completion: {summary['completion']['mean']:.3f}"
          f" ± {summary['completion']['std']:.3f}")
    print(f"collision ep rate: {summary['collision_episode_rate']:.3f}   drop: {summary['drop_episode_rate']:.3f}"
          f"   fallen: {summary['fallen_episode_rate']:.3f}   timeout: {summary['timeout_episode_rate']:.3f}")
    print(f"fail split: {summary['fail_split']}")
    print(f"PViol(a_nom): {summary['pviol_nominal']['mean']:.3f}   "
          + "  ".join(f"{f}={summary[f'pviol_{f}']['mean']:.3f}" for f in FAMILY_NAMES))
    print(f"||Q(a_nom)-a_nom||: {summary['nominal_correction']['mean']:.4f}   "
          f"||a_exec-a_nom||: {summary['correction']['mean']:.4f}   "
          f"intervention rate: {summary['intervention_rate']['mean']:.3f}")
    print(f"QP infeasible: {summary['qp_infeasible']['mean']:.3f}   active rows: {summary['rows_active']['mean']:.2f}"
          f"   min h: {summary['min_h']['mean']:.3f}")
    print(f"tilt mean/max (m): {summary['tilt_mean_m']['mean']:.3f} / {summary['tilt_max_m']['mean']:.3f}")
    print(f"leaves used: {summary['leaves_used']}")
    print("======================================")

    if args_cli.out:
        out = Path(args_cli.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "episodes": rows_out}, indent=2))
        print(f"[eval] wrote {out}")
    gym_env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()   # simulation_app.close() exits the process without flushing
    simulation_app.close()
