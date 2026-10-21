#!/usr/bin/env python3
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

from isaac_sage.isaac_env import setup_isaaclab_path

setup_isaaclab_path()

from isaaclab.app import AppLauncher

from isaac_sage.isaac_env import TASK_OPTIONS

H_CAP = 5.0
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

import gymnasium as gym
import isaaclab
import numpy as np
import torch

import isaaclab_tasks

print(f"[SAGE] isaaclab from {isaaclab.__file__}")
from isaaclab.utils.math import quat_rotate_inverse
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaac_sage.isaac_env import (
    IsaacCoopTransportAdapter, apply_task_options, configure_task_cfg, episode_flags, randomize_spawn,
    register_task)
from isaac_sage.safety import FAMILY_NAMES, FormationSafetyModel, SafetyConfig
from isaac_sage.trainer import RunningMeanStd, deploy_action
from isaac_sage.tree_policy import TeamActor

register_task()
from isaac_sage.task.cooperative_transport_env import (
    ALL_ROBOT_IDS, BOARD_LENGTH, BOARD_THICKNESS, BOARD_WIDTH, _transform_points)


def _resolve_env_toggles() -> dict:
    toggles: dict = {}
    candidates = [Path(args_cli.run_args)] if args_cli.run_args else [
        Path(args_cli.checkpoint).resolve().parent / "run_args.json",
        Path(args_cli.checkpoint).resolve().parent.parent / "run_args.json"]
    for path in candidates:
        if path.is_file():
            run = json.loads(path.read_text())
            toggles = {key: run.get(key) for key in ENV_TOGGLES if run.get(key) not in (None, False)}
            print(f"[eval] env toggles from {path}: {toggles}")
            break
    else:
        print("[eval] no run_args.json found next to the checkpoint; using CLI toggles only")
    for key in ENV_TOGGLES:
        value = getattr(args_cli, key)
        if value is not None:
            toggles[key] = value
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
        cfg.obstacle_specs = [(tuple(size), tuple(pos)) for size, pos in scene["obstacle_specs"]]
        cfg.path_waypoints = [tuple(waypoint) for waypoint in scene["path_waypoints"]]
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
def soft_action(actor, safety, obs_norm, raw_obs, state, tau: float):
    obs = obs_norm.normalize(raw_obs)
    actions, leaves = [], []
    for i, carrier in enumerate(actor.actors):
        action, tree_info = carrier.tree(actor.agent_obs(obs, i), tau, hard=False)
        actions.append(action)
        leaves.append(tree_info["leaf"])
    nominal_actions = torch.cat(actions, -1)
    rows = safety.rows(state)
    shielded_actions, qp_info = safety.exact_projection(nominal_actions, rows)
    return shielded_actions, {"a_nom": nominal_actions, "leaves": torch.stack(leaves, -1), "rows": rows,
                    "violated": rows.violated(nominal_actions), **qp_info}


def _load_checkpoint(path: str, adapter, device):
    loaded_dict = torch.load(path, map_location=device, weights_only=False)
    for key, current in (("n_agents", adapter.num_agents), ("obs_dim_per", adapter.obs_dim_per),
                   ("act_dim_per", adapter.act_dim_per)):
        if int(loaded_dict[key]) != int(current):
            raise ValueError(f"checkpoint {key}={loaded_dict[key]} != env {key}={current}")
    cfg = loaded_dict["cfg"]
    actor = TeamActor(adapter.obs_dim_per, adapter.act_dim_per, adapter.num_agents,
                      int(cfg["tree_depth"]), float(cfg.get("log_std_init", -0.5))).to(device)
    actor.load_state_dict(loaded_dict["actor"])
    actor.eval()
    obs_norm = RunningMeanStd(adapter.obs_dim_per * adapter.num_agents).to(device)
    obs_norm.load_state_dict(loaded_dict["obs_norm"])
    known = {item.name for item in fields(SafetyConfig)}
    safety_cfg = SafetyConfig(**{key: v for key, v in dict(cfg.get("safety", {})).items() if key in known})
    return actor, obs_norm, FormationSafetyModel(safety_cfg), cfg


def _board_tilt_m(env) -> np.ndarray:
    corners = torch.tensor([[sx * BOARD_LENGTH / 2, sy * BOARD_WIDTH / 2, BOARD_THICKNESS / 2]
                            for sx in (1, -1) for sy in (1, -1)], device=env.device, dtype=torch.float32)
    z = _transform_points(corners, env._board.data.root_pos_w, env._board.data.root_quat_w)[:, :, 2]
    return (z - z.mean(dim=-1, keepdim=True)).abs().sum(dim=-1).cpu().numpy()


def _board_roll_pitch_deg(env) -> tuple[np.ndarray, np.ndarray]:
    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    gravity_body = quat_rotate_inverse(env._board.data.root_quat_w, gravity_world)
    pitch = torch.asin(gravity_body[:, 0].clamp(-1.0, 1.0)) * (180.0 / math.pi)
    roll = torch.asin(gravity_body[:, 1].clamp(-1.0, 1.0)) * (180.0 / math.pi)
    return roll.cpu().numpy(), pitch.cpu().numpy()


def _any_fallen(env) -> np.ndarray:
    out = np.zeros(env.num_envs, dtype=bool)
    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    for agent_id in ALL_ROBOT_IDS:
        robot = env.robots[agent_id]
        low = robot.data.root_pos_w[:, 2] < env.cfg.termination_height
        gravity_body = quat_rotate_inverse(robot.data.root_quat_w, gravity_world)
        out |= (low | (gravity_body[:, 0].abs() > 0.7) | (gravity_body[:, 1].abs() > 0.7)).cpu().numpy()
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
    PHYS = ("col_steps", "tilt_sum", "tilt_max", "roll_max", "pitch_max", "n_hl")
    SAFE = ("n_safe", "pviol", "interv", "corr", "nomcorr", "infeas", "active") + tuple(
        f"pviol_{f}" for f in FAMILY_NAMES)

    def __init__(self, num_envs: int):
        self.n = num_envs
        for k in self.PHYS + self.SAFE:
            setattr(self, k, np.zeros(num_envs, dtype=np.float64))
        self.z_min = np.full(num_envs, np.inf)
        self.min_h = np.full(num_envs, H_CAP)
        self.dropped = np.zeros(num_envs, dtype=bool)
        self.fallen = np.zeros(num_envs, dtype=bool)

    def clear(self, i: int) -> None:
        for k in self.PHYS + self.SAFE:
            getattr(self, k)[i] = 0.0
        self.z_min[i] = np.inf
        self.min_h[i] = H_CAP
        self.dropped[i] = self.fallen[i] = False


def main():
    toggles = _resolve_env_toggles()
    scene = json.loads(Path(args_cli.scene_json).read_text()) if args_cli.scene_json else None
    num_episodes = args_cli.num_episodes
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_episodes,
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
        partner_kwargs = {
            k: task_opts[f"partner_{suffix}"]
            for k, suffix in (
                ("lin", "noise_lin"),
                ("yaw_deg", "noise_yaw_deg"),
                ("gain_lin", "gain_lin"),
                ("gain_yaw", "gain_yaw"),
            )
            if task_opts.get(f"partner_{suffix}") is not None
        }
        adapter.set_partner_perturbation(**partner_kwargs)
        print(f"[eval] partner perturbation on {adapter.agent_ids[1]}: {adapter._partner}")
    device = adapter.device
    actor, obs_norm, safety, ckpt_cfg = _load_checkpoint(args_cli.checkpoint, adapter, device)
    variant = str(ckpt_cfg.get("variant", "?"))
    shield = (variant != "tree") if args_cli.shield == "auto" else (args_cli.shield == "on")
    scene_name = scene.get("name", Path(args_cli.scene_json).stem) if scene else "training_maze"
    num_waypoints = max(int(env._path.shape[0]) - 1, 1)
    max_len = int(env.max_episode_length)
    drop_height = float(env.cfg.board_drop_height)
    if args_cli.soft_tau is not None:
        print(f"[eval] soft routing at tau={args_cli.soft_tau} (not the deployed hard tree)")
    print(f"[eval] variant={variant} shield={'on' if shield else 'off'} scene={scene_name} "
          f"episodes={num_episodes} waypoints={num_waypoints} budget={env_cfg.episode_length_s}s "
          f"spawn={'randomised' if spawn_pool else 'fixed'}")

    stats = EpisodeStats(env.num_envs)
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
        stats.tilt_sum[sel] += tilt[sel]
        stats.tilt_max[sel] = np.maximum(stats.tilt_max[sel], tilt[sel])
        stats.roll_max[sel] = np.maximum(stats.roll_max[sel], np.abs(roll[sel]))
        stats.pitch_max[sel] = np.maximum(stats.pitch_max[sel], np.abs(pitch[sel]))
        stats.z_min[sel] = np.minimum(stats.z_min[sel], z[sel])
        stats.n_hl[sel] += 1
        stats.dropped[sel] |= z[sel] < drop_height
        stats.fallen[sel] |= fallen[sel]
        stats.col_steps[sel] += hit[sel].astype(np.float64)

    def patched_reset_idx(env_ids):
        if recording:
            ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
            acc_phys(ids)
            success_flag, loose_flag = episode_flags(env)
            for i in ids:
                steps = int(env.episode_length_buf[i].item())
                num_hl, num_safe = max(stats.n_hl[i], 1.0), max(stats.n_safe[i], 1.0)
                record = {
                    "env": int(i),
                    "wp": int(env._wp_reached_count[i].item()),
                    "success": bool(success_flag[i].item()),
                    "success_loose": bool(loose_flag[i].item()),
                    "steps": steps,
                    "timeout": steps >= max_len,
                    "collision_hl_steps": int(stats.col_steps[i]),
                    "drop": bool(stats.dropped[i]),
                    "fallen": bool(stats.fallen[i]),
                    "tilt_mean_m": float(stats.tilt_sum[i] / num_hl),
                    "tilt_max_m": float(stats.tilt_max[i]),
                    "roll_abs_max_deg": float(stats.roll_max[i]),
                    "pitch_abs_max_deg": float(stats.pitch_max[i]),
                    "board_z_min": float(stats.z_min[i]),
                    "pviol_nominal": float(stats.pviol[i] / num_safe),
                    "intervention_rate": float(stats.interv[i] / num_safe),
                    "correction": float(stats.corr[i] / num_safe),
                    "nominal_correction": float(stats.nomcorr[i] / num_safe),
                    "qp_infeasible": float(stats.infeas[i] / num_safe),
                    "rows_active": float(stats.active[i] / num_safe),
                    "min_h": float(stats.min_h[i]),
                }
                for family in FAMILY_NAMES:
                    record[f"pviol_{family}"] = float(getattr(stats, f"pviol_{family}")[i] / num_safe)
                record["completion"] = 1.0 if record["success"] else min(record["wp"] / num_waypoints, 1.0)
                record["reason"] = _fail_reason(record)
                records.append(record)
                stats.clear(i)
        return orig_reset_idx(env_ids)

    env._reset_idx = patched_reset_idx

    raw_obs = adapter.reset()
    recording = True
    first: dict[int, dict] = {}
    done_first = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    leaf_counts = torch.zeros(adapter.num_agents, actor.actors[0].tree.num_leaves, device=device)
    tolerance = args_cli.violation_tol

    for step in range(max_len + 5):
        state = adapter.safety_state()
        if args_cli.soft_tau is None:
            shielded_actions, info = deploy_action(actor, safety, obs_norm, raw_obs, state)
        else:
            shielded_actions, info = soft_action(actor, safety, obs_norm, raw_obs, state, args_cli.soft_tau)
        nominal_actions = info["a_nom"]
        actions = shielded_actions if shield else nominal_actions
        with torch.no_grad():
            rows = info["rows"]
            violated_rows = rows.margins(nominal_actions) < -tolerance
            nominal_correction = (shielded_actions - nominal_actions).norm(dim=-1)
            correction = (actions - nominal_actions).norm(dim=-1)
            min_barrier = rows.barrier.masked_fill(rows.mask <= 0, H_CAP).min(dim=1).values.clamp(max=H_CAP)
            live = ~done_first
            for agent in range(adapter.num_agents):
                leaf_counts[agent] += torch.bincount(info["leaves"][live, agent],
                                                 minlength=leaf_counts.shape[1]).float()
        stats.n_safe += 1.0
        stats.pviol += violated_rows.any(1).cpu().numpy()
        for family_id, family in enumerate(FAMILY_NAMES):
            getattr(stats, f"pviol_{family}")[:] += (violated_rows & (rows.family == family_id)).any(1).cpu().numpy()
        stats.interv += (correction > 1e-4).cpu().numpy()
        stats.corr += correction.cpu().numpy()
        stats.nomcorr += nominal_correction.cpu().numpy()
        stats.infeas += (~info["feasible"]).cpu().numpy()
        stats.active += info["active"].float().sum(1).cpu().numpy()
        stats.min_h = np.minimum(stats.min_h, min_barrier.cpu().numpy())

        raw_obs, _r, _done, _log = adapter.step(actions)
        acc_phys()
        for row in records:
            if row["env"] not in first:
                first[row["env"]] = row
                done_first[row["env"]] = True
        records.clear()
        if len(first) >= num_episodes:
            break
        if (step + 1) % 50 == 0:
            print(f"[eval] hl-step {step + 1}: {len(first)}/{num_episodes} episodes finished", flush=True)

    episode_rows = [first[k] for k in sorted(first)]
    if not episode_rows:
        raise RuntimeError("no episode finished within the step budget")

    def arr(key):
        return np.array([float(row[key]) for row in episode_rows], dtype=np.float64)

    def ms(key):
        values = arr(key)
        return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}

    summary = {
        "checkpoint": str(args_cli.checkpoint), "variant": variant, "shield": shield,
        "routing": "hard" if args_cli.soft_tau is None else f"soft(tau={args_cli.soft_tau})",
        "scene": scene_name, "episodes": len(episode_rows), "n_path_wp": num_waypoints,
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
        "fail_split": {k: sum(1 for row in episode_rows if row["reason"] == k)
                       for k in ("success", "drop", "fallen", "timeout", "other")},
        "tilt_mean_m": ms("tilt_mean_m"), "tilt_max_m": ms("tilt_max_m"),
        "steps": ms("steps"),
    }
    for k in ("pviol_nominal", "intervention_rate", "correction", "nominal_correction",
              "qp_infeasible", "rows_active", "min_h") + tuple(f"pviol_{family}" for family in FAMILY_NAMES):
        summary[k] = ms(k)
    occupancy = leaf_counts / leaf_counts.sum(dim=1, keepdim=True).clamp(min=1.0)
    summary["leaf_occupancy"] = {f"carrier_{agent}": [round(float(v), 4) for v in occupancy[agent]]
                                 for agent in range(adapter.num_agents)}
    summary["leaves_used"] = {
        f"carrier_{agent}": int((occupancy[agent] > 0).sum()) for agent in range(adapter.num_agents)
    }

    print("\n===== SAGE deployment evaluation =====")
    print(f"checkpoint: {args_cli.checkpoint}")
    print(f"variant: {variant} | shield: {'on' if shield else 'off'} | scene: {scene_name} | "
          f"episodes: {len(episode_rows)} | seed: {args_cli.seed}")
    print(f"success rate ({summary['success_mode']}): {summary['success_rate']:.3f}   "
          f"loose: {summary['success_rate_loose']:.3f}   completion: {summary['completion']['mean']:.3f}"
          f" ± {summary['completion']['std']:.3f}")
    print(f"collision ep rate: {summary['collision_episode_rate']:.3f}   drop: {summary['drop_episode_rate']:.3f}"
          f"   fallen: {summary['fallen_episode_rate']:.3f}   timeout: {summary['timeout_episode_rate']:.3f}")
    print(f"fail split: {summary['fail_split']}")
    print(f"PViol(a_nom): {summary['pviol_nominal']['mean']:.3f}   "
          + "  ".join(f"{family}={summary[f'pviol_{family}']['mean']:.3f}" for family in FAMILY_NAMES))
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
        out.write_text(json.dumps({"summary": summary, "episodes": episode_rows}, indent=2))
        print(f"[eval] wrote {out}")
    gym_env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    simulation_app.close()
