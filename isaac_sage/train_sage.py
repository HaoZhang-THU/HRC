#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from isaac_sage.isaac_env import setup_isaaclab_path

setup_isaaclab_path()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SAGE on G1 CooperativeTransport (Isaac Lab).")
parser.add_argument("--task", type=str, default="Isaac-G1-CooperativeTransport-Direct-v0")
parser.add_argument("--variant", type=str, default="sage", choices=["tree", "tree_cbf", "sail", "talo", "sage"])
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_env_steps", type=int, default=20_000_000)
parser.add_argument("--num_steps", type=int, default=60, help="Rollout length per update (high-level steps).")
parser.add_argument("--exp_name", type=str, default="coop_sage")
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--results_root", type=str, default=str(_PKG_ROOT / "results_isaac_sage"))
parser.add_argument("--checkpoint_interval", type=int, default=25)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--resume_fresh_critic", action="store_true")
parser.add_argument("--resume_fresh_obsnorm", action="store_true")
parser.add_argument("--resume_continue", action="store_true",
                    help="Continue the same run from --resume (restores env-step counter, optimisers, tau/kappa); "
                         "pass the original --log_dir so curves continue in place.")
parser.add_argument("--stall_timeout_s", type=float, default=900.0,
                    help="Dump all Python thread stacks to the log if one iteration exceeds this (0 = off).")
parser.add_argument("--tree_depth", type=int, default=4)
parser.add_argument("--actor_lr", type=float, default=1e-4)
parser.add_argument("--critic_lr", type=float, default=5e-5)
parser.add_argument("--tau_init", type=float, default=2.0)
parser.add_argument("--tau_final", type=float, default=0.12)
parser.add_argument("--tau_anneal_start", type=float, default=0.1)
parser.add_argument("--tau_anneal_end", type=float, default=0.8)
parser.add_argument("--hard_finetune_fraction", type=float, default=0.15,
                    help="Final fraction of training with straight-through hard routing (1.0 = hard from the start).")
parser.add_argument("--lambda_tree", type=float, default=0.01)
parser.add_argument("--no_freeze_routing_on_hard", action="store_true",
                    help="Keep training split hyperplanes/thresholds after the soft -> hard switch.")
parser.add_argument("--hard_lr_scale", type=float, default=0.5,
                    help="Multiply the actor lr by this once at the soft -> hard switch.")
parser.add_argument("--kappa_init", type=float, default=8.0)
parser.add_argument("--kappa_final", type=float, default=32.0)
parser.add_argument("--kappa_anneal_start", type=float, default=0.0)
parser.add_argument("--kappa_anneal_end", type=float, default=1.0)
parser.add_argument("--sail_newton_steps", type=int, default=8)
parser.add_argument("--sail_damping", type=float, default=1.0)
parser.add_argument("--talo_sigma", type=float, default=0.1)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--critic", type=str, default="auto", choices=["auto", "team", "joint"],
                    help="auto: joint-observation critic for tree / tree_cbf, team-context critic otherwise.")
parser.add_argument("--actor_update", type=str, default="auto", choices=["auto", "happo", "joint"],
                    help="Actor update of the arms without TALO.  auto/happo: sequential HAPPO updates; "
                         "joint: one clipped joint-ratio step (g_team).  TALO arms always use g_TALO.")
parser.add_argument("--critic_warmup_iters", type=int, default=5,
                    help="Critic-only iterations when a warm-started actor gets a fresh critic.")
parser.add_argument("--obstacle_margin", type=float, default=0.10)
parser.add_argument("--activation_distance", type=float, default=1.0)
parser.add_argument("--eta", type=float, default=1.0)
parser.add_argument("--no_curvature_margin", action="store_true")
parser.add_argument("--no_grasp_rows", action="store_true")
parser.add_argument("--no_rotation_rows", action="store_true")
parser.add_argument("--max_rows", type=int, default=32)
parser.add_argument("--mask_lidar", action="store_true", default=False)
parser.add_argument("--activate_lidar", action="store_true", default=False,
                    help="Warm start from a policy trained with --mask_lidar: zero its never-trained LiDAR "
                         "weights and re-estimate the LiDAR normalisation (fresh warm starts only).")
parser.add_argument("--scene_json", type=str, default=None,
                    help="Obstacle course (format of isaac_sage/scenes.py / eval_sage.py).")
parser.add_argument("--no_obstacles", action="store_true", default=False)
parser.add_argument("--no_fixed_path", action="store_true", default=False)
parser.add_argument("--target_radius", type=float, default=None)
parser.add_argument("--curriculum", action="store_true", default=False)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--w_heading", type=float, default=None)
parser.add_argument("--ungate_progress", action="store_true", default=False,
                    help="Pay path progress whatever the board heading (the task pays none for a broadside carry).")
parser.add_argument("--projection_progress", action="store_true", default=False,
                    help="Progress = advance of the board's projection onto the path; no waypoint has to be "
                         "visited and there is no per-waypoint bonus (excludes --ungate_progress).  With "
                         "--success_mode loose the goal is reached within 0.15 m of the final path point.")
parser.add_argument("--success_compensation", action="store_true", default=False,
                    help="With the goal bonus also pay the discounted per-step reward for the remaining time "
                         "budget, so finishing early never pays less than lingering near the goal.")
parser.add_argument("--success_mode", type=str, default="loose", choices=["loose", "tight"],
                    help="loose: the board reaches the final path point.  tight: within 0.20 m of it, yaw "
                         "within 0.25 rad of the scene's goal_yaw, speed < 0.15 m/s, yaw rate < 0.3 rad/s, "
                         "held for 2 high-level steps; the goal bonus moves there, plus near-goal yaw shaping.")
parser.add_argument("--partner_noise", action="store_true", default=False,
                    help="Perturb the second carrier's executed command (see the partner_* options).")
parser.add_argument("--partner_noise_lin", type=float, default=0.01, help="m per axis, uniform")
parser.add_argument("--partner_noise_yaw_deg", type=float, default=1.5, help="std of the command rotation")
parser.add_argument("--partner_gain_lin", type=float, default=1.0)
parser.add_argument("--partner_gain_yaw", type=float, default=1.0)
parser.add_argument("--start_curriculum_prob", type=float, default=0.0,
                    help="Fraction of training episodes started mid-path (scene spawn_pool with "
                         "curriculum_yaw / curriculum_ok required); evaluation always uses true starts.")
parser.add_argument("--start_curriculum_initial_low", type=float, default=0.8)
parser.add_argument("--start_curriculum_final_low", type=float, default=0.0)
parser.add_argument("--start_curriculum_anneal_fraction", type=float, default=0.6)
parser.add_argument("--w_obstacle", type=float, default=None)
parser.add_argument("--w_collision_contact", type=float, default=None,
                    help="Per-step penalty while the footprint overlaps a wall (non-terminating collisions).")
parser.add_argument("--gait_frequency", type=float, default=None)
parser.add_argument("--g1_usd_path", type=str, default=None,
                    help="Override the G1 robot USD used by the task (default: isaac_sage/assets/g1).")
parser.add_argument("--low_level_policy_path", type=str, default=None,
                    help="Override the frozen whole-body-controller TorchScript file used by the task.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab

import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaac_sage.isaac_env import (
    TASK_OPTIONS, IsaacCoopTransportAdapter, apply_task_options, configure_task_cfg, randomize_spawn,
    register_task, ungate_progress)
from isaac_sage.safety import SafetyConfig
from isaac_sage.trainer import SAGEConfig, train

register_task()


def _configure_env_cfg(cfg):
    if args_cli.mask_lidar:
        cfg.mask_lidar_obs = True
    if args_cli.no_obstacles:
        cfg.obstacle_specs = []
    if args_cli.no_fixed_path:
        cfg.use_fixed_path = False
    if args_cli.target_radius is not None:
        cfg.target_radius = args_cli.target_radius
        if not args_cli.curriculum:
            cfg.use_target_radius_curriculum = False
    elif not args_cli.curriculum and args_cli.no_fixed_path:
        cfg.use_target_radius_curriculum = False
    if args_cli.episode_length_s is not None:
        cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.w_heading is not None:
        cfg.nrew_w_heading = args_cli.w_heading
    if args_cli.w_obstacle is not None:
        cfg.nrew_w_obstacle = args_cli.w_obstacle
    if args_cli.w_collision_contact is not None:
        cfg.nrew_w_collision_contact = args_cli.w_collision_contact
    if args_cli.gait_frequency is not None:
        cfg.gait_frequency = args_cli.gait_frequency
    if args_cli.g1_usd_path is not None:
        cfg.training_g1_usd_path = args_cli.g1_usd_path
    if args_cli.low_level_policy_path is not None:
        cfg.low_level_policy_path = args_cli.low_level_policy_path
    if args_cli.scene_json is not None:
        scene = json.loads(Path(args_cli.scene_json).read_text())
        cfg.obstacle_specs = [(tuple(size), tuple(pos)) for size, pos in scene["obstacle_specs"]]
        cfg.path_waypoints = [tuple(waypoint) for waypoint in scene["path_waypoints"]]
        cfg.start_xy = tuple(scene.get("start_xy", scene["path_waypoints"][0]))
        cfg.use_fixed_path = True
        if "env_spacing" in scene:
            cfg.scene.env_spacing = float(scene["env_spacing"])
        if "episode_length_s" in scene and args_cli.episode_length_s is None:
            cfg.episode_length_s = float(scene["episode_length_s"])
    return cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not getattr(args_cli, "disable_fabric", False))
    env_cfg.seed = args_cli.seed
    env_cfg = _configure_env_cfg(env_cfg)
    if args_cli.ungate_progress and args_cli.projection_progress:
        raise ValueError("--projection_progress already pays progress whatever the heading; drop --ungate_progress")
    task_opts = {key: getattr(args_cli, key) for key in TASK_OPTIONS}
    configure_task_cfg(env_cfg, task_opts)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if args_cli.ungate_progress:
        ungate_progress(env.unwrapped)
        print("[SAGE] progress reward ungated (paid whatever the board heading)")
    pool = json.loads(Path(args_cli.scene_json).read_text()).get("spawn_pool") if args_cli.scene_json else None
    curriculum = None
    if args_cli.start_curriculum_prob > 0.0:
        if not pool:
            raise ValueError("--start_curriculum_prob needs a --scene_json with a spawn_pool")
        curriculum = {"prob": args_cli.start_curriculum_prob,
                      "initial_low": args_cli.start_curriculum_initial_low,
                      "final_low": args_cli.start_curriculum_final_low,
                      "anneal_fraction": args_cli.start_curriculum_anneal_fraction}
    if pool:
        randomize_spawn(env.unwrapped, pool, curriculum)
        print(f"[SAGE] randomised spawn: {len(pool)} start poses / paths"
              + (f"; start curriculum {curriculum}" if curriculum else ""))
    scene = json.loads(Path(args_cli.scene_json).read_text()) if args_cli.scene_json else None
    for msg in apply_task_options(env.unwrapped, task_opts, scene):
        print(f"[SAGE] {msg}")
    adapter = IsaacCoopTransportAdapter(env.unwrapped)
    if args_cli.partner_noise:
        adapter.set_partner_perturbation(lin=args_cli.partner_noise_lin, yaw_deg=args_cli.partner_noise_yaw_deg,
                                         gain_lin=args_cli.partner_gain_lin, gain_yaw=args_cli.partner_gain_yaw)
        print(f"[SAGE] partner perturbation on {adapter.agent_ids[1]}: {adapter._partner}")
    device = adapter.device
    activate_dims = None
    if args_cli.activate_lidar:
        if args_cli.mask_lidar:
            raise ValueError("--activate_lidar and --mask_lidar are mutually exclusive")
        num_rays = int(env_cfg.lidar_num_rays)
        core_dim = (adapter.obs_dim_per - num_rays) // 3
        activate_dims = list(range(core_dim, core_dim + num_rays))

    safety = SafetyConfig(obstacle_margin=args_cli.obstacle_margin,
                          activation_distance=args_cli.activation_distance, eta=args_cli.eta,
                          curvature_margin=not args_cli.no_curvature_margin,
                          use_grasp_rows=not args_cli.no_grasp_rows,
                          use_rotation_rows=not args_cli.no_rotation_rows,
                          yaw_step_bound=float(env_cfg.max_target_yaw), max_rows=args_cli.max_rows)
    cfg = SAGEConfig(variant=args_cli.variant, tree_depth=args_cli.tree_depth,
                     tau_init=args_cli.tau_init, tau_final=args_cli.tau_final,
                     tau_anneal_start=args_cli.tau_anneal_start, tau_anneal_end=args_cli.tau_anneal_end,
                     hard_finetune_fraction=args_cli.hard_finetune_fraction, lambda_tree=args_cli.lambda_tree,
                     freeze_routing_on_hard=not args_cli.no_freeze_routing_on_hard,
                     hard_lr_scale=args_cli.hard_lr_scale,
                     kappa_init=args_cli.kappa_init, kappa_final=args_cli.kappa_final,
                     kappa_anneal_start=args_cli.kappa_anneal_start, kappa_anneal_end=args_cli.kappa_anneal_end,
                     sail_newton_steps=args_cli.sail_newton_steps, sail_damping=args_cli.sail_damping,
                     talo_sigma=args_cli.talo_sigma, max_grad_norm=args_cli.max_grad_norm,
                     critic=args_cli.critic, actor_update=args_cli.actor_update,
                     critic_warmup_iters=args_cli.critic_warmup_iters,
                     actor_lr=args_cli.actor_lr, critic_lr=args_cli.critic_lr,
                     num_steps=args_cli.num_steps, safety=safety)

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = Path(args_cli.log_dir or
                   Path(args_cli.results_root) / f"{args_cli.exp_name}_{args_cli.variant}_seed{args_cli.seed}_{stamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    args_name = "run_args_continue.json" if args_cli.resume_continue else "run_args.json"
    run_args = {**vars(args_cli), "isaaclab_module": isaaclab.__file__}
    print(f"[SAGE] isaaclab from {isaaclab.__file__}")
    (log_dir / args_name).write_text(json.dumps(run_args, indent=2, default=str))
    train(adapter, cfg, total_steps=args_cli.num_env_steps, log_dir=log_dir, device=device,
          seed=args_cli.seed, checkpoint_interval=args_cli.checkpoint_interval, resume=args_cli.resume,
          resume_load_critic=not args_cli.resume_fresh_critic,
          resume_load_obs_norm=not args_cli.resume_fresh_obsnorm,
          resume_continue=args_cli.resume_continue, stall_timeout_s=args_cli.stall_timeout_s,
          activate_dims=activate_dims)
    env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    simulation_app.close()
