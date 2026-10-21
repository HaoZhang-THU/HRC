from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
from torch import Tensor

from .safety import SafetyState

TASK_PKG = "isaac_sage.task"


ISAACLAB_CORE_PACKAGES = ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_mimic", "isaaclab_tasks")


def use_isaaclab_core(core_root: str | Path) -> None:
    core_root = Path(core_root)
    if not (core_root / "source" / "isaaclab" / "isaaclab").is_dir():
        raise FileNotFoundError(f"no Isaac Lab checkout at {core_root}")
    os.environ["ISAACLAB_PATH"] = str(core_root)
    sys.meta_path[:] = [finder for finder in sys.meta_path
                        if not getattr(type(finder), "__module__", "").startswith("__editable___isaaclab")]
    for pkg in reversed(ISAACLAB_CORE_PACKAGES):
        path = core_root / "source" / pkg
        if path.is_dir():
            path_str = str(path)
            if path_str in sys.path:
                sys.path.remove(path_str)
            sys.path.insert(0, path_str)


def setup_isaaclab_path() -> Path | None:
    core = os.environ.get("ISAACLAB_CORE_ROOT")
    if not core:
        return None
    use_isaaclab_core(core)
    return Path(core)


def register_task() -> None:
    import importlib

    importlib.import_module(TASK_PKG)


def ungate_progress(env) -> None:
    import math

    from .task.cooperative_transport_env import _get_heading_yaw

    if not env.cfg.use_normalized_reward:
        raise ValueError("ungate_progress needs the task's normalised reward")
    orig = env._get_rewards
    hl_dt = env.cfg.sim.dt * env.cfg.decimation
    v_ref = max(env.cfg.nrew_progress_speed * hl_dt, 1e-6)

    def _get_rewards():
        board_xy = env._board.data.root_pos_w[:, :2]
        dist = torch.norm(env._waypoints[:, 0] - board_xy, dim=-1)
        progress = torch.clamp((env._prev_dist_to_wp - dist) / v_ref, -1.0, 1.0)
        rewards = orig()
        seg = env._waypoints[:, 1] - env._waypoints[:, 0]
        seg_ok = torch.norm(seg, dim=-1, keepdim=True) > 1e-3
        dir_vec = torch.where(seg_ok, seg, env._waypoints[:, 0] - board_xy)
        board_yaw = _get_heading_yaw(env._board.data.root_quat_w)
        align = torch.clamp(torch.cos(board_yaw + math.pi - torch.atan2(dir_vec[:, 1], dir_vec[:, 0])), 0.0, 1.0)
        refund = env.cfg.nrew_w_progress * progress.clamp(min=0.0) * (1.0 - align)
        log = env.extras.get("log")
        if log is not None:
            log["rew_progress_refund"] = refund.mean()
            log["rew_progress"] = log["rew_progress"] + refund.mean()
            log["reward_total"] = log["reward_total"] + refund.mean()
        return {agent: reward + refund for agent, reward in rewards.items()}

    env._get_rewards = _get_rewards


def _env_paths(env) -> Tensor:
    path = getattr(env, "_sage_env_path", None)
    return path if path is not None else env._path.unsqueeze(0).expand(env.num_envs, -1, -1)


def _project(paths: Tensor, points: Tensor) -> tuple[Tensor, Tensor]:
    seg_start, seg = paths[:, :-1], paths[:, 1:] - paths[:, :-1]
    seg_len_sq = (seg * seg).sum(-1).clamp_min(1e-12)
    ratio = (((points.unsqueeze(1) - seg_start) * seg).sum(-1) / seg_len_sq).clamp(0.0, 1.0)
    dist_sq = ((seg_start + ratio.unsqueeze(-1) * seg - points.unsqueeze(1)) ** 2).sum(-1)
    nearest = dist_sq.argmin(1, keepdim=True)
    seg_len = seg_len_sq.sqrt()
    cum_length = torch.cat((torch.zeros_like(seg_len[:, :1]), seg_len.cumsum(1)), 1)
    arc = cum_length.gather(1, nearest).squeeze(1) + ratio.gather(1, nearest).squeeze(1) * seg_len.gather(
        1, nearest
    ).squeeze(1)
    return arc, cum_length


def episode_flags(env) -> tuple[Tensor, Tensor]:
    loose = getattr(env, "_sage_loose", None)
    loose = env._goal_reached if loose is None else loose
    success = getattr(env, "_sage_success", None)
    return (loose if success is None else success), loose


LOOSE_GOAL_RADIUS = 0.15


def projected_progress(env, loose_goal: bool = True) -> None:
    from .task import cooperative_transport_env as task_mod

    if not env.cfg.use_normalized_reward or not env.cfg.use_fixed_path:
        raise ValueError("projected_progress needs the normalised reward and a fixed path")
    task_mod.WP_REACH_THRESHOLD = -1.0
    num_envs, device = env.num_envs, env.device
    num_points = int(env._path.shape[0])
    hl_dt = env.cfg.sim.dt * env.cfg.decimation
    v_ref = max(env.cfg.nrew_progress_speed * hl_dt, 1e-6)
    env._sage_s = torch.zeros(num_envs, device=device)
    env._sage_loose = torch.zeros(num_envs, dtype=torch.bool, device=device)
    orig_rewards, orig_reset = env._get_rewards, env._reset_idx

    def _sync(ids, arc, cum_length):
        index = (
            torch.searchsorted(cum_length.contiguous(), (arc + 1e-3).unsqueeze(1))
            .squeeze(1)
            .clamp(1, num_points - 1)
        )
        env._path_idx[ids] = index
        env._wp_reached_count[ids] = index - 1
        env._refresh_path_window(ids)
        board_world = env._board.data.root_pos_w[ids, :2]
        env._prev_dist_to_wp[ids] = torch.norm(env._waypoints[ids, 0] - board_world, dim=-1)

    def _get_rewards():
        ids = torch.arange(num_envs, device=device)
        local = env._board.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        paths = _env_paths(env)
        arc, cum_length = _project(paths, local)
        progress = ((arc - env._sage_s) / v_ref).clamp(-1.0, 1.0)
        env._sage_s = arc
        _sync(ids, arc, cum_length)
        rewards = orig_rewards()
        env._sage_stream = dict(rewards)
        at_goal = torch.norm(paths[:, -1] - local, dim=-1) < LOOSE_GOAL_RADIUS
        env._sage_loose |= at_goal
        extra = env.cfg.nrew_w_progress * progress
        log = env.extras.get("log")
        if loose_goal:
            newly = at_goal & ~env._goal_reached
            env._goal_reached |= at_goal
            bonus = newly.float() * env.cfg.nrew_w_goal
            extra = extra + bonus
            if log is not None:
                log["reward_goal"] = bonus.mean()
        if log is not None:
            log["rew_progress"] = log["rew_progress"] + (env.cfg.nrew_w_progress * progress).mean()
            log["reward_total"] = log["reward_total"] + extra.mean()
        return {agent: value + extra for agent, value in rewards.items()}

    def _reset_idx(env_ids):
        orig_reset(env_ids)
        ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(list(env_ids), dtype=torch.long, device=device)
        local = env._board.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2]
        arc, cum_length = _project(_env_paths(env)[ids], local)
        env._sage_s[ids] = arc
        env._sage_loose[ids] = False
        _sync(ids, arc, cum_length)

    env._get_rewards = _get_rewards
    env._reset_idx = _reset_idx


def tight_success(env, goal_yaw: float, yaw_symmetry: float, pos_tol: float = 0.20, yaw_tol: float = 0.25,
                  speed_tol: float = 0.15, yaw_rate_tol: float = 0.30, hold_steps: int = 2,
                  shaping_radius: float = 1.75, shaping_weight: float = 0.2) -> None:
    from .task.cooperative_transport_env import _get_heading_yaw

    if env.cfg.terminate_on_goal:
        raise ValueError("tight_success needs cfg.terminate_on_goal = False (set before gym.make)")
    num_envs, device = env.num_envs, env.device
    env._sage_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    streak = torch.zeros(num_envs, dtype=torch.long, device=device)
    now = torch.zeros(num_envs, dtype=torch.bool, device=device)
    orig_dones, orig_rewards, orig_reset = env._get_dones, env._get_rewards, env._reset_idx

    def _goal_error():
        local = env._board.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        dist = torch.norm(local - _env_paths(env)[:, -1], dim=-1)
        yaw_error = torch.remainder(_get_heading_yaw(env._board.data.root_quat_w) - goal_yaw, yaw_symmetry)
        yaw_error = torch.where(yaw_error > yaw_symmetry / 2, yaw_error - yaw_symmetry, yaw_error)
        return dist, yaw_error

    def _get_dones():
        term, trunc = orig_dones()
        dist, yaw_error = _goal_error()
        speed = torch.norm(env._board.data.root_lin_vel_w[:, :2], dim=-1)
        yaw_rate = env._board.data.root_ang_vel_w[:, 2].abs()
        reached = (dist < pos_tol) & (yaw_error.abs() < yaw_tol) & (speed < speed_tol) & (yaw_rate < yaw_rate_tol)
        streak.copy_(torch.where(reached, streak + 1, torch.zeros_like(streak)))
        failed = next(iter(term.values())).bool()
        now.copy_((streak >= hold_steps) & ~failed & ~env._sage_success)
        env._sage_success |= now
        env._sage_success_now = now
        return {agent: flag | now for agent, flag in term.items()}, trunc

    def _get_rewards():
        latched = env._goal_reached.clone()
        rewards = orig_rewards()
        task_goal = (env._goal_reached & ~latched).float() * env.cfg.nrew_w_goal
        dist, yaw_error = _goal_error()
        shaping = shaping_weight * (1.0 - yaw_error.abs() / (yaw_symmetry / 2)) * (dist < shaping_radius).float()
        bonus = now.float() * env.cfg.nrew_w_goal
        delta = bonus - task_goal + shaping
        log = env.extras.get("log")
        if log is not None:
            log["reward_goal"] = bonus.mean()
            log["rew_goal_yaw"] = shaping.mean()
            log["reward_total"] = log["reward_total"] + delta.mean()
        return {agent: reward + delta for agent, reward in rewards.items()}

    def _reset_idx(env_ids):
        orig_reset(env_ids)
        env._sage_success[env_ids] = False
        streak[env_ids] = 0
        now[env_ids] = False

    env._get_dones = _get_dones
    env._get_rewards = _get_rewards
    env._reset_idx = _reset_idx


def success_compensation(env, gamma: float = 0.99) -> None:
    orig = env._get_rewards
    w_goal = float(env.cfg.nrew_w_goal)

    def _get_rewards():
        before = env._goal_reached.clone()
        rewards = orig()
        now = getattr(env, "_sage_success_now", None)
        event = (env._goal_reached & ~before) if now is None else now
        remaining = (env.max_episode_length - env.episode_length_buf).clamp(min=0).float()
        scale = (1.0 - gamma ** remaining) / (1.0 - gamma)
        stream = getattr(env, "_sage_stream", None)
        out, compensation_mean = {}, None
        for agent, reward in rewards.items():
            base = stream[agent] if stream is not None else reward - w_goal * event.float()
            compensation = event.float() * base.clamp(min=0.0) * scale
            out[agent] = reward + compensation
            compensation_mean = compensation.mean()
        log = env.extras.get("log")
        if log is not None and compensation_mean is not None:
            log["rew_success_comp"] = compensation_mean
            log["reward_total"] = log["reward_total"] + compensation_mean
        return out

    env._get_rewards = _get_rewards


TASK_OPTIONS = ("projection_progress", "success_mode", "partner_noise", "partner_noise_lin",
                "partner_noise_yaw_deg", "partner_gain_lin", "partner_gain_yaw", "success_compensation")


def configure_task_cfg(cfg, opts: dict) -> None:
    if opts.get("success_mode") == "tight":
        cfg.terminate_on_goal = False


def apply_task_options(env, opts: dict, scene: dict | None) -> list[str]:
    done = []
    if opts.get("projection_progress"):
        loose = opts.get("success_mode") != "tight"
        projected_progress(env, loose_goal=loose)
        done.append("projected path progress (no waypoint bonus)"
                    + (f"; goal = within {LOOSE_GOAL_RADIUS} m of the final path point" if loose else ""))
    if opts.get("success_mode") == "tight":
        if scene is None or "goal_yaw" not in scene:
            raise ValueError("success_mode tight needs a scene with goal_yaw")
        tight_success(env, float(scene["goal_yaw"]), float(scene.get("yaw_symmetry", 2 * math.pi)))
        done.append(
            f"tight success (goal yaw {scene['goal_yaw']}, symmetry {scene.get('yaw_symmetry', 2 * math.pi):.3f})"
        )
    if opts.get("success_compensation"):
        success_compensation(env)
        done.append("success compensation (goal bonus + discounted reward for the remaining budget)")
    return done


def randomize_spawn(env, pool: list[dict], curriculum: dict | None = None) -> None:
    from isaaclab.utils.math import quat_mul
    from .task.cooperative_transport_env import (
        ALL_ROBOT_IDS, NUM_WAYPOINTS, _get_heading_yaw)

    device = env.device
    poses = torch.tensor([entry["pose"] for entry in pool], dtype=torch.float32, device=device)
    paths = torch.tensor([entry["path"] for entry in pool], dtype=torch.float32, device=device)
    num_points = int(env._path.shape[0])
    if paths.shape[1] != num_points:
        raise ValueError(f"spawn_pool paths have {paths.shape[1]} points, the task path {num_points}")
    env_path = paths[0].expand(env.num_envs, -1, -1).clone()
    env._sage_env_path = env_path
    centre = torch.tensor(env.cfg.start_xy, dtype=torch.float32, device=device)
    orig_reset = env._reset_idx
    if curriculum is not None:
        if "curriculum_ok" not in pool[0]:
            raise ValueError("spawn_pool has no curriculum_yaw / curriculum_ok; regenerate the scene")
        curriculum_yaw = torch.tensor([entry["curriculum_yaw"] for entry in pool], dtype=torch.float32, device=device)
        curriculum_ok = torch.tensor([entry["curriculum_ok"] for entry in pool], dtype=torch.bool, device=device)
    env._sage_train_progress = 0.0
    from_curriculum = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    def _refresh_path_window(env_ids):
        origins = env.scene.env_origins[env_ids, :2]
        base = env._path_idx[env_ids]
        for k in range(NUM_WAYPOINTS):
            idx = torch.clamp(base + k, max=num_points - 1)
            env._waypoints[env_ids, k] = env_path[env_ids, idx] + origins

    def _place(obj, env_ids, xy, yaw):
        state = obj.data.default_root_state[env_ids].clone()
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        rel = state[:, :2] - centre
        state[:, 0] = xy[:, 0] + cos * rel[:, 0] - sin * rel[:, 1]
        state[:, 1] = xy[:, 1] + sin * rel[:, 0] + cos * rel[:, 1]
        state[:, :3] += env.scene.env_origins[env_ids]
        q_yaw = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw),
                             torch.sin(yaw / 2)], dim=-1)
        state[:, 3:7] = quat_mul(q_yaw, state[:, 3:7])
        state[:, 7:] = 0.0
        obj.write_root_pose_to_sim(state[:, :7], env_ids)
        obj.write_root_velocity_to_sim(state[:, 7:], env_ids)
        return state[:, :2]

    def _reset_idx(env_ids):
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=device)
        ended = env.episode_length_buf[env_ids] > 0
        success, loose = episode_flags(env)
        success, loose = success[env_ids] & ended, loose[env_ids] & ended
        is_curriculum = from_curriculum[env_ids]
        log = env.extras.setdefault("log", {})
        for tag, sel in (("true_start", ~is_curriculum), ("curriculum", is_curriculum)):
            for key, flag in (("episodes", ended), ("success", success), ("loose_success", loose)):
                log[f"{key}_{tag}"] = log.get(f"{key}_{tag}", 0.0) + float((flag & sel).sum())
        draw = torch.randint(len(pool), (len(env_ids),), device=device)
        env_path[env_ids] = paths[draw]
        orig_reset(env_ids)
        xy, yaw = poses[draw, :2], poses[draw, 2]
        if curriculum is not None:
            frac = min(1.0, env._sage_train_progress / max(curriculum["anneal_fraction"], 1e-6))
            low = curriculum["initial_low"] + frac * (curriculum["final_low"] - curriculum["initial_low"])
            start_frac = low + torch.rand(len(env_ids), device=device) * (0.95 - low)
            idx = (start_frac * (num_points - 1)).long().clamp(0, num_points - 2)
            use = (torch.rand(len(env_ids), device=device) < curriculum["prob"]) & curriculum_ok[draw, idx]
            from_curriculum[env_ids] = use
            if use.any():
                xy = torch.where(use.unsqueeze(-1), paths[draw, idx], xy)
                yaw = torch.where(use, curriculum_yaw[draw, idx], yaw)
                env._path_idx[env_ids] = torch.where(use, idx + 1, env._path_idx[env_ids])
                env._wp_reached_count[env_ids] = torch.where(use, idx, env._wp_reached_count[env_ids])
                env._refresh_path_window(env_ids)
        for agent_id in ALL_ROBOT_IDS:
            world_xy = _place(env.robots[agent_id], env_ids, xy, yaw)
            robot_yaw = _get_heading_yaw(env.robots[agent_id].data.default_root_state[env_ids, 3:7]) + yaw
            for target_buf in (env._target_pos_world, env._target_start_pos, env._target_final_pos):
                target_buf[agent_id][env_ids] = world_xy
            for target_buf in (env._target_yaw_world, env._target_start_yaw, env._target_final_yaw):
                target_buf[agent_id][env_ids] = robot_yaw
        board_world = _place(env._board, env_ids, xy, yaw)
        env._prev_dist_to_wp[env_ids] = torch.norm(env._waypoints[env_ids, 0] - board_world, dim=-1)

    env._refresh_path_window = _refresh_path_window
    env._reset_idx = _reset_idx


class IsaacCoopTransportAdapter:
    def __init__(self, env_unwrapped):
        register_task()
        from .task.cooperative_transport_env import (
            ALL_ROBOT_IDS, BOARD_LENGTH, BOARD_WIDTH, _get_heading_yaw)

        self.env = env_unwrapped
        self._heading_yaw = _get_heading_yaw
        cfg = env_unwrapped.cfg
        self.agent_ids = list(cfg.possible_agents)
        self.all_ids = list(ALL_ROBOT_IDS)
        self.num_agents = len(self.agent_ids)
        self.act_dim_per = 3
        self.obs_dim_per = int(cfg.observation_spaces[self.agent_ids[0]])
        self.num_envs = int(env_unwrapped.num_envs)
        self.device = torch.device(env_unwrapped.device)
        self.action_scale = torch.tensor(
            [cfg.max_target_dist, cfg.max_target_dist, cfg.max_target_yaw], device=self.device)
        offset = cfg.robot_offset
        anchors = {"robot_0": (BOARD_LENGTH / 2 + offset, 0.0),
                   "robot_1": (-(BOARD_LENGTH / 2 + offset), 0.0),
                   "robot_2": (0.0, BOARD_WIDTH / 2 + offset)}
        self.anchors_local = torch.tensor([anchors[agent] for agent in self.all_ids], device=self.device)
        self.controlled = torch.tensor([self.all_ids.index(agent) for agent in self.agent_ids],
                                       dtype=torch.long, device=self.device)
        self.aabbs = env_unwrapped._obs_aabb.to(self.device)
        self.board_dims = (BOARD_LENGTH, BOARD_WIDTH)
        self._partner = None

    def _stack(self, obs_dict: dict) -> Tensor:
        return torch.cat([obs_dict[agent].to(self.device) for agent in self.agent_ids], dim=-1)

    def set_partner_perturbation(self, slot: int = 1, lin: float = 0.01, yaw_deg: float = 1.5,
                                 gain_lin: float = 1.0, gain_yaw: float = 1.0) -> None:
        self._partner = {"slot": int(slot), "lin": float(lin), "yaw": math.radians(yaw_deg),
                         "gain_lin": float(gain_lin), "gain_yaw": float(gain_yaw)}

    def _perturb(self, command: Tensor) -> Tensor:
        params = self._partner
        xy = command[:, :2] * params["gain_lin"] + (torch.rand_like(command[:, :2]) * 2.0 - 1.0) * params["lin"]
        angle = torch.randn(command.shape[0], device=command.device) * params["yaw"]
        cos, sin = torch.cos(angle), torch.sin(angle)
        xy = torch.stack((cos * xy[:, 0] - sin * xy[:, 1], sin * xy[:, 0] + cos * xy[:, 1]), -1)
        return torch.cat((xy, command[:, 2:] * params["gain_yaw"]), -1)

    def set_train_progress(self, frac: float) -> None:
        self.env._sage_train_progress = float(frac)

    def reset(self) -> Tensor:
        out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        return self._stack(obs)

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
        commands = {}
        for i, agent_id in enumerate(self.agent_ids):
            commands[agent_id] = actions[:, 3 * i:3 * i + 3].clamp(-1.0, 1.0) * self.action_scale
            if self._partner is not None and i == self._partner["slot"]:
                commands[agent_id] = self._perturb(commands[agent_id])
        obs, rewards, term, trunc, extras = self.env.step(commands)
        reward = sum(rewards[agent].to(self.device) for agent in self.agent_ids)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.agent_ids:
            done |= term[agent].bool() | trunc[agent].bool()
        log = {}
        for key, value in (extras.get("log", {}) or {}).items():
            try:
                log[key] = float(value)
            except (TypeError, ValueError):
                pass
        return self._stack(obs), reward, done, log

    @torch.no_grad()
    def safety_state(self) -> SafetyState:
        env = self.env
        origins = env.scene.env_origins[:, :2]
        board_xy = env._board.data.root_pos_w[:, :2] - origins
        board_yaw = self._heading_yaw(env._board.data.root_quat_w)
        positions, yaws = [], []
        for agent_id in self.all_ids:
            robot = env.robots[agent_id]
            positions.append(robot.data.root_pos_w[:, :2] - origins)
            yaws.append(self._heading_yaw(robot.data.root_quat_w))
        return SafetyState(board_xy=board_xy, board_yaw=board_yaw,
                           robot_xy=torch.stack(positions, 1), robot_yaw=torch.stack(yaws, 1),
                           controlled=self.controlled, anchors_local=self.anchors_local,
                           aabbs=self.aabbs, action_scale=self.action_scale)
