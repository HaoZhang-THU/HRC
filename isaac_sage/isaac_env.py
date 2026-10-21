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
    sys.meta_path[:] = [f for f in sys.meta_path
                        if not getattr(type(f), "__module__", "").startswith("__editable___isaaclab")]
    for pkg in reversed(ISAACLAB_CORE_PACKAGES):
        p = core_root / "source" / pkg
        if p.is_dir():
            sp = str(p)
            if sp in sys.path:
                sys.path.remove(sp)
            sys.path.insert(0, sp)


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
        cur = torch.norm(env._waypoints[:, 0] - board_xy, dim=-1)
        progress = torch.clamp((env._prev_dist_to_wp - cur) / v_ref, -1.0, 1.0)
        rew = orig()
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
        return {a: r + refund for a, r in rew.items()}

    env._get_rewards = _get_rewards


def _env_paths(env) -> Tensor:
    p = getattr(env, "_sage_env_path", None)
    return p if p is not None else env._path.unsqueeze(0).expand(env.num_envs, -1, -1)


def _project(paths: Tensor, pts: Tensor) -> tuple[Tensor, Tensor]:
    a, ab = paths[:, :-1], paths[:, 1:] - paths[:, :-1]
    seg2 = (ab * ab).sum(-1).clamp_min(1e-12)
    t = (((pts.unsqueeze(1) - a) * ab).sum(-1) / seg2).clamp(0.0, 1.0)
    d2 = ((a + t.unsqueeze(-1) * ab - pts.unsqueeze(1)) ** 2).sum(-1)
    k = d2.argmin(1, keepdim=True)
    seg = seg2.sqrt()
    s_cum = torch.cat((torch.zeros_like(seg[:, :1]), seg.cumsum(1)), 1)
    s = s_cum.gather(1, k).squeeze(1) + t.gather(1, k).squeeze(1) * seg.gather(1, k).squeeze(1)
    return s, s_cum


def episode_flags(env) -> tuple[Tensor, Tensor]:
    loose = getattr(env, "_sage_loose", None)
    loose = env._goal_reached if loose is None else loose
    succ = getattr(env, "_sage_success", None)
    return (loose if succ is None else succ), loose


LOOSE_GOAL_RADIUS = 0.15


def projected_progress(env, loose_goal: bool = True) -> None:
    from .task import cooperative_transport_env as task_mod

    if not env.cfg.use_normalized_reward or not env.cfg.use_fixed_path:
        raise ValueError("projected_progress needs the normalised reward and a fixed path")
    task_mod.WP_REACH_THRESHOLD = -1.0
    n, dev = env.num_envs, env.device
    n_pts = int(env._path.shape[0])
    hl_dt = env.cfg.sim.dt * env.cfg.decimation
    v_ref = max(env.cfg.nrew_progress_speed * hl_dt, 1e-6)
    env._sage_s = torch.zeros(n, device=dev)
    env._sage_loose = torch.zeros(n, dtype=torch.bool, device=dev)
    orig_rewards, orig_reset = env._get_rewards, env._reset_idx

    def _sync(ids, s, s_cum):
        k = torch.searchsorted(s_cum.contiguous(), (s + 1e-3).unsqueeze(1)).squeeze(1).clamp(1, n_pts - 1)
        env._path_idx[ids] = k
        env._wp_reached_count[ids] = k - 1
        env._refresh_path_window(ids)
        board_w = env._board.data.root_pos_w[ids, :2]
        env._prev_dist_to_wp[ids] = torch.norm(env._waypoints[ids, 0] - board_w, dim=-1)

    def _get_rewards():
        ids = torch.arange(n, device=dev)
        local = env._board.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        paths = _env_paths(env)
        s, s_cum = _project(paths, local)
        progress = ((s - env._sage_s) / v_ref).clamp(-1.0, 1.0)
        env._sage_s = s
        _sync(ids, s, s_cum)
        rew = orig_rewards()
        env._sage_stream = dict(rew)
        at_goal = torch.norm(paths[:, -1] - local, dim=-1) < LOOSE_GOAL_RADIUS
        env._sage_loose |= at_goal
        r = env.cfg.nrew_w_progress * progress
        log = env.extras.get("log")
        if loose_goal:
            newly = at_goal & ~env._goal_reached
            env._goal_reached |= at_goal
            bonus = newly.float() * env.cfg.nrew_w_goal
            r = r + bonus
            if log is not None:
                log["reward_goal"] = bonus.mean()
        if log is not None:
            log["rew_progress"] = log["rew_progress"] + (env.cfg.nrew_w_progress * progress).mean()
            log["reward_total"] = log["reward_total"] + r.mean()
        return {a: v + r for a, v in rew.items()}

    def _reset_idx(env_ids):
        orig_reset(env_ids)
        ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(list(env_ids), dtype=torch.long, device=dev)
        local = env._board.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2]
        s, s_cum = _project(_env_paths(env)[ids], local)
        env._sage_s[ids] = s
        env._sage_loose[ids] = False
        _sync(ids, s, s_cum)

    env._get_rewards = _get_rewards
    env._reset_idx = _reset_idx


def tight_success(env, goal_yaw: float, yaw_symmetry: float, pos_tol: float = 0.20, yaw_tol: float = 0.25,
                  speed_tol: float = 0.15, yaw_rate_tol: float = 0.30, hold_steps: int = 2,
                  shaping_radius: float = 1.75, shaping_weight: float = 0.2) -> None:
    from .task.cooperative_transport_env import _get_heading_yaw

    if env.cfg.terminate_on_goal:
        raise ValueError("tight_success needs cfg.terminate_on_goal = False (set before gym.make)")
    n, dev = env.num_envs, env.device
    env._sage_success = torch.zeros(n, dtype=torch.bool, device=dev)
    streak = torch.zeros(n, dtype=torch.long, device=dev)
    now = torch.zeros(n, dtype=torch.bool, device=dev)
    orig_dones, orig_rewards, orig_reset = env._get_dones, env._get_rewards, env._reset_idx

    def _goal_error():
        local = env._board.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        d = torch.norm(local - _env_paths(env)[:, -1], dim=-1)
        e = torch.remainder(_get_heading_yaw(env._board.data.root_quat_w) - goal_yaw, yaw_symmetry)
        e = torch.where(e > yaw_symmetry / 2, e - yaw_symmetry, e)
        return d, e

    def _get_dones():
        term, trunc = orig_dones()
        d, e = _goal_error()
        v = torch.norm(env._board.data.root_lin_vel_w[:, :2], dim=-1)
        w = env._board.data.root_ang_vel_w[:, 2].abs()
        ok = (d < pos_tol) & (e.abs() < yaw_tol) & (v < speed_tol) & (w < yaw_rate_tol)
        streak.copy_(torch.where(ok, streak + 1, torch.zeros_like(streak)))
        failed = next(iter(term.values())).bool()
        now.copy_((streak >= hold_steps) & ~failed & ~env._sage_success)
        env._sage_success |= now
        env._sage_success_now = now
        return {a: t | now for a, t in term.items()}, trunc

    def _get_rewards():
        latched = env._goal_reached.clone()
        rew = orig_rewards()
        task_goal = (env._goal_reached & ~latched).float() * env.cfg.nrew_w_goal
        d, e = _goal_error()
        shaping = shaping_weight * (1.0 - e.abs() / (yaw_symmetry / 2)) * (d < shaping_radius).float()
        bonus = now.float() * env.cfg.nrew_w_goal
        delta = bonus - task_goal + shaping
        log = env.extras.get("log")
        if log is not None:
            log["reward_goal"] = bonus.mean()
            log["rew_goal_yaw"] = shaping.mean()
            log["reward_total"] = log["reward_total"] + delta.mean()
        return {a: r + delta for a, r in rew.items()}

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
        rew = orig()
        now = getattr(env, "_sage_success_now", None)
        event = (env._goal_reached & ~before) if now is None else now
        remaining = (env.max_episode_length - env.episode_length_buf).clamp(min=0).float()
        scale = (1.0 - gamma ** remaining) / (1.0 - gamma)
        stream = getattr(env, "_sage_stream", None)
        out, comp_mean = {}, None
        for a, r in rew.items():
            base = stream[a] if stream is not None else r - w_goal * event.float()
            comp = event.float() * base.clamp(min=0.0) * scale
            out[a] = r + comp
            comp_mean = comp.mean()
        log = env.extras.get("log")
        if log is not None and comp_mean is not None:
            log["rew_success_comp"] = comp_mean
            log["reward_total"] = log["reward_total"] + comp_mean
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
        done.append(f"tight success (goal yaw {scene['goal_yaw']}, symmetry {scene.get('yaw_symmetry', 2 * math.pi):.3f})")
    if opts.get("success_compensation"):
        success_compensation(env)
        done.append("success compensation (goal bonus + discounted reward for the remaining budget)")
    return done


def randomize_spawn(env, pool: list[dict], curriculum: dict | None = None) -> None:
    from isaaclab.utils.math import quat_mul
    from .task.cooperative_transport_env import (
        ALL_ROBOT_IDS, NUM_WAYPOINTS, _get_heading_yaw)

    dev = env.device
    poses = torch.tensor([e["pose"] for e in pool], dtype=torch.float32, device=dev)
    paths = torch.tensor([e["path"] for e in pool], dtype=torch.float32, device=dev)
    n_pts = int(env._path.shape[0])
    if paths.shape[1] != n_pts:
        raise ValueError(f"spawn_pool paths have {paths.shape[1]} points, the task path {n_pts}")
    env_path = paths[0].expand(env.num_envs, -1, -1).clone()
    env._sage_env_path = env_path
    centre = torch.tensor(env.cfg.start_xy, dtype=torch.float32, device=dev)
    orig_reset = env._reset_idx
    if curriculum is not None:
        if "curriculum_ok" not in pool[0]:
            raise ValueError("spawn_pool has no curriculum_yaw / curriculum_ok; regenerate the scene")
        cur_yaw = torch.tensor([e["curriculum_yaw"] for e in pool], dtype=torch.float32, device=dev)
        cur_ok = torch.tensor([e["curriculum_ok"] for e in pool], dtype=torch.bool, device=dev)
    env._sage_train_progress = 0.0
    from_curriculum = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)

    def _refresh_path_window(env_ids):
        origins = env.scene.env_origins[env_ids, :2]
        base = env._path_idx[env_ids]
        for k in range(NUM_WAYPOINTS):
            idx = torch.clamp(base + k, max=n_pts - 1)
            env._waypoints[env_ids, k] = env_path[env_ids, idx] + origins

    def _place(obj, env_ids, xy, yaw):
        st = obj.data.default_root_state[env_ids].clone()
        c, s = torch.cos(yaw), torch.sin(yaw)
        rel = st[:, :2] - centre
        st[:, 0] = xy[:, 0] + c * rel[:, 0] - s * rel[:, 1]
        st[:, 1] = xy[:, 1] + s * rel[:, 0] + c * rel[:, 1]
        st[:, :3] += env.scene.env_origins[env_ids]
        q_yaw = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw),
                             torch.sin(yaw / 2)], dim=-1)
        st[:, 3:7] = quat_mul(q_yaw, st[:, 3:7])
        st[:, 7:] = 0.0
        obj.write_root_pose_to_sim(st[:, :7], env_ids)
        obj.write_root_velocity_to_sim(st[:, 7:], env_ids)
        return st[:, :2]

    def _reset_idx(env_ids):
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=dev)
        ended = env.episode_length_buf[env_ids] > 0
        succ, loose = episode_flags(env)
        succ, loose = succ[env_ids] & ended, loose[env_ids] & ended
        cur = from_curriculum[env_ids]
        log = env.extras.setdefault("log", {})
        for tag, sel in (("true_start", ~cur), ("curriculum", cur)):
            for key, flag in (("episodes", ended), ("success", succ), ("loose_success", loose)):
                log[f"{key}_{tag}"] = log.get(f"{key}_{tag}", 0.0) + float((flag & sel).sum())
        draw = torch.randint(len(pool), (len(env_ids),), device=dev)
        env_path[env_ids] = paths[draw]
        orig_reset(env_ids)
        xy, yaw = poses[draw, :2], poses[draw, 2]
        if curriculum is not None:
            frac = min(1.0, env._sage_train_progress / max(curriculum["anneal_fraction"], 1e-6))
            low = curriculum["initial_low"] + frac * (curriculum["final_low"] - curriculum["initial_low"])
            s = low + torch.rand(len(env_ids), device=dev) * (0.95 - low)
            idx = (s * (n_pts - 1)).long().clamp(0, n_pts - 2)
            use = (torch.rand(len(env_ids), device=dev) < curriculum["prob"]) & cur_ok[draw, idx]
            from_curriculum[env_ids] = use
            if use.any():
                xy = torch.where(use.unsqueeze(-1), paths[draw, idx], xy)
                yaw = torch.where(use, cur_yaw[draw, idx], yaw)
                env._path_idx[env_ids] = torch.where(use, idx + 1, env._path_idx[env_ids])
                env._wp_reached_count[env_ids] = torch.where(use, idx, env._wp_reached_count[env_ids])
                env._refresh_path_window(env_ids)
        for aid in ALL_ROBOT_IDS:
            w = _place(env.robots[aid], env_ids, xy, yaw)
            ryaw = _get_heading_yaw(env.robots[aid].data.default_root_state[env_ids, 3:7]) + yaw
            for buf in (env._target_pos_world, env._target_start_pos, env._target_final_pos):
                buf[aid][env_ids] = w
            for buf in (env._target_yaw_world, env._target_start_yaw, env._target_final_yaw):
                buf[aid][env_ids] = ryaw
        board_w = _place(env._board, env_ids, xy, yaw)
        env._prev_dist_to_wp[env_ids] = torch.norm(env._waypoints[env_ids, 0] - board_w, dim=-1)

    env._refresh_path_window = _refresh_path_window
    env._reset_idx = _reset_idx


class IsaacCoopTransportAdapter:
    def __init__(self, env_unwrapped):
        register_task()
        from .task.cooperative_transport_env import (
            ALL_ROBOT_IDS, BOARD_LENGTH, BOARD_WIDTH, _get_heading_yaw)

        self.env = env_unwrapped
        self._yaw = _get_heading_yaw
        cfg = env_unwrapped.cfg
        self.agent_ids = list(cfg.possible_agents)
        self.all_ids = list(ALL_ROBOT_IDS)
        self.n_agents = len(self.agent_ids)
        self.act_dim_per = 3
        self.obs_dim_per = int(cfg.observation_spaces[self.agent_ids[0]])
        self.num_envs = int(env_unwrapped.num_envs)
        self.device = torch.device(env_unwrapped.device)
        self.action_scale = torch.tensor(
            [cfg.max_target_dist, cfg.max_target_dist, cfg.max_target_yaw], device=self.device)
        off = cfg.robot_offset
        anchors = {"robot_0": (BOARD_LENGTH / 2 + off, 0.0),
                   "robot_1": (-(BOARD_LENGTH / 2 + off), 0.0),
                   "robot_2": (0.0, BOARD_WIDTH / 2 + off)}
        self.anchors_local = torch.tensor([anchors[a] for a in self.all_ids], device=self.device)
        self.controlled = torch.tensor([self.all_ids.index(a) for a in self.agent_ids],
                                       dtype=torch.long, device=self.device)
        self.aabbs = env_unwrapped._obs_aabb.to(self.device)
        self.board_dims = (BOARD_LENGTH, BOARD_WIDTH)
        self._partner = None

    def _stack(self, obs_dict: dict) -> Tensor:
        return torch.cat([obs_dict[a].to(self.device) for a in self.agent_ids], dim=-1)

    def set_partner_perturbation(self, slot: int = 1, lin: float = 0.01, yaw_deg: float = 1.5,
                                 gain_lin: float = 1.0, gain_yaw: float = 1.0) -> None:
        self._partner = {"slot": int(slot), "lin": float(lin), "yaw": math.radians(yaw_deg),
                         "gain_lin": float(gain_lin), "gain_yaw": float(gain_yaw)}

    def _perturb(self, cmd: Tensor) -> Tensor:
        p = self._partner
        xy = cmd[:, :2] * p["gain_lin"] + (torch.rand_like(cmd[:, :2]) * 2.0 - 1.0) * p["lin"]
        th = torch.randn(cmd.shape[0], device=cmd.device) * p["yaw"]
        c, s = torch.cos(th), torch.sin(th)
        xy = torch.stack((c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]), -1)
        return torch.cat((xy, cmd[:, 2:] * p["gain_yaw"]), -1)

    def set_train_progress(self, frac: float) -> None:
        self.env._sage_train_progress = float(frac)

    def reset(self) -> Tensor:
        out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        return self._stack(obs)

    def step(self, a_norm: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
        act = {}
        for i, aid in enumerate(self.agent_ids):
            act[aid] = a_norm[:, 3 * i:3 * i + 3].clamp(-1.0, 1.0) * self.action_scale
            if self._partner is not None and i == self._partner["slot"]:
                act[aid] = self._perturb(act[aid])
        obs, rew, term, trunc, extras = self.env.step(act)
        r = sum(rew[a].to(self.device) for a in self.agent_ids)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for a in self.agent_ids:
            done |= term[a].bool() | trunc[a].bool()
        log = {}
        for k, v in (extras.get("log", {}) or {}).items():
            try:
                log[k] = float(v)
            except (TypeError, ValueError):
                pass
        return self._stack(obs), r, done, log

    @torch.no_grad()
    def safety_state(self) -> SafetyState:
        env = self.env
        origins = env.scene.env_origins[:, :2]
        board_xy = env._board.data.root_pos_w[:, :2] - origins
        board_yaw = self._yaw(env._board.data.root_quat_w)
        rx, ry = [], []
        for aid in self.all_ids:
            rob = env.robots[aid]
            rx.append(rob.data.root_pos_w[:, :2] - origins)
            ry.append(self._yaw(rob.data.root_quat_w))
        return SafetyState(board_xy=board_xy, board_yaw=board_yaw,
                           robot_xy=torch.stack(rx, 1), robot_yaw=torch.stack(ry, 1),
                           controlled=self.controlled, anchors_local=self.anchors_local,
                           aabbs=self.aabbs, action_scale=self.action_scale)
