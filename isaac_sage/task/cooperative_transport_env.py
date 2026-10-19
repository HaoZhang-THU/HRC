"""G1 cooperative-transport environment.

Three G1 robots (N = 3) carry a rectangular board along a path.  Each robot is
driven by a frozen waist-tracking whole-body controller.  The two learning
carriers (robot_0, robot_1) receive high-level [dx, dy, dyaw] commands at 1 Hz;
robot_2 is a follower rigidly tied to the payload.

Architecture:
  high-level policy (1 Hz) → [dx, dy, dyaw] per carrier → waist-tracking controller (50 Hz) → 23-DOF torques
"""

from __future__ import annotations

import math
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse

# ---------------------------------------------------------------------------
# Assets shipped with the package and low-level controller configuration
# ---------------------------------------------------------------------------

ASSETS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets"))
TRAINING_G1_USD_PATH = os.path.join(ASSETS_DIR, "g1", "g1_29dof_anneal_23dof_palmup.usd")
LOW_LEVEL_POLICY_PATH = os.path.join(ASSETS_DIR, "policy_waist_tracking_g1_23dof.pt")

G1_23DOF_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]

# Default joint angles: legs from training, arms at default (upper arm vertical, forearm horizontal)
G1_23DOF_DEFAULT_POS_CARRY = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right leg
    0.0, 0.0, 0.0,                      # waist (locked)
    0.0, 0.0, 0.0, 0.0,                # left arm: all zero = upper arm down, forearm horizontal
    0.0, 0.0, 0.0, 0.0,                # right arm: all zero = upper arm down, forearm horizontal
]

G1_23DOF_EFFORT = [
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 50.0, 50.0,
    25.0, 25.0, 25.0, 25.0,
    25.0, 25.0, 25.0, 25.0,
]
G1_23DOF_VEL_LIMIT = [
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
    32.0, 37.0, 37.0,
    37.0, 37.0, 37.0, 37.0,
    37.0, 37.0, 37.0, 37.0,
]
G1_23DOF_ARMATURE = [
    0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
    0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
    0.03, 0.03, 0.03,
    0.03, 0.03, 0.03, 0.03,
    0.04, 0.03, 0.03, 0.03,
]
G1_23DOF_FRICTION = [
    0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
    0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
    0.03, 0.03, 0.03,
    0.03, 0.03, 0.03, 0.03,
    0.03, 0.03, 0.03, 0.03,
]

LOCKED_DOF_INDICES = list(range(12, 23))

G1_STIFFNESS = {
    "hip_pitch": 100.0, "hip_roll": 100.0, "hip_yaw": 100.0, "knee": 200.0,
    "ankle_pitch": 20.0, "ankle_roll": 20.0,
    "waist_yaw": 400.0, "waist_roll": 400.0, "waist_pitch": 400.0,
    "shoulder_pitch": 90.0, "shoulder_roll": 60.0, "shoulder_yaw": 20.0, "elbow": 60.0,
}
G1_DAMPING = {
    "hip_pitch": 2.5, "hip_roll": 2.5, "hip_yaw": 2.5, "knee": 5.0,
    "ankle_pitch": 0.2, "ankle_roll": 0.1,
    "waist_yaw": 5.0, "waist_roll": 5.0, "waist_pitch": 5.0,
    "shoulder_pitch": 2.0, "shoulder_roll": 1.0, "shoulder_yaw": 0.4, "elbow": 1.0,
}

NUM_ROBOTS = 3
ALL_ROBOT_IDS = ["robot_0", "robot_1", "robot_2"]  # all robots in scene
FOLLOWER_ROBOT_IDS = ["robot_2"]  # passive followers (not learning carriers)
NUM_WAYPOINTS = 5  # sliding window of VLM-style path waypoints
OBS_CORE_PER_FRAME = 49  # psi_task(10) + psi_ego(7) + psi_ptn1(7) + psi_ptn2(7) + psi_obj(18)
OBS_LIDAR = 36
OBS_CURRENT_FRAME = OBS_CORE_PER_FRAME + OBS_LIDAR  # 85
OBS_TEMPORAL_FRAMES = 2  # previous 2 frames (core only, no LiDAR)
OBS_TOTAL = OBS_CURRENT_FRAME + OBS_CORE_PER_FRAME * OBS_TEMPORAL_FRAMES  # 85 + 49*2 = 183
WP_REACH_THRESHOLD = 0.15  # meters - advance waypoint when board is within this distance

# Board dimensions
BOARD_LENGTH = 2.0    # m (long side, along X in object frame)
BOARD_WIDTH = 1.0     # m (short side, along Y in object frame)
BOARD_THICKNESS = 0.10  # m
BOARD_MASS = 6.0      # kg


def _pd_gains_for_joint(joint_name: str) -> tuple[float, float]:
    base = joint_name.replace("_joint", "")
    for key, kp in G1_STIFFNESS.items():
        if key in base:
            return kp, G1_DAMPING[key]
    raise KeyError(f"No PD gains for joint {joint_name}")


def _build_kp_kd_tensors(device: str = "cuda:0") -> tuple[torch.Tensor, torch.Tensor]:
    kp = torch.zeros(23, dtype=torch.float32, device=device)
    kd = torch.zeros(23, dtype=torch.float32, device=device)
    for i, name in enumerate(G1_23DOF_JOINT_NAMES):
        kp[i], kd[i] = _pd_gains_for_joint(name)
    return kp, kd


def _make_robot_articulation_cfg(
    usd_path: str, prim_path: str, init_pos: tuple[float, float, float],
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> ArticulationCfg:
    """Build ArticulationCfg for one G1 robot in carrying posture."""
    joint_pos = {name: G1_23DOF_DEFAULT_POS_CARRY[i] for i, name in enumerate(G1_23DOF_JOINT_NAMES)}
    actuators = {}
    for i, name in enumerate(G1_23DOF_JOINT_NAMES):
        actuators[f"dof_{i}"] = IdealPDActuatorCfg(
            joint_names_expr=[name],
            effort_limit=G1_23DOF_EFFORT[i],
            velocity_limit=G1_23DOF_VEL_LIMIT[i],
            stiffness=0.0,
            damping=0.0,
            armature=G1_23DOF_ARMATURE[i],
            friction=G1_23DOF_FRICTION[i],
        )

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        actuators=actuators,
        prim_path=prim_path,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@configclass
class G1CooperativeTransportEnvCfg(DirectMARLEnvCfg):
    """Configuration for the 3-robot cooperative transport task."""

    # --- learning carriers ---
    possible_agents: list = ["robot_0", "robot_1"]

    # --- env timing ---
    episode_length_s: float = 30.0
    decimation: int = 200  # high-level at 1 Hz (200 * 0.005 = 1.0s)

    # --- action / observation spaces ---
    action_spaces: dict = {"robot_0": 3, "robot_1": 3}
    observation_spaces: dict = {"robot_0": OBS_TOTAL, "robot_1": OBS_TOTAL}
    state_space: int = 0
    state_spaces: dict = {"robot_0": 0, "robot_1": 0}

    # --- low-level policy ---
    low_level_policy_path: str = LOW_LEVEL_POLICY_PATH
    low_level_decimation: int = 4
    action_scale: float = 0.25
    ll_obs_dim: int = 83
    ll_action_dim: int = 23

    # --- obs scales ---
    obs_scale_ang_vel: float = 0.25
    obs_scale_dof_vel: float = 0.05

    # --- gait clock ---
    gait_frequency: float = 0.77

    # --- EMA action filter ---
    action_filter_alpha: float = 0.3

    # --- high-level action limits ---
    # NOTE: this must match how far the LL can actually travel in ONE HL period
    # (1 s), NOT the LL's target *sampling* range. The LL walks at ~0.2 m/s
    # (target_walk_speed=0.2 + overspeed penalty), so in 1 s it covers ~0.2 m.
    # The old value 0.4 created a dead-zone: any HL command in [0.2, 0.4] mapped
    # to the same ~0.2 m of real motion, wasting half the action range and
    # letting the policy sit in saturation -> coarse, oscillating ("shuffling") gait.
    max_target_dist: float = 0.2  # = target_walk_speed(0.2 m/s) * HL period(1 s)
    max_target_yaw: float = 0.3  # ~17 degrees/s, prevents excessive spinning

    # --- target interpolation ---
    # If True, the HL target is linearly slid from its current value toward
    # (robot_pos + command) across the 1 s HL window, so the LL sees a smoothly
    # MOVING target (like the manual rotation test) instead of one that snaps and
    # then sits still. A static target gets "reached" mid-window -> the robot
    # settles/marches in place -> the "walk-then-shuffle" gait. A continuously
    # moving target keeps the LL walking smoothly. Pairs with max_target_dist=0.2
    # so the moving target advances at ~0.2 m/s (the LL's walking speed).
    interpolate_target: bool = True
    ll_steps_per_hl: int = 50  # LL inferences per HL decision (HL 1 Hz / LL 50 Hz)

    # --- reward ---
    # Scaled so that scale * target_radius stays ~3 (was 10 * 0.3). With 1.0 m
    # waypoint spacing, keeping scale=10 would push the per-step distance penalty
    # up to -10 (= -300 over a 30 s episode), making it cheaper for the agent to
    # drop the board (-50 one-time) than to carry it -> "suicide" degeneracy.
    rew_dist_penalty_scale: float = 3.0
    rew_tilt_penalty_scale: float = 2.0
    rew_height_penalty_scale: float = 1.0
    rew_alive: float = 5.5
    rew_wp_reached_bonus: float = 10.0
    rew_death_penalty: float = -50.0
    rew_facing_penalty_scale: float = 2.0  # penalize robots not facing toward board center

    # --- normalized reward (v2) ---
    # Each continuous term is mapped to a bounded range so the weights below are the
    # *relative* priorities (instead of being entangled with raw metric magnitudes).
    # Distance is replaced by potential-based progress shaping, which is radius-invariant
    # (moving toward the waypoint at the reference speed gives +1 regardless of spacing),
    # removing the difficulty confound of the old linear distance penalty.
    use_normalized_reward: bool = True
    nrew_progress_speed: float = 0.2   # m/s reference; full +1 progress when closing at this speed
    nrew_w_progress: float = 2.5       # weight on potential-based progress, range [-w, w].
                                       # Raised from 1.0: posture/alive are UNCONDITIONAL
                                       # (paid for just standing upright) while progress is
                                       # CONDITIONAL (paid only for moving toward the wp). On
                                       # obstacle courses a warm-started policy can stall and farm
                                       # the posture floor; a larger progress gain makes "keep moving"
                                       # dominate that floor so the gradient favours advancing.
    nrew_w_wp: float = 5.0             # one-time bonus per waypoint reached
    nrew_w_tilt: float = 0.6           # exp(-(tilt/sigma)^2) reward, 1 = perfectly level
    nrew_w_height: float = 0.4         # exp(-(height_loss/sigma)^2) reward, 1 = at full height
    nrew_w_facing: float = 0.4         # exp(-(facing_err/sigma)^2) reward, 1 = aligned
    nrew_w_alive: float = 0.05         # small positive floor for staying alive (lowered from
                                       # 0.2 so standing still is barely rewarded; keeps the
                                       # anti-suicide floor without making "do nothing" lucrative)
    nrew_w_death: float = 50.0         # one-time penalty (subtracted) when the board drops
    nrew_sigma_tilt: float = 0.20      # tilt scale (m, summed corner deviation)
    nrew_sigma_height: float = 0.08    # height-loss scale (m below 0.97)
    nrew_sigma_facing: float = 0.5     # facing-error scale (rad)
    nrew_w_offset: float = 0.6         # exp(-(anchor_err/sigma)^2) reward; keeps each robot
                                       # at its board-relative holding anchor (offset beyond
                                       # the short edge): too close -> collide, too far -> drop
    nrew_sigma_offset: float = 0.15    # anchor-distance scale (m)
    # Board-heading alignment (TIER-1 "trick" term; default OFF): reward the board
    # LONG axis (board-X) parallel to the path tangent. This directly prescribes the
    # formation orientation, so obstacle avoidance stops depending on perception ->
    # LiDAR becomes decorative. Kept for A/B but disabled by default; the tier-2
    # setup drives pose from the perception penalty below instead. The long axis is a
    # line, so the error is folded mod pi to [0, pi/2].
    nrew_w_heading: float = 0.0        # exp(-(heading_err/sigma)^2) reward; 0 = tier-2 (off)
    nrew_sigma_heading: float = 0.25   # heading-error scale (rad); sharpened from 0.5 so
                                       # alignment is actually rewarded (at 16 deg err the
                                       # old sigma=0.5 already gave 73% of max -> no gradient)
    # Dense obstacle-proximity penalty (TIER-2, perception-driven avoidance; paper
    # Eq. 5). Penalizes the nearest wall distance SENSED by the two learning carriers'
    # LiDAR: quadratic ramp inside obstacle_penalty_dist, EXACTLY 0 beyond it (no cost
    # for open-corridor travel). This is the signal that makes the policy use LiDAR to
    # adjust the board's pose / squeeze through, replacing the heading trick. Kept
    # small vs progress(1.0)/wp(5.0) so it never discourages forward motion; it only
    # nudges the formation off walls and toward the corridor centerline.
    nrew_w_obstacle: float = 0.3       # weight; penalty range [0, w]
    obstacle_penalty_dist: float = 1.0 # m; wall-distance at/below which the penalty ramps up

    # --- termination ---
    termination_height: float = 0.5
    board_drop_height: float = 0.70  # early termination: board below 0.7m → fast fail, fast learn

    # --- obstacle-collision safety net (tier 2) ---
    # Paper-faithful avoidance comes from collision-free anchors + LiDAR, but the
    # board (0.97m) flies over the 0.71m walls, so wall contact only stalls the
    # robots indirectly. As a clean, death-level failure signal we terminate when
    # the formation footprint overlaps any obstacle AABB (robot bases inflated by
    # a body radius; board corners by a small margin). Off when no obstacles.
    use_obstacle_collision_termination: bool = True
    # Collision check uses the SAME footprint validated by check_user_path.py: a
    # board-CoM-centered rectangle with the long axis along the board heading, and
    # NO inflation by default. This guarantees the check is never more conservative
    # than the path validation, so the validated path won't false-terminate.
    formation_length: float = 2.7   # m, long axis (the robot_0 <-> robot_1 axis)
    formation_width: float = 1.35   # m, short axis
    obstacle_collision_margin: float = 0.0  # extra clearance on the board half-extents (m)
    # Collision now tests real solids (board OBB + 3 robot base circles) instead of
    # the monolithic 2.7x1.35 box, which over-counted the empty gaps and clipped
    # walls while board+robots still had clearance. This is the robot base radius.
    robot_collision_radius: float = 0.2  # m, real G1 base footprint radius
    nrew_w_collision: float = 50.0  # one-time penalty when collision TERMINATES the episode
    # Whether a footprint-vs-wall collision ENDS the episode. Terminating on first
    # contact means the policy dies before it can learn to graze a corner, back off
    # and recover -> it plateaus at the first hard turn. With this False the episode
    # continues and we instead apply a per-step contact penalty (below): the walls
    # are solid colliders so the formation still can't pass through, but it now gets
    # to practice nudging along a wall and correcting, which is the corner skill.
    obstacle_collision_terminates: bool = False
    nrew_w_collision_contact: float = 3.0  # per-step penalty while footprint overlaps a wall
                                           # (used only when obstacle_collision_terminates=False).
                                           # Lowered 10->3: with the NARROWED corridor, rotating
                                           # the board at a corner requires the +X-end robot to
                                           # sweep out near the wall, so brief scraping is
                                           # UNAVOIDABLE while learning the turn. At w=10 each
                                           # attempt cost -10/step, so the policy learned to avoid
                                           # the corner (stall) instead of pushing through -> the
                                           # penalty suppressed the very exploration needed. w=3
                                           # keeps contact NET-NEGATIVE (still repels wall-farming)
                                           # but leaves room to try the sweep/recover maneuver;
                                           # once learned, contact drops and so does the penalty.

    # --- goal completion (fixed-path / maze mode) ---
    # In the maze we don't want a fixed time budget to define success; success is
    # "the board reached the final waypoint of the fixed path". When the last
    # waypoint is consumed (_path_idx == P) we (a) hand out a one-time goal bonus
    # and (b) optionally terminate the episode as a SUCCESS (not a death). Pair
    # this with a generous episode_length_s so slow-but-successful runs aren't
    # cut off by the time limit. Only active when use_fixed_path is True.
    terminate_on_goal: bool = True
    nrew_w_goal: float = 20.0  # one-time bonus (added) when the final waypoint is reached

    # --- board physics ---
    board_friction: float = 5.0

    # --- static obstacles ---
    # Fixed rectangular obstacles spawned as static collision bodies (no rigid
    # body -> immovable, like a wall). Each entry is (size_xyz, pos_xyz) in the
    # env-local frame (same frame as the board init pos). z is the box CENTER,
    # so a box of height h resting on the ground has pos_z = h / 2.
    # Default walls (env-local boxes: size, centre), used when no scene is given;
    # train.sh always replaces them with the tilt-gate scene (--scene_json).
    # Leave empty to disable.
    obstacle_specs: list = [
        ((5.05, 0.25, 0.71), (1.55, -0.695, 0.355)),
        ((7.7, 0.25, 0.71), (0.225, -3.945, 0.355)),
        ((0.25, 1.675, 0.71), (-1.1, 0.0175, 0.355)),
        ((0.25, 8.075, 0.71), (-3.75, -0.0325, 0.355)),
        ((5.05, 0.25, 0.71), (-1.1, 3.88, 0.355)),
        ((0.25, 3.15, 0.71), (1.55, 3.505, 0.355)),
        ((2.4, 0.25, 0.71), (2.875, 4.955, 0.355)),
        ((0.25, 9.15, 0.71), (4.2, 0.505, 0.355)),
    ]

    # --- formation start position ---
    # Env-local (x, y) where the whole carrying formation (board + 3 robots)
    # spawns. The obstacle coordinate origin stays at (0, 0); moving the start
    # here shifts board, all robots, and the waypoint origin together. Set to
    # the first path waypoint so the board spawns at the course entrance.
    start_xy: tuple = (2.0, -2.32)

    # --- fixed waypoint path (obstacle course) ---
    # When True, the sliding window follows this fixed env-local path instead of
    # the random walk (open-field curriculum). The board spawns at
    # path_waypoints[0] (== start_xy); the first active target is
    # path_waypoints[1]. Resampled to ~0.5 m spacing with all corners preserved
    # (matches the 0.5 m policy working point; validated 0-collision footprint).
    use_fixed_path: bool = True
    path_waypoints: list = [
        (2.0, -2.32), (1.5, -2.32), (1.0, -2.32), (0.5, -2.32),
        (0.0, -2.32), (-0.5, -2.32), (-1.0, -2.32), (-1.325, -2.32),
        (-1.65, -2.32), (-1.895, -2.07), (-2.14, -1.82), (-2.14, -1.335),
        (-2.14, -0.85), (-2.14, -0.365), (-2.14, 0.12), (-2.14, 0.62),
        (-2.14, 0.87), (-2.14, 1.12), (-2.14, 1.62), (-1.87, 1.925),
        (-1.6, 2.23), (-1.15, 2.23), (-0.7, 2.23), (-0.32, 1.978),
        (0.06, 1.726), (0.44, 1.474), (0.82, 1.222), (1.2, 0.97),
        (1.7, 0.97), (1.95, 0.97), (2.2, 0.97), (2.425, 1.32),
        (2.65, 1.67), (2.65, 2.145), (2.65, 2.62), (2.65, 3.0),
    ]

    # --- LiDAR perception (paper Eq. 5) ---
    # lidar_num_rays synthetic rays evenly spaced over 360 deg in each learning
    # carrier's body frame; each detection d is normalized as
    # 1 - min(d, dmax)/dmax (0 = far / no hit, ->1 = wall right at the robot).
    # Rays are cast against the static obstacle AABBs.
    lidar_num_rays: int = 36
    lidar_max_range: float = 5.0
    # Force the LiDAR channels of the observation to 0. Use when warm-starting a
    # policy trained WITHOUT obstacles (open-field phase 1, LiDAR always 0): those
    # 36 input weights were never trained (constant-zero input -> zero gradient ->
    # stuck at init), so activating LiDAR on an obstacle course injects garbage into every
    # hidden unit and breaks the walker. Masking keeps them dead so the transferred
    # policy stays intact and follows the (collision-free) path via waypoints/heading.
    mask_lidar_obs: bool = False

    # --- target ---
    # Used directly only when the curriculum is OFF. With the curriculum ON, the
    # live spacing starts at target_radius_start and ramps to target_radius_end.
    target_radius: float = 1.0  # waypoint-to-waypoint spacing (was 0.3); harder task

    # --- target_radius curriculum (performance-gated) ---
    # Jumping straight from 0.3 m to 1.0 m spacing stalls fine-tuning (the warm
    # policy's exploration has collapsed and can't bridge the 3.3x distribution
    # shift). Instead ramp the spacing: every curriculum_window finished episodes,
    # if the rolling-mean wp_reached >= curriculum_wp_threshold, bump the spacing by
    # one step (up to target_radius_end). The policy only ever faces a small step up.
    use_target_radius_curriculum: bool = True
    target_radius_start: float = 0.3   # where the warm policy is already competent
    target_radius_end: float = 1.0     # final desired spacing
    target_radius_curriculum_step: float = 0.1
    curriculum_window: int = 300       # finished episodes per curriculum evaluation
    # Advancement is TREND-based: we keep a history of per-window mean wp_reached and
    # advance only when the average of the last `patience` windows is NOT higher than
    # the average of the `patience` windows before that (by more than improve_eps) --
    # i.e. wp_reached has genuinely stopped trending up over a 2*patience-window span.
    # This is more robust than "N consecutive windows below the all-time best":
    #   - a temporary plateau mid-rise does NOT trigger advance (during the later rise
    #     the recent half > previous half, so we keep dwelling);
    #   - a single noisy-high window cannot freeze a virtual "best" that real but slow
    #     progress can never beat (we compare moving averages, not the max).
    # Easy levels (flat from the start) still pass quickly; hard levels dwell until the
    # upward trend actually flattens.
    curriculum_patience: int = 5       # windows per half of the trend comparison
    curriculum_improve_eps: float = 0.15  # min rise of recent half over previous half
    # Small FLOOR (units = logged "total num timesteps" = num_envs * steps) the
    # policy must spend at a level before it can advance, even if the threshold is
    # already met. This is NOT meant to be long: it just (1) lets the 5-slot sliding
    # window flush its old-radius waypoints so wp_reached reflects the TRUE new
    # spacing (otherwise the queued easy points keep wp high and the curriculum
    # blasts to 1.0 in a few iterations), and (2) gives a few real updates + a clean
    # eval window. Beyond this floor the wp threshold decides, so easy levels pass
    # almost immediately and hard levels dwell as long as needed. ~50k ≈ 13 iters.
    curriculum_min_steps_per_level: int = 50000

    # --- robot USD ---
    training_g1_usd_path: str = TRAINING_G1_USD_PATH

    # --- Robot initial positions (relative to board center at origin) ---
    # Board: 2.0m along X (long), 1.0m along Y (short)
    # Robot 0: short edge at +X (facing -X toward board center)
    # Robot 1: short edge at -X (facing +X toward board center)
    # Robot 2: long edge at +Y (facing -Y toward board center)
    robot_offset: float = 0.20  # clearance from board edge to robot center

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=200,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=10.0, replicate_physics=True
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class G1CooperativeTransportEnv(DirectMARLEnv):
    """3-robot cooperative transport with waist-tracking low-level policy.

    Scene: 3 G1 robots in carrying posture around a rectangular board.
    Task: Move the board's center to a randomly generated target.
    Control: the high-level policy outputs [dx, dy, dyaw] per carrier → waist tracking LL policy.
    """

    cfg: G1CooperativeTransportEnvCfg

    def __init__(self, cfg: G1CooperativeTransportEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Load pretrained low-level policy (shared across all robots)
        policy_path = cfg.low_level_policy_path
        if not policy_path or not os.path.exists(policy_path):
            raise FileNotFoundError(f"Low-level policy not found: '{policy_path}'.")
        self._ll_policy = torch.jit.load(policy_path, map_location=self.device).eval()

        # Per-robot DOF mappings (all robots including followers)
        self._dof_ids = {}
        for agent_id in ALL_ROBOT_IDS:
            ids, _ = self.robots[agent_id].find_joints(G1_23DOF_JOINT_NAMES, preserve_order=True)
            self._dof_ids[agent_id] = torch.tensor(ids, device=self.device, dtype=torch.long)

        # Default joint positions for carrying posture
        self._default_dof_pos = torch.tensor(
            G1_23DOF_DEFAULT_POS_CARRY, device=self.device, dtype=torch.float32
        ).unsqueeze(0)  # (1, 23)

        # PD gains
        self._kp, self._kd = _build_kp_kd_tensors(self.device)

        # Torque limits
        self._torque_limits = torch.tensor(G1_23DOF_EFFORT, device=self.device, dtype=torch.float32)

        # Locked DOF mask
        self._locked_mask = torch.zeros(cfg.ll_action_dim, device=self.device, dtype=torch.bool)
        self._locked_mask[LOCKED_DOF_INDICES] = True

        # Per-robot buffers (all robots including followers)
        self._ll_actions = {
            aid: torch.zeros(self.num_envs, cfg.ll_action_dim, device=self.device)
            for aid in ALL_ROBOT_IDS
        }
        self._filtered_actions = {
            aid: torch.zeros(self.num_envs, cfg.ll_action_dim, device=self.device)
            for aid in ALL_ROBOT_IDS
        }
        self._target_cmd = {
            aid: torch.zeros(self.num_envs, 3, device=self.device)
            for aid in ALL_ROBOT_IDS
        }
        self._target_pos_world = {
            aid: torch.zeros(self.num_envs, 2, device=self.device)
            for aid in ALL_ROBOT_IDS
        }
        self._target_yaw_world = {
            aid: torch.zeros(self.num_envs, device=self.device)
            for aid in ALL_ROBOT_IDS
        }
        # Interpolation endpoints: target slides from _start -> _final over the HL
        # window (only used for the learning carriers when cfg.interpolate_target is True).
        self._target_start_pos = {
            aid: torch.zeros(self.num_envs, 2, device=self.device)
            for aid in ALL_ROBOT_IDS
        }
        self._target_final_pos = {
            aid: torch.zeros(self.num_envs, 2, device=self.device)
            for aid in ALL_ROBOT_IDS
        }
        self._target_start_yaw = {
            aid: torch.zeros(self.num_envs, device=self.device)
            for aid in ALL_ROBOT_IDS
        }
        self._target_final_yaw = {
            aid: torch.zeros(self.num_envs, device=self.device)
            for aid in ALL_ROBOT_IDS
        }
        self._hl_ll_count = 0  # LL steps elapsed since last HL decision (for interp)
        self._gait_phase = {
            aid: torch.zeros(self.num_envs, device=self.device)
            for aid in ALL_ROBOT_IDS
        }

        self._ll_step_counter = 0

        # Waypoint buffers: 5 visible waypoints per env (sliding window)
        self._waypoints = torch.zeros(self.num_envs, NUM_WAYPOINTS, 2, device=self.device)
        # Distance to the active waypoint at the previous HL step, for potential-based
        # progress shaping. Reset to the spawn distance in _reset_idx so the first step
        # yields ~0 progress (no artificial jump).
        self._prev_dist_to_wp = torch.zeros(self.num_envs, device=self.device)
        self._wp_reached_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Fixed-path buffers (obstacle course). _path holds the env-local path;
        # _path_idx is the per-env index of the active (next) target waypoint.
        if self.cfg.use_fixed_path and len(self.cfg.path_waypoints) > 0:
            self._path = torch.tensor(
                self.cfg.path_waypoints, dtype=torch.float32, device=self.device
            )  # (P, 2), env-local
        else:
            self._path = torch.zeros(0, 2, device=self.device)
        self._path_idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        # Per-env latch: True once the board has consumed the final waypoint.
        # Drives the one-time goal bonus and (optionally) success termination.
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Static obstacle AABBs in env-local frame: [x0, x1, y0, y1] per box, for
        # LiDAR ray casting. Same for every env (obstacles cloned per tile).
        if len(self.cfg.obstacle_specs) > 0:
            aabb = []
            for size, pos in self.cfg.obstacle_specs:
                aabb.append([
                    pos[0] - size[0] / 2.0, pos[0] + size[0] / 2.0,
                    pos[1] - size[1] / 2.0, pos[1] + size[1] / 2.0,
                ])
            self._obs_aabb = torch.tensor(aabb, dtype=torch.float32, device=self.device)  # (M, 4)
        else:
            self._obs_aabb = torch.zeros(0, 4, device=self.device)

        # Evenly-spaced ray offsets (body frame), 360 deg, for the synthetic LiDAR.
        self._lidar_ray_offsets = torch.linspace(
            0.0, 2.0 * math.pi, self.cfg.lidar_num_rays + 1, device=self.device
        )[:-1]  # (R,)

        # Live (curriculum-controlled) waypoint spacing + rolling wp_reached history.
        self._curr_target_radius = (
            self.cfg.target_radius_start
            if self.cfg.use_target_radius_curriculum
            else self.cfg.target_radius
        )
        self._curr_wp_history: list[float] = []
        self._level_step_counter = 0  # env-steps spent at the current curriculum level
        self._level_wp_means: list[float] = []  # per-window mean wp_reached at this level

        # Temporal observation buffer (only for the learning carriers)
        self._obs_history = {
            aid: torch.zeros(self.num_envs, OBS_TEMPORAL_FRAMES + 1, OBS_CORE_PER_FRAME, device=self.device)
            for aid in self.cfg.possible_agents
        }

    def _setup_scene(self):
        """Set up scene with 3 robots, 1 board, and ground plane."""
        cfg = self.cfg

        # Compute robot positions around the board
        half_w = BOARD_WIDTH / 2.0   # 0.5 (short dimension, along Y)
        half_l = BOARD_LENGTH / 2.0  # 1.0 (long dimension, along X)
        offset = cfg.robot_offset
        # From diagnostic: forearm/wrist at z≈0.895, top surface ≈0.92
        # Board bottom should rest ON TOP of forearm: 0.92 + thickness/2 = 0.97 → use 0.97
        board_z = 0.97

        # Whole formation spawns at this env-local start (obstacle origin stays 0,0).
        sx, sy = cfg.start_xy

        # Robot 0: SHORT edge at +X, facing -X (toward board center)
        r0_pos = (sx + half_l + offset, sy + 0.0, 0.8)
        r0_rot = _yaw_to_quat(math.pi)  # facing -X

        # Robot 1: SHORT edge at -X, facing +X (toward board center)
        r1_pos = (sx - (half_l + offset), sy + 0.0, 0.8)
        r1_rot = _yaw_to_quat(0.0)  # facing +X

        # Robot 2: LONG edge at +Y, facing -Y (toward board center)
        r2_pos = (sx + 0.0, sy + half_w + offset, 0.8)
        r2_rot = _yaw_to_quat(-math.pi / 2)  # facing -Y

        # Create robot articulations
        self.robots = {}
        robot_configs = [
            ("robot_0", r0_pos, r0_rot),
            ("robot_1", r1_pos, r1_rot),
            ("robot_2", r2_pos, r2_rot),
        ]
        for agent_id, pos, rot in robot_configs:
            art_cfg = _make_robot_articulation_cfg(
                usd_path=cfg.training_g1_usd_path,
                prim_path=f"/World/envs/env_.*/{agent_id}",
                init_pos=pos,
                init_rot=rot,
            )
            self.robots[agent_id] = Articulation(art_cfg)

        # Create the board (rigid body)
        board_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Board",
            spawn=sim_utils.CuboidCfg(
                size=(BOARD_LENGTH, BOARD_WIDTH, BOARD_THICKNESS),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=BOARD_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=cfg.board_friction,
                    dynamic_friction=cfg.board_friction,
                    restitution=0.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(sx, sy, board_z),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self._board = RigidObject(board_cfg)

        # Static obstacles (fixed rectangular collision bodies, no rigid body).
        # Spawned before cloning so they replicate into every env at the same
        # env-local position. They do not move, so they are not registered in
        # the scene tensors.
        for i, (size, pos) in enumerate(cfg.obstacle_specs):
            obs_spawn = sim_utils.CuboidCfg(
                size=tuple(size),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
                # NOTE: no visual_material — PreviewSurfaceCfg's spawn path is broken
                # on Isaac Sim 5.1 (CreateShaderPrimFromSdrCommand rejects 'name').
                # Obstacles render with the default material; collision is unaffected.
                # No rigid_props / mass_props -> static collider (immovable).
            )
            obs_spawn.func(f"/World/envs/env_.*/Obstacle_{i}", obs_spawn, translation=tuple(pos))

        # Ground plane
        cfg.terrain.num_envs = self.scene.cfg.num_envs
        cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = cfg.terrain.class_type(cfg.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Register assets with scene (all robots including followers)
        for agent_id in ALL_ROBOT_IDS:
            self.scene.articulations[agent_id] = self.robots[agent_id]
        self.scene.rigid_objects["board"] = self._board

        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # DirectMARLEnv interface
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: dict):
        """Process the high-level actions of the learning carriers.

        Convert body-frame displacement commands to world-space targets.
        The actual LL observation (live error) is recomputed every LL step in _build_ll_obs.
        """
        for agent_id in self.cfg.possible_agents:
            act = actions[agent_id]
            dx_body = act[:, 0].clamp(-self.cfg.max_target_dist, self.cfg.max_target_dist)
            dy_body = act[:, 1].clamp(-self.cfg.max_target_dist, self.cfg.max_target_dist)
            dyaw = act[:, 2].clamp(-self.cfg.max_target_yaw, self.cfg.max_target_yaw)

            # Current robot state
            robot = self.robots[agent_id]
            current_pos = robot.data.root_pos_w[:, :2]  # (N, 2)
            current_yaw = _get_heading_yaw(robot.data.root_quat_w)  # (N,)

            # Convert body-frame displacement to world-frame target
            cos_yaw = torch.cos(current_yaw)
            sin_yaw = torch.sin(current_yaw)
            dx_world = cos_yaw * dx_body - sin_yaw * dy_body
            dy_world = sin_yaw * dx_body + cos_yaw * dy_body

            final_x = current_pos[:, 0] + dx_world
            final_y = current_pos[:, 1] + dy_world
            final_yaw = current_yaw + dyaw

            if self.cfg.interpolate_target:
                # Anchor at where the target currently is (continuous across HL steps),
                # then slide toward the new final over the HL window (see _apply_action).
                self._target_start_pos[agent_id][:, 0] = self._target_pos_world[agent_id][:, 0]
                self._target_start_pos[agent_id][:, 1] = self._target_pos_world[agent_id][:, 1]
                self._target_start_yaw[agent_id][:] = self._target_yaw_world[agent_id]
                self._target_final_pos[agent_id][:, 0] = final_x
                self._target_final_pos[agent_id][:, 1] = final_y
                self._target_final_yaw[agent_id][:] = final_yaw
            else:
                self._target_pos_world[agent_id][:, 0] = final_x
                self._target_pos_world[agent_id][:, 1] = final_y
                self._target_yaw_world[agent_id][:] = final_yaw

        if self.cfg.interpolate_target:
            self._hl_ll_count = 0

    def _apply_action(self):
        """Run low-level policy for each robot at decimated rate."""
        # Slide each carrier target from _start toward _final across the HL window so
        # the LL tracks a smoothly moving point. Runs once per LL step (same gate as
        # the LL inference below), before _build_ll_obs reads _target_pos_world.
        if self.cfg.interpolate_target and (self._ll_step_counter % self.cfg.low_level_decimation == 0):
            self._hl_ll_count += 1
            frac = min(self._hl_ll_count / float(self.cfg.ll_steps_per_hl), 1.0)
            for aid in self.cfg.possible_agents:
                sp = self._target_start_pos[aid]
                fp = self._target_final_pos[aid]
                self._target_pos_world[aid][:, 0] = sp[:, 0] + frac * (fp[:, 0] - sp[:, 0])
                self._target_pos_world[aid][:, 1] = sp[:, 1] + frac * (fp[:, 1] - sp[:, 1])
                sy = self._target_start_yaw[aid]
                fy = self._target_final_yaw[aid]
                dyaw_interp = torch.atan2(torch.sin(fy - sy), torch.cos(fy - sy))
                self._target_yaw_world[aid][:] = sy + frac * dyaw_interp

        for agent_id in ALL_ROBOT_IDS:
            robot = self.robots[agent_id]
            dof_ids = self._dof_ids[agent_id]

            if self._ll_step_counter % self.cfg.low_level_decimation == 0:
                ll_obs = self._build_ll_obs(agent_id)
                with torch.no_grad():
                    self._ll_actions[agent_id][:] = self._ll_policy(ll_obs)

                self._ll_actions[agent_id][:, self._locked_mask] = 0.0

                alpha = self.cfg.action_filter_alpha
                self._filtered_actions[agent_id].mul_(1.0 - alpha).add_(
                    self._ll_actions[agent_id], alpha=alpha
                )

            # PD torque computation
            actions_scaled = self._filtered_actions[agent_id] * self.cfg.action_scale
            current_pos = robot.data.joint_pos[:, dof_ids]
            current_vel = robot.data.joint_vel[:, dof_ids]
            target_pos = actions_scaled + self._default_dof_pos
            torques_23 = self._kp * (target_pos - current_pos) - self._kd * current_vel
            torques_23 = torques_23.clamp(-self._torque_limits, self._torque_limits)

            full_efforts = torch.zeros_like(robot.data.joint_pos)
            full_efforts[:, dof_ids] = torques_23
            robot.set_joint_effort_target(full_efforts)

        # Advance gait phase for all robots
        dt = self.cfg.sim.dt
        for agent_id in ALL_ROBOT_IDS:
            self._gait_phase[agent_id] += 2.0 * math.pi * self.cfg.gait_frequency * dt
            self._gait_phase[agent_id] = self._gait_phase[agent_id] % (2.0 * math.pi)

        self._ll_step_counter += 1

    def _get_observations(self) -> dict:
        """Build per-agent observations with temporal stacking (paper-aligned)."""
        obs_dict = {}

        # Get board state (shared across agents)
        board_pos = self._board.data.root_pos_w  # (N, 3)
        board_quat = self._board.data.root_quat_w  # (N, 4)
        board_vel = self._board.data.root_lin_vel_w  # (N, 3)
        board_xy = board_pos[:, :2]  # (N, 2)

        # Board corners in world frame (4 corners of the top surface)
        corners_local = torch.tensor([
            [BOARD_LENGTH / 2, BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [BOARD_LENGTH / 2, -BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [-BOARD_LENGTH / 2, BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [-BOARD_LENGTH / 2, -BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
        ], device=self.device, dtype=torch.float32)  # (4, 3)

        corners_world = _transform_points(corners_local, board_pos, board_quat)  # (N, 4, 3)
        board_yaw = _get_heading_yaw(board_quat)  # (N,)

        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)

        for i, agent_id in enumerate(self.cfg.possible_agents):
            robot = self.robots[agent_id]
            root_pos = robot.data.root_pos_w      # (N, 3)
            root_quat = robot.data.root_quat_w    # (N, 4)
            root_lin_vel = robot.data.root_lin_vel_b  # (N, 3) already body frame
            robot_xy = root_pos[:, :2]  # (N, 2)
            robot_yaw = _get_heading_yaw(root_quat)  # (N,)
            cos_yaw = torch.cos(robot_yaw)  # (N,)
            sin_yaw = torch.sin(robot_yaw)  # (N,)

            # --- psi_task (10D): 5 waypoints in robot body frame ---
            wp_rel_world = self._waypoints - robot_xy.unsqueeze(1)  # (N, 5, 2)
            wp_rel_body_x = cos_yaw.unsqueeze(1) * wp_rel_world[:, :, 0] + sin_yaw.unsqueeze(1) * wp_rel_world[:, :, 1]
            wp_rel_body_y = -sin_yaw.unsqueeze(1) * wp_rel_world[:, :, 0] + cos_yaw.unsqueeze(1) * wp_rel_world[:, :, 1]
            psi_task = torch.stack([wp_rel_body_x, wp_rel_body_y], dim=-1).reshape(self.num_envs, -1)  # (N, 10)

            # --- psi_ego (7D): board-relative info in body frame ---
            # Board center in robot body frame (2D)
            board_rel_world = board_xy - robot_xy  # (N, 2)
            board_rel_body_x = cos_yaw * board_rel_world[:, 0] + sin_yaw * board_rel_world[:, 1]
            board_rel_body_y = -sin_yaw * board_rel_world[:, 0] + cos_yaw * board_rel_world[:, 1]
            # Velocity in body frame (2D) - already body frame
            ego_vel_xy = root_lin_vel[:, :2]  # (N, 2)
            # Relative yaw: board heading - robot heading (1D)
            yaw_diff = board_yaw - robot_yaw
            yaw_diff_wrapped = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).unsqueeze(-1)  # (N, 1)
            # Height and pitch (invariant to yaw)
            com_height = root_pos[:, 2:3]  # (N, 1)
            gravity_b = quat_rotate_inverse(root_quat, gravity_w)
            torso_pitch = gravity_b[:, 0:1]  # (N, 1)

            psi_ego = torch.cat([
                board_rel_body_x.unsqueeze(-1), board_rel_body_y.unsqueeze(-1),
                ego_vel_xy, yaw_diff_wrapped, com_height, torso_pitch,
            ], dim=-1)  # (N, 7)

            # --- psi_partners (7D × 2 = 14D): partner states in robot body frame ---
            partners = [aid for aid in ALL_ROBOT_IDS if aid != agent_id]
            psi_partners = []
            for partner_id in partners:
                p_robot = self.robots[partner_id]
                p_pos = p_robot.data.root_pos_w
                p_quat = p_robot.data.root_quat_w
                p_vel = p_robot.data.root_lin_vel_w  # world frame vel

                # Partner position in my body frame (2D)
                p_rel_world = p_pos[:, :2] - robot_xy  # (N, 2)
                p_rel_body_x = cos_yaw * p_rel_world[:, 0] + sin_yaw * p_rel_world[:, 1]
                p_rel_body_y = -sin_yaw * p_rel_world[:, 0] + cos_yaw * p_rel_world[:, 1]
                # Partner velocity in my body frame (2D)
                p_vel_body_x = cos_yaw * p_vel[:, 0] + sin_yaw * p_vel[:, 1]
                p_vel_body_y = -sin_yaw * p_vel[:, 0] + cos_yaw * p_vel[:, 1]
                # Partner yaw relative to me (1D)
                p_yaw = _get_heading_yaw(p_quat)
                p_yaw_rel = p_yaw - robot_yaw
                p_yaw_rel_wrapped = torch.atan2(torch.sin(p_yaw_rel), torch.cos(p_yaw_rel)).unsqueeze(-1)
                # Partner height (1D)
                p_height = p_pos[:, 2:3]
                # Partner torso pitch (1D)
                p_gravity_b = quat_rotate_inverse(p_quat, gravity_w)
                p_pitch = p_gravity_b[:, 0:1]

                psi_p = torch.cat([
                    p_rel_body_x.unsqueeze(-1), p_rel_body_y.unsqueeze(-1),
                    p_vel_body_x.unsqueeze(-1), p_vel_body_y.unsqueeze(-1),
                    p_yaw_rel_wrapped, p_height, p_pitch,
                ], dim=-1)  # (N, 7)
                psi_partners.append(psi_p)

            psi_ptn = torch.cat(psi_partners, dim=-1)  # (N, 14)

            # --- psi_obj (18D): board geometry in robot body frame ---
            # 4 corners relative to robot, in body frame (4 × 3 = 12D)
            corners_rel = corners_world - root_pos.unsqueeze(1)  # (N, 4, 3)
            corners_body_x = cos_yaw.unsqueeze(1) * corners_rel[:, :, 0] + sin_yaw.unsqueeze(1) * corners_rel[:, :, 1]
            corners_body_y = -sin_yaw.unsqueeze(1) * corners_rel[:, :, 0] + cos_yaw.unsqueeze(1) * corners_rel[:, :, 1]
            corners_body_z = corners_rel[:, :, 2]
            corners_body = torch.stack([corners_body_x, corners_body_y, corners_body_z], dim=-1)  # (N, 4, 3)
            # Board CoM relative to robot, in body frame (3D)
            board_com_rel = board_pos - root_pos  # (N, 3)
            board_com_body_x = cos_yaw * board_com_rel[:, 0] + sin_yaw * board_com_rel[:, 1]
            board_com_body_y = -sin_yaw * board_com_rel[:, 0] + cos_yaw * board_com_rel[:, 1]
            board_com_body_z = board_com_rel[:, 2]
            # Board velocity in robot body frame (3D)
            board_vel_body_x = cos_yaw * board_vel[:, 0] + sin_yaw * board_vel[:, 1]
            board_vel_body_y = -sin_yaw * board_vel[:, 0] + cos_yaw * board_vel[:, 1]
            board_vel_body_z = board_vel[:, 2]

            psi_obj = torch.cat([
                corners_body.reshape(self.num_envs, 12),
                board_com_body_x.unsqueeze(-1), board_com_body_y.unsqueeze(-1), board_com_body_z.unsqueeze(-1),
                board_vel_body_x.unsqueeze(-1), board_vel_body_y.unsqueeze(-1), board_vel_body_z.unsqueeze(-1),
            ], dim=-1)  # (N, 18)

            # LiDAR: synthetic 360-deg rangefinder vs static obstacles (paper Eq. 5)
            robot_xy_local = robot_xy - self.scene.env_origins[:, :2]
            psi_lidar = self._compute_lidar_normalized(robot_xy_local, robot_yaw)  # (N, 36)
            if self.cfg.mask_lidar_obs:
                psi_lidar = torch.zeros_like(psi_lidar)

            # Assemble single-frame observation
            obs_core = torch.cat([
                psi_task, psi_ego, psi_ptn, psi_obj,
            ], dim=-1)  # (N, 49)

            obs_current = torch.cat([obs_core, psi_lidar], dim=-1)  # (N, 85)

            # Update temporal buffer: shift and insert new core frame (no LiDAR)
            self._obs_history[agent_id] = self._obs_history[agent_id].roll(1, dims=1)
            self._obs_history[agent_id][:, 0] = obs_core

            # Final observation: [current_frame(85), prev_core_1(49), prev_core_2(49)] = 183
            temporal_buffer = self._obs_history[agent_id][:, 1:].reshape(self.num_envs, -1)  # (N, 98)
            obs_stacked = torch.cat([obs_current, temporal_buffer], dim=-1)  # (N, 183)

            obs_dict[agent_id] = obs_stacked

        return obs_dict

    def _get_rewards(self) -> dict:
        """Compute shared cooperative reward."""
        # Count env-steps spent at the current curriculum level (matches the logged
        # "total num timesteps" units = num_envs * steps).
        if self.cfg.use_target_radius_curriculum and not self.cfg.use_fixed_path:
            self._level_step_counter += self.num_envs

        board_pos = self._board.data.root_pos_w
        board_quat = self._board.data.root_quat_w
        board_xy = board_pos[:, :2]
        board_z = board_pos[:, 2]

        # Active waypoint: always the first in the sliding window
        active_wp = self._waypoints[:, 0]  # (N, 2)
        cur_dist = torch.norm(active_wp - board_xy, dim=-1)

        # Potential-based progress shaping against the SAME waypoint as last step.
        # +1 when closing at the reference speed, -1 when receding at it, ~0 when static.
        # Telescopes over a trajectory (no free reward from oscillating) and is
        # independent of the absolute waypoint spacing -> radius-invariant.
        hl_dt = self.cfg.sim.dt * self.cfg.decimation
        v_ref = max(self.cfg.nrew_progress_speed * hl_dt, 1e-6)
        progress = torch.clamp((self._prev_dist_to_wp - cur_dist) / v_ref, -1.0, 1.0)

        # Sliding window: when reached, advance to the next waypoint.
        reached = cur_dist < WP_REACH_THRESHOLD
        if self.cfg.use_fixed_path and self._path.shape[0] > 0:
            # Once the final waypoint is consumed (idx == P sentinel), stop
            # counting/bonusing so the board can't farm the wp bonus at the goal.
            reached = reached & (self._path_idx < self._path.shape[0])
        if reached.any():
            reached_ids = reached.nonzero(as_tuple=False).view(-1)
            self._advance_waypoints(reached_ids)
            self._wp_reached_count[reached_ids] += 1

        # Distance to the (possibly new) active waypoint; also the baseline for next step.
        active_wp = self._waypoints[:, 0]
        dist_to_wp = torch.where(reached, torch.norm(active_wp - board_xy, dim=-1), cur_dist)
        self._prev_dist_to_wp = dist_to_wp

        # Tilt: summed deviation of corner z-values from their mean (0 = perfectly level)
        corners_local = torch.tensor([
            [BOARD_LENGTH / 2, BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [BOARD_LENGTH / 2, -BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [-BOARD_LENGTH / 2, BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [-BOARD_LENGTH / 2, -BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
        ], device=self.device, dtype=torch.float32)
        corners_world = _transform_points(corners_local, board_pos, board_quat)  # (N, 4, 3)
        corner_z = corners_world[:, :, 2]  # (N, 4)
        z_mean = corner_z.mean(dim=-1, keepdim=True)
        tilt = (corner_z - z_mean).abs().sum(dim=-1)  # (N,)

        # Height loss: how far the board is below its initial height (0.97m)
        height_loss = torch.clamp(0.97 - board_z, min=0.0)

        # Board "dying": dropped below threshold (also triggers termination)
        board_dying = (board_z < self.cfg.board_drop_height).float()

        # Wall collision (tier-2 safety net): death-level one-time penalty, paired
        # with the matching termination in _get_dones.
        collided = self._compute_obstacle_collision().float()

        # Facing error: robots must align with board's local frame (prevents board rotation)
        # Robot_0 at +X end: should face board's local -X → desired_yaw = board_yaw + π
        # Robot_1 at -X end: should face board's local +X → desired_yaw = board_yaw
        board_yaw = _get_heading_yaw(board_quat)  # (N,)
        desired_yaws = {
            "robot_0": board_yaw + math.pi,  # face -X in board frame
            "robot_1": board_yaw,            # face +X in board frame
        }
        per_robot_facing_err = []
        for aid in self.cfg.possible_agents:
            robot = self.robots[aid]
            robot_yaw = _get_heading_yaw(robot.data.root_quat_w)  # (N,)
            angle_diff = desired_yaws[aid] - robot_yaw
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            per_robot_facing_err.append(angle_diff.abs())
        per_robot_facing_err = torch.stack(per_robot_facing_err, dim=0)  # (A, N)
        facing_error = per_robot_facing_err.mean(dim=0)  # mean L1 (legacy term + logging)

        # Offset-keeping: each controlled robot should stay at its holding anchor
        # relative to the board. The anchor is `offset` beyond its short edge, on the
        # board's local +X (robot_0) / -X (robot_1) axis, laterally centered. We measure
        # the robot's base position in the board's local frame and penalize the distance
        # to that anchor: too close -> collides into the board, too far -> loses grip.
        cos_y = torch.cos(board_yaw)
        sin_y = torch.sin(board_yaw)
        anchor_local_x = {
            "robot_0": BOARD_LENGTH / 2.0 + self.cfg.robot_offset,
            "robot_1": -(BOARD_LENGTH / 2.0 + self.cfg.robot_offset),
        }
        per_robot_offset_err = []
        for aid in self.cfg.possible_agents:
            rel = self.robots[aid].data.root_pos_w[:, :2] - board_xy  # world (N, 2)
            local_x = cos_y * rel[:, 0] + sin_y * rel[:, 1]
            local_y = -sin_y * rel[:, 0] + cos_y * rel[:, 1]
            err = torch.sqrt((local_x - anchor_local_x[aid]) ** 2 + local_y ** 2 + 1e-8)
            per_robot_offset_err.append(err)
        per_robot_offset_err = torch.stack(per_robot_offset_err, dim=0)  # (A, N)
        offset_error = per_robot_offset_err.mean(dim=0)  # for logging

        if self.cfg.use_normalized_reward:
            # --- board-heading alignment (computed FIRST so it can gate progress) ---
            # We want the board's -X axis to point ALONG the path tangent (DIRECTIONAL:
            # +X-leading is a full error). board_yaw is the board +X direction, so -X is
            # board_yaw + pi. Tangent = active->next waypoint (stable); fall back to
            # board->active waypoint when the segment is degenerate (near goal / padding).
            seg = self._waypoints[:, 1] - self._waypoints[:, 0]  # (N, 2)
            seg_ok = torch.norm(seg, dim=-1, keepdim=True) > 1e-3
            dir_vec = torch.where(seg_ok, seg, active_wp - board_xy)
            desired_heading = torch.atan2(dir_vec[:, 1], dir_vec[:, 0])
            head_err_signed = (board_yaw + math.pi) - desired_heading  # board -X vs tangent
            head_err = torch.atan2(torch.sin(head_err_signed), torch.cos(head_err_signed)).abs()
            # Alignment gate in [0,1]: 1 when -X points exactly along the tangent, 0 at
            # >=90 deg off. clamp(cos,0,1) (not 0.5*(1+cos)) so a BROADSIDE pose (90 deg)
            # is EXACTLY 0, not 0.5 -> sliding through a corner sideways earns nothing.
            align = torch.clamp(torch.cos(head_err_signed), 0.0, 1.0)

            # All continuous terms bounded; weights == relative priorities.
            # Progress is GATED by alignment: forward motion only pays when the board's
            # -X leads along the tangent. Broadside/reversed forward motion pays ~0, so
            # the ONLY way to keep collecting progress through a corner is to rotate the
            # board into alignment. Receding is NOT gated (full penalty regardless of
            # pose) so the policy can't dodge it by turning sideways.
            progress_gate = torch.where(progress > 0, align, torch.ones_like(align))
            r_progress = self.cfg.nrew_w_progress * progress * progress_gate
            r_wp = self.cfg.nrew_w_wp * reached.float()
            r_tilt = self.cfg.nrew_w_tilt * torch.exp(-((tilt / self.cfg.nrew_sigma_tilt) ** 2))
            r_height = self.cfg.nrew_w_height * torch.exp(-((height_loss / self.cfg.nrew_sigma_height) ** 2))
            # Average of PER-ROBOT Gaussians (not a Gaussian of the averaged error):
            # both robots must align to score high — a single misaligned robot cannot
            # be masked by a well-aligned partner.
            r_facing = self.cfg.nrew_w_facing * torch.exp(
                -((per_robot_facing_err / self.cfg.nrew_sigma_facing) ** 2)
            ).mean(dim=0)
            # Average of per-robot anchor Gaussians (both robots must stay tethered).
            r_offset = self.cfg.nrew_w_offset * torch.exp(
                -((per_robot_offset_err / self.cfg.nrew_sigma_offset) ** 2)
            ).mean(dim=0)
            r_alive = torch.full((self.num_envs,), self.cfg.nrew_w_alive, device=self.device)
            death_pen = board_dying * self.cfg.nrew_w_death
            # Terminal collision -> big one-time cliff; non-terminal -> small per-step
            # contact penalty (episode continues, so it must be recoverable in scale).
            _w_col = (self.cfg.nrew_w_collision if self.cfg.obstacle_collision_terminates
                      else self.cfg.nrew_w_collision_contact)
            collision_pen = collided * _w_col

            # Dense obstacle-proximity penalty (tier-2 perception-driven avoidance).
            # d = nearest wall distance SENSED by robot_0/1 LiDAR (max normalized ray
            # -> min distance). penalty = w * clamp(1 - d/d_pen, 0, 1)^2: quadratic
            # inside d_pen, exactly 0 beyond it. Recomputed here (cheap) so reward and
            # obs use the same rays.
            if self._obs_aabb.shape[0] > 0 and self.cfg.nrew_w_obstacle > 0.0:
                origins2 = self.scene.env_origins[:, :2]
                psi_max = torch.zeros(self.num_envs, device=self.device)
                for aid in self.cfg.possible_agents:  # LiDAR-equipped learning carriers
                    r = self.robots[aid]
                    rxy = r.data.root_pos_w[:, :2] - origins2
                    ryaw = _get_heading_yaw(r.data.root_quat_w)
                    psi = self._compute_lidar_normalized(rxy, ryaw)  # (N, R)
                    psi_max = torch.maximum(psi_max, psi.max(dim=1).values)
                d_min = self.cfg.lidar_max_range * (1.0 - psi_max)  # (N,) nearest wall (m)
                proximity = torch.clamp(1.0 - d_min / self.cfg.obstacle_penalty_dist, min=0.0)
                obstacle_pen = self.cfg.nrew_w_obstacle * proximity ** 2
            else:
                obstacle_pen = torch.zeros(self.num_envs, device=self.device)

            # One-time goal bonus: the board just consumed the final fixed-path
            # waypoint this step (idx reached the P sentinel for the first time).
            # _goal_reached is a persistent latch so the bonus fires exactly once
            # and _get_dones can terminate the episode as a success next step.
            if self.cfg.use_fixed_path and self._path.shape[0] > 0:
                at_goal = self._path_idx >= self._path.shape[0]
                newly_at_goal = at_goal & (~self._goal_reached)
                self._goal_reached |= at_goal
            else:
                newly_at_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            goal_bonus = newly_at_goal.float() * self.cfg.nrew_w_goal

            # Board-heading reward: reuse the alignment gate computed above. clamp(cos,0,1)
            # gives a NON-ZERO gradient (sin) across [0, 90deg] that pulls the board -X
            # toward the tangent, and is EXACTLY 0 at >=90deg so a broadside/reversed pose
            # earns nothing (removes the "sideways through the corner still gets 0.5"
            # loophole of the old 0.5*(1+cos) shaping).
            r_heading = self.cfg.nrew_w_heading * align

            reward = (
                r_progress + r_wp + r_tilt + r_height + r_facing + r_offset + r_alive
                + r_heading - death_pen - collision_pen - obstacle_pen + goal_bonus
            )

            self.extras["log"] = {
                "reward_total": reward.mean(),
                "rew_progress": r_progress.mean(),
                "reward_wp_bonus": r_wp.mean(),
                "rew_tilt": r_tilt.mean(),
                "rew_height": r_height.mean(),
                "rew_facing": r_facing.mean(),
                "rew_offset": r_offset.mean(),
                "rew_heading": r_heading.mean(),
                "heading_err": head_err.mean(),
                "reward_alive": r_alive.mean(),
                "reward_goal": goal_bonus.mean(),
                "penalty_death": (-death_pen).mean(),
                "penalty_collision": (-collision_pen).mean(),
                "penalty_obstacle": (-obstacle_pen).mean(),
                "board_height": board_z.mean(),
                "dist_to_waypoint": dist_to_wp.mean(),
                "offset_error": offset_error.mean(),
                "tilt": tilt.mean(),
                "wp_reached_count": self._wp_reached_count.float().mean(),
                "curr_target_radius": torch.tensor(self._curr_target_radius, device=self.device),
            }
        else:
            # --- legacy (unnormalized) reward, kept for A/B comparison ---
            dist_penalty = self.cfg.rew_dist_penalty_scale * dist_to_wp
            wp_bonus = reached.float() * self.cfg.rew_wp_reached_bonus
            alive = torch.ones(self.num_envs, device=self.device) * self.cfg.rew_alive
            death_penalty = board_dying * self.cfg.rew_death_penalty
            facing_penalty = self.cfg.rew_facing_penalty_scale * facing_error

            collision_pen = collided * self.cfg.nrew_w_collision
            reward = (
                - dist_penalty
                + wp_bonus
                - self.cfg.rew_tilt_penalty_scale * tilt
                - self.cfg.rew_height_penalty_scale * height_loss
                + death_penalty
                + alive
                - facing_penalty
                - collision_pen
            )

            self.extras["log"] = {
                "reward_total": reward.mean(),
                "penalty_dist": dist_penalty.mean(),
                "reward_wp_bonus": wp_bonus.mean(),
                "penalty_tilt": (self.cfg.rew_tilt_penalty_scale * tilt).mean(),
                "penalty_height": (self.cfg.rew_height_penalty_scale * height_loss).mean(),
                "penalty_death": death_penalty.mean(),
                "penalty_collision": (-collision_pen).mean(),
                "penalty_facing": facing_penalty.mean(),
                "reward_alive": alive.mean(),
                "board_height": board_z.mean(),
                "dist_to_waypoint": dist_to_wp.mean(),
                "offset_error": offset_error.mean(),
                "tilt": tilt.mean(),
                "wp_reached_count": self._wp_reached_count.float().mean(),
                "curr_target_radius": torch.tensor(self._curr_target_radius, device=self.device),
            }

        return {aid: reward for aid in self.cfg.possible_agents}
    def _get_dones(self) -> tuple[dict, dict]:
        """Check termination: any robot falls OR board drops."""
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Check ALL robots for falls (including followers)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        for agent_id in ALL_ROBOT_IDS:
            robot = self.robots[agent_id]
            height = robot.data.root_pos_w[:, 2]
            fallen = height < self.cfg.termination_height

            gravity_b = quat_rotate_inverse(robot.data.root_quat_w, gravity_w)
            tilted = (gravity_b[:, 0].abs() > 0.7) | (gravity_b[:, 1].abs() > 0.7)

            terminated = terminated | fallen | tilted

        # Board dropped
        board_dropped = self._board.data.root_pos_w[:, 2] < self.cfg.board_drop_height
        terminated = terminated | board_dropped

        # Wall collision: terminal only when obstacle_collision_terminates is True.
        # Otherwise the episode continues and the collision is penalized per-step in
        # _get_rewards (walls are solid so the formation still can't pass through).
        if self.cfg.obstacle_collision_terminates:
            terminated = terminated | self._compute_obstacle_collision()

        # Goal reached (fixed-path / maze): the board consumed the final waypoint.
        # Terminal SUCCESS (the goal bonus is granted in _get_rewards). The latch
        # is set during the previous step's reward pass, so this fires one step
        # later -- harmless. Lets the episode end on completion instead of waiting
        # out the (deliberately generous) time budget.
        if self.cfg.terminate_on_goal and self.cfg.use_fixed_path:
            terminated = terminated | self._goal_reached

        timed_out = self.episode_length_buf >= self.max_episode_length

        # Only return for the learning carriers
        term_dict = {aid: terminated for aid in self.cfg.possible_agents}
        timeout_dict = {aid: timed_out for aid in self.cfg.possible_agents}

        return term_dict, timeout_dict

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments and generate new waypoints."""
        super()._reset_idx(env_ids)
        num_reset = len(env_ids)

        # Reset ALL robots (including followers)
        for agent_id in ALL_ROBOT_IDS:
            robot = self.robots[agent_id]
            default_root_state = robot.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] += self.scene.env_origins[env_ids]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = torch.zeros_like(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            # Reset per-robot buffers
            self._ll_actions[agent_id][env_ids] = 0.0
            self._filtered_actions[agent_id][env_ids] = 0.0
            self._gait_phase[agent_id][env_ids] = (
                torch.rand(num_reset, device=self.device) * 2.0 * math.pi
            )
            self._target_cmd[agent_id][env_ids] = 0.0
            self._target_pos_world[agent_id][env_ids] = self.robots[agent_id].data.root_pos_w[env_ids, :2]
            self._target_yaw_world[agent_id][env_ids] = _get_heading_yaw(
                self.robots[agent_id].data.root_quat_w[env_ids]
            )
            # Init interpolation endpoints to the robot's pose (no target motion until
            # the first HL command arrives).
            self._target_start_pos[agent_id][env_ids] = self._target_pos_world[agent_id][env_ids]
            self._target_final_pos[agent_id][env_ids] = self._target_pos_world[agent_id][env_ids]
            self._target_start_yaw[agent_id][env_ids] = self._target_yaw_world[agent_id][env_ids]
            self._target_final_yaw[agent_id][env_ids] = self._target_yaw_world[agent_id][env_ids]

        # Reset obs history for the learning carriers only
        for agent_id in self.cfg.possible_agents:
            self._obs_history[agent_id][env_ids] = 0.0

        # Reset board
        board_default = self._board.data.default_root_state[env_ids].clone()
        board_default[:, :3] += self.scene.env_origins[env_ids]
        self._board.write_root_pose_to_sim(board_default[:, :7], env_ids)
        self._board.write_root_velocity_to_sim(board_default[:, 7:], env_ids)

        # --- curriculum: collect finished-episode wp_reached BEFORE it is zeroed ---
        if (
            self.cfg.use_target_radius_curriculum
            and not self.cfg.use_fixed_path
            and self._curr_target_radius < self.cfg.target_radius_end - 1e-6
        ):
            self._curr_wp_history.extend(
                self._wp_reached_count[env_ids].float().tolist()
            )
            if len(self._curr_wp_history) >= self.cfg.curriculum_window:
                mean_wp = sum(self._curr_wp_history) / len(self._curr_wp_history)
                self._curr_wp_history = []
                self._level_wp_means.append(mean_wp)

                # Trend-based plateau: compare the recent `patience` windows against the
                # `patience` windows before them. Only advance if the recent half is NOT
                # meaningfully higher (no upward trend) AND we have dwelled past the floor.
                p = self.cfg.curriculum_patience
                dwell_ok = self._level_step_counter >= self.cfg.curriculum_min_steps_per_level
                if dwell_ok and len(self._level_wp_means) >= 2 * p:
                    recent = sum(self._level_wp_means[-p:]) / p
                    prev = sum(self._level_wp_means[-2 * p:-p]) / p
                    still_improving = recent > prev + self.cfg.curriculum_improve_eps
                    if not still_improving:
                        self._curr_target_radius = min(
                            self._curr_target_radius + self.cfg.target_radius_curriculum_step,
                            self.cfg.target_radius_end,
                        )
                        print(
                            f"[curriculum] wp trend flat (recent {recent:.2f} vs prev "
                            f"{prev:.2f}) after {self._level_step_counter} steps -> "
                            f"target_radius = {self._curr_target_radius:.2f}"
                        )
                        self._level_step_counter = 0
                        self._level_wp_means = []

        # Generate new waypoints
        board_xy = board_default[:, :2]
        self._generate_waypoints(env_ids, board_xy)

        # Seed progress-shaping baseline with the spawn distance to the active waypoint
        # (board_xy here is in world frame including env origin, matching _waypoints).
        active_wp_reset = self._waypoints[env_ids, 0]
        self._prev_dist_to_wp[env_ids] = torch.norm(active_wp_reset - board_xy, dim=-1)

    def _compute_lidar_normalized(self, origin_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Synthetic 360-deg LiDAR vs static obstacle AABBs, normalized per paper Eq. 5.

        Args:
            origin_xy: (N, 2) ray origin in ENV-LOCAL frame (same frame as obstacles).
            yaw:       (N,) robot world yaw (= env-local yaw; origins are translations).

        Returns:
            (N, R) where each entry = 1 - min(d, dmax)/dmax. 0 means far/no hit
            within range; ->1 means a wall right at the robot.
        """
        N = origin_xy.shape[0]
        R = self.cfg.lidar_num_rays
        dmax = self.cfg.lidar_max_range
        M = self._obs_aabb.shape[0]
        if M == 0:
            return torch.zeros(N, R, device=self.device)

        # Ray directions in world/env frame: (N, R)
        ang = yaw.unsqueeze(1) + self._lidar_ray_offsets.unsqueeze(0)  # (N, R)
        dx = torch.cos(ang)
        dy = torch.sin(ang)
        # Avoid division by zero for axis-aligned rays (slab method).
        eps = 1e-6
        dx = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
        dy = torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)

        # Broadcast to (N, R, M)
        ox = origin_xy[:, 0].view(N, 1, 1)
        oy = origin_xy[:, 1].view(N, 1, 1)
        inv_dx = (1.0 / dx).unsqueeze(-1)  # (N, R, 1)
        inv_dy = (1.0 / dy).unsqueeze(-1)

        aabb = self._obs_aabb.view(1, 1, M, 4)  # [x0, x1, y0, y1]
        x0, x1, y0, y1 = aabb[..., 0], aabb[..., 1], aabb[..., 2], aabb[..., 3]

        t1x = (x0 - ox) * inv_dx
        t2x = (x1 - ox) * inv_dx
        t1y = (y0 - oy) * inv_dy
        t2y = (y1 - oy) * inv_dy

        tmin_x = torch.minimum(t1x, t2x)
        tmax_x = torch.maximum(t1x, t2x)
        tmin_y = torch.minimum(t1y, t2y)
        tmax_y = torch.maximum(t1y, t2y)

        t_enter = torch.maximum(tmin_x, tmin_y)  # (N, R, M)
        t_exit = torch.minimum(tmax_x, tmax_y)

        # Valid hit: box is in front (t_exit >= max(t_enter, 0)) and within range.
        hit = (t_exit >= torch.clamp(t_enter, min=0.0)) & (t_enter <= dmax)
        # Distance along the ray to first contact; 0 if origin already inside box.
        dist = torch.clamp(t_enter, min=0.0)
        dist = torch.where(hit, dist, torch.full_like(dist, dmax))
        d_min = dist.min(dim=-1).values  # (N, R) nearest obstacle per ray

        return 1.0 - torch.clamp(d_min, max=dmax) / dmax

    def _compute_obstacle_collision(self) -> torch.Tensor:
        """Per-env flag: does any REAL solid body overlap an obstacle AABB?

        Earlier this used one monolithic 2.7x1.35 formation box (matching
        check_user_path.py's planning footprint), but that treats the empty gaps
        between the board and the three robots -- and the air beyond the board's
        real 2.0x1.0 extent -- as solid, so a slight board rotation/drift made the
        oversized box's corner clip a wall while the actual board+robots still had
        clearance (visible in play). We now test the ACTUAL geometry instead:

          * the board's real OBB (BOARD_LENGTH x BOARD_WIDTH), oriented by board
            yaw, via SAT against each AABB;
          * each of the 3 robot bases as a circle of radius `robot_collision_radius`
            (real body radius), via point-to-AABB distance.

        This is strictly less conservative than the 2.7x1.35 planning footprint, so
        the check_user_path-validated path never false-terminates. Returns a (N,)
        bool; always False when obstacles or the safety net are disabled.
        """
        M = self._obs_aabb.shape[0]
        if M == 0 or not self.cfg.use_obstacle_collision_termination:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        origins = self.scene.env_origins[:, :2]

        # Obstacle AABB edges / centers / half-extents (env-local).
        aabb = self._obs_aabb  # (M, 4): x0, x1, y0, y1
        x0 = aabb[:, 0].view(1, M)
        x1 = aabb[:, 1].view(1, M)
        y0 = aabb[:, 2].view(1, M)
        y1 = aabb[:, 3].view(1, M)
        mx = ((aabb[:, 0] + aabb[:, 1]) * 0.5).view(1, M)
        my = ((aabb[:, 2] + aabb[:, 3]) * 0.5).view(1, M)
        hx = ((aabb[:, 1] - aabb[:, 0]) * 0.5).view(1, M)
        hy = ((aabb[:, 3] - aabb[:, 2]) * 0.5).view(1, M)

        hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- 1) Board OBB (real 2.0 x 1.0) vs each AABB via SAT ---
        c = self._board.data.root_pos_w[:, :2] - origins  # (N, 2)
        yaw = _get_heading_yaw(self._board.data.root_quat_w)  # (N,)
        co = torch.cos(yaw).unsqueeze(1)  # (N, 1)
        si = torch.sin(yaw).unsqueeze(1)
        aco = co.abs()
        asi = si.abs()
        m = self.cfg.obstacle_collision_margin
        eu = BOARD_LENGTH / 2.0 + m  # half-extent along board-X
        ev = BOARD_WIDTH / 2.0 + m   # half-extent along board-Y
        dx = c[:, 0:1] - mx  # (N, M)
        dy = c[:, 1:2] - my
        ax_wx = dx.abs() <= hx + (eu * aco + ev * asi)
        ax_wy = dy.abs() <= hy + (eu * asi + ev * aco)
        du = (dx * co + dy * si).abs()
        ax_u = du <= eu + (hx * aco + hy * asi)
        dv = (-dx * si + dy * co).abs()
        ax_v = dv <= ev + (hx * asi + hy * aco)
        hit = hit | (ax_wx & ax_wy & ax_u & ax_v).any(dim=1)

        # --- 2) Each robot base as a circle vs each AABB (point-to-AABB dist) ---
        r = self.cfg.robot_collision_radius
        for aid in ALL_ROBOT_IDS:
            p = self.robots[aid].data.root_pos_w[:, :2] - origins  # (N, 2)
            px = p[:, 0:1]  # (N, 1)
            py = p[:, 1:2]
            cxp = torch.clamp(px, x0, x1)  # closest point on AABB (N, M)
            cyp = torch.clamp(py, y0, y1)
            dist2 = (px - cxp) ** 2 + (py - cyp) ** 2
            hit = hit | (dist2 <= r * r).any(dim=1)

        return hit

    def _refresh_path_window(self, env_ids: torch.Tensor):
        """Fill the 5-wide sliding window from the fixed path at the current index.

        World waypoint = env-local path point + env origin. Indices past the goal
        are clamped to the last point so the window pads with the goal.
        """
        P = self._path.shape[0]
        origins = self.scene.env_origins[env_ids, :2]  # (n, 2)
        base = self._path_idx[env_ids]  # (n,)
        for k in range(NUM_WAYPOINTS):
            idx = torch.clamp(base + k, max=P - 1)  # (n,)
            self._waypoints[env_ids, k] = self._path[idx] + origins

    def _generate_waypoints(self, env_ids: torch.Tensor, board_xy: torch.Tensor):
        """Generate the initial 5-waypoint sliding window.

        Fixed-path mode: window is taken from the predefined course path starting
        at the first target (index 1). Open-field mode: random chained walk at
        `target_radius` spacing.
        """
        if self.cfg.use_fixed_path and self._path.shape[0] > 0:
            self._path_idx[env_ids] = 1
            self._goal_reached[env_ids] = False
            self._refresh_path_window(env_ids)
            self._wp_reached_count[env_ids] = 0
            return

        num = len(env_ids)
        radius = self._curr_target_radius

        prev_x = board_xy[:, 0]
        prev_y = board_xy[:, 1]

        for k in range(NUM_WAYPOINTS):
            theta = torch.rand(num, device=self.device) * 2.0 * math.pi
            wp_x = prev_x + radius * torch.cos(theta)
            wp_y = prev_y + radius * torch.sin(theta)

            self._waypoints[env_ids, k, 0] = wp_x
            self._waypoints[env_ids, k, 1] = wp_y

            prev_x = wp_x
            prev_y = wp_y

        self._wp_reached_count[env_ids] = 0

    def _advance_waypoints(self, env_ids: torch.Tensor):
        """Advance the sliding window when the board reaches waypoints[:, 0].

        Fixed-path mode: bump the path index by one (clamped to a sentinel P that
        marks "goal reached", which gates off further advancement in the reward).
        Open-field mode: shift the window and append a new random chained point.
        """
        if self.cfg.use_fixed_path and self._path.shape[0] > 0:
            P = self._path.shape[0]
            self._path_idx[env_ids] = torch.clamp(self._path_idx[env_ids] + 1, max=P)
            self._refresh_path_window(env_ids)
            return

        num = len(env_ids)
        radius = self._curr_target_radius

        # Shift waypoints forward: [1,2,3,4] -> [0,1,2,3]
        self._waypoints[env_ids, :NUM_WAYPOINTS - 1] = self._waypoints[env_ids, 1:].clone()

        # Generate new waypoint at the end, chained from the current last point
        last_wp = self._waypoints[env_ids, NUM_WAYPOINTS - 2]  # the new second-to-last
        theta = torch.rand(num, device=self.device) * 2.0 * math.pi
        new_wp_x = last_wp[:, 0] + radius * torch.cos(theta)
        new_wp_y = last_wp[:, 1] + radius * torch.sin(theta)
        self._waypoints[env_ids, NUM_WAYPOINTS - 1, 0] = new_wp_x
        self._waypoints[env_ids, NUM_WAYPOINTS - 1, 1] = new_wp_y


        # ------------------------------------------------------------------
        # Low-level helpers
        # ------------------------------------------------------------------

    def _build_ll_obs(self, agent_id: str) -> torch.Tensor:
        """Build 83-dim observation for the waist-tracking JIT policy.

        For the learning carriers: recomputes live error from world-space target.
        For follower agents (robot_2): computes target from board's current pose (50Hz).
        """
        robot = self.robots[agent_id]
        dof_ids = self._dof_ids[agent_id]
        root_quat = robot.data.root_quat_w
        root_ang_vel = robot.data.root_ang_vel_b

        dof_pos = robot.data.joint_pos[:, dof_ids] - self._default_dof_pos
        dof_vel = robot.data.joint_vel[:, dof_ids]

        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        gravity_b = quat_rotate_inverse(root_quat, gravity_w)

        gait_sin = torch.sin(self._gait_phase[agent_id]).unsqueeze(-1)
        gait_cos = torch.cos(self._gait_phase[agent_id]).unsqueeze(-1)
        gait_clock = torch.cat([gait_sin, gait_cos], dim=-1)

        # Compute live position error in body frame
        current_pos = robot.data.root_pos_w[:, :2]
        current_yaw = _get_heading_yaw(root_quat)

        if agent_id in FOLLOWER_ROBOT_IDS:
            # Follower: target derived from board pose with directional lookahead
            # Only apply lookahead when board approaches robot_2 (prevents collision)
            # No lookahead when board moves away (prevents overshooting into board)
            board_xy = self._board.data.root_pos_w[:, :2]
            board_vel = self._board.data.root_lin_vel_w[:, :2]  # (N, 2)
            board_yaw = _get_heading_yaw(self._board.data.root_quat_w)
            half_w = BOARD_WIDTH / 2.0 + self.cfg.robot_offset

            # Outward normal from board center toward robot_2's side (+Y in board frame)
            outward_x = -torch.sin(board_yaw)
            outward_y = torch.cos(board_yaw)

            # Project board velocity onto outward direction
            vel_outward = board_vel[:, 0] * outward_x + board_vel[:, 1] * outward_y

            # Only lookahead when board approaches robot_2 (vel_outward > 0)
            lookahead_t = 0.5
            lookahead_dist = torch.clamp(vel_outward * lookahead_t, min=0.0)

            # Target = board position + (half_w + lookahead) along outward direction
            target_x = board_xy[:, 0] + (half_w + lookahead_dist) * outward_x
            target_y = board_xy[:, 1] + (half_w + lookahead_dist) * outward_y
            target_yaw_val = board_yaw - math.pi / 2  # facing -Y in board frame
        else:
            # learning carrier: use world-space target set by _pre_physics_step
            target_x = self._target_pos_world[agent_id][:, 0]
            target_y = self._target_pos_world[agent_id][:, 1]
            target_yaw_val = self._target_yaw_world[agent_id]

        dx_world = target_x - current_pos[:, 0]
        dy_world = target_y - current_pos[:, 1]

        cos_neg_yaw = torch.cos(-current_yaw)
        sin_neg_yaw = torch.sin(-current_yaw)
        dx_body = cos_neg_yaw * dx_world - sin_neg_yaw * dy_world
        dy_body = sin_neg_yaw * dx_world + cos_neg_yaw * dy_world

        dyaw = target_yaw_val - current_yaw
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))

        target_cmd = torch.stack([dx_body, dy_body, dyaw], dim=-1)  # (N, 3)

        ang_vel_scaled = root_ang_vel * self.cfg.obs_scale_ang_vel
        dof_vel_scaled = dof_vel * self.cfg.obs_scale_dof_vel

        ll_obs = torch.cat([
            self._ll_actions[agent_id],  # actions (23)
            ang_vel_scaled,              # base_ang_vel (3)
            dof_pos,                     # dof_pos (23)
            dof_vel_scaled,              # dof_vel (23)
            gait_clock,                  # gait_clock (2)
            gravity_b,                   # projected_gravity (3)
            target_cmd,                  # waist_pos_error (3)
            target_cmd,                  # waist_target_cmd (3)
        ], dim=-1)

        return ll_obs.clamp(-100.0, 100.0)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """Convert yaw angle to quaternion (w, x, y, z)."""
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (w, 0.0, 0.0, z)


def _get_heading_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion (w, x, y, z) format. Returns (N,)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_vec_by_quat(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate vector by quaternion. vec: (N, 3), quat: (N, 4) w,x,y,z."""
    q_w = quat[:, 0:1]
    q_vec = quat[:, 1:4]
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec, t, dim=-1)


def _transform_points(
    points: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor
) -> torch.Tensor:
    """Transform local points to world frame.

    Args:
        points: (P, 3) local points
        pos: (N, 3) position
        quat: (N, 4) quaternion (w, x, y, z)

    Returns:
        (N, P, 3) world-frame points
    """
    N = pos.shape[0]
    P = points.shape[0]
    # Expand points: (1, P, 3) -> (N, P, 3)
    pts = points.unsqueeze(0).expand(N, P, 3).reshape(N * P, 3)
    # Expand quat: (N, 4) -> (N*P, 4)
    q = quat.unsqueeze(1).expand(N, P, 4).reshape(N * P, 4)
    # Rotate
    rotated = _rotate_vec_by_quat(pts, q)
    rotated = rotated.reshape(N, P, 3)
    # Translate
    return rotated + pos.unsqueeze(1)
