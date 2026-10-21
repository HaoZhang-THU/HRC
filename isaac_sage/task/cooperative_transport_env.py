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

G1_23DOF_DEFAULT_POS_CARRY = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
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
ALL_ROBOT_IDS = ["robot_0", "robot_1", "robot_2"]
PAYLOAD_TRACKING_ROBOT_IDS = ["robot_2"]
NUM_WAYPOINTS = 5
OBS_CORE_PER_FRAME = 49
OBS_LIDAR = 36
OBS_CURRENT_FRAME = OBS_CORE_PER_FRAME + OBS_LIDAR
OBS_TEMPORAL_FRAMES = 2
OBS_TOTAL = OBS_CURRENT_FRAME + OBS_CORE_PER_FRAME * OBS_TEMPORAL_FRAMES
WP_REACH_THRESHOLD = 0.15

BOARD_LENGTH = 2.0
BOARD_WIDTH = 1.0
BOARD_THICKNESS = 0.10
BOARD_MASS = 6.0


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


@configclass
class G1CooperativeTransportEnvCfg(DirectMARLEnvCfg):
    possible_agents: list = ["robot_0", "robot_1"]

    episode_length_s: float = 30.0
    decimation: int = 200

    action_spaces: dict = {"robot_0": 3, "robot_1": 3}
    observation_spaces: dict = {"robot_0": OBS_TOTAL, "robot_1": OBS_TOTAL}
    state_space: int = 0
    state_spaces: dict = {"robot_0": 0, "robot_1": 0}

    low_level_policy_path: str = LOW_LEVEL_POLICY_PATH
    low_level_decimation: int = 4
    action_scale: float = 0.25
    ll_obs_dim: int = 83
    ll_action_dim: int = 23

    obs_scale_ang_vel: float = 0.25
    obs_scale_dof_vel: float = 0.05

    gait_frequency: float = 0.77

    action_filter_alpha: float = 0.3

    max_target_dist: float = 0.2
    max_target_yaw: float = 0.3

    interpolate_target: bool = True
    ll_steps_per_hl: int = 50

    rew_dist_penalty_scale: float = 3.0
    rew_tilt_penalty_scale: float = 2.0
    rew_height_penalty_scale: float = 1.0
    rew_alive: float = 5.5
    rew_wp_reached_bonus: float = 10.0
    rew_death_penalty: float = -50.0
    rew_facing_penalty_scale: float = 2.0

    use_normalized_reward: bool = True
    nrew_progress_speed: float = 0.2
    nrew_w_progress: float = 2.5
    nrew_w_wp: float = 5.0
    nrew_w_tilt: float = 0.6
    nrew_w_height: float = 0.4
    nrew_w_facing: float = 0.4
    nrew_w_alive: float = 0.05
    nrew_w_death: float = 50.0
    nrew_sigma_tilt: float = 0.20
    nrew_sigma_height: float = 0.08
    nrew_sigma_facing: float = 0.5
    nrew_w_offset: float = 0.6
    nrew_sigma_offset: float = 0.15
    nrew_w_heading: float = 0.0
    nrew_sigma_heading: float = 0.25
    nrew_w_obstacle: float = 0.3
    obstacle_penalty_dist: float = 1.0

    termination_height: float = 0.5
    board_drop_height: float = 0.70

    use_obstacle_collision_termination: bool = True
    formation_length: float = 2.7
    formation_width: float = 1.35
    obstacle_collision_margin: float = 0.0
    robot_collision_radius: float = 0.2
    nrew_w_collision: float = 50.0
    obstacle_collision_terminates: bool = False
    nrew_w_collision_contact: float = 3.0

    terminate_on_goal: bool = True
    nrew_w_goal: float = 20.0

    board_friction: float = 5.0

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

    start_xy: tuple = (2.0, -2.32)

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

    lidar_num_rays: int = 36
    lidar_max_range: float = 5.0
    mask_lidar_obs: bool = False

    target_radius: float = 1.0

    use_target_radius_curriculum: bool = True
    target_radius_start: float = 0.3
    target_radius_end: float = 1.0
    target_radius_curriculum_step: float = 0.1
    curriculum_window: int = 300
    curriculum_patience: int = 5
    curriculum_improve_eps: float = 0.15
    curriculum_min_steps_per_level: int = 50000

    training_g1_usd_path: str = TRAINING_G1_USD_PATH

    robot_offset: float = 0.20

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


class G1CooperativeTransportEnv(DirectMARLEnv):
    cfg: G1CooperativeTransportEnvCfg

    def __init__(self, cfg: G1CooperativeTransportEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        policy_path = cfg.low_level_policy_path
        if not policy_path or not os.path.exists(policy_path):
            raise FileNotFoundError(f"Low-level policy not found: '{policy_path}'.")
        self._ll_policy = torch.jit.load(policy_path, map_location=self.device).eval()

        self._dof_ids = {}
        for agent_id in ALL_ROBOT_IDS:
            ids, _ = self.robots[agent_id].find_joints(G1_23DOF_JOINT_NAMES, preserve_order=True)
            self._dof_ids[agent_id] = torch.tensor(ids, device=self.device, dtype=torch.long)

        self._default_dof_pos = torch.tensor(
            G1_23DOF_DEFAULT_POS_CARRY, device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        self._kp, self._kd = _build_kp_kd_tensors(self.device)

        self._torque_limits = torch.tensor(G1_23DOF_EFFORT, device=self.device, dtype=torch.float32)

        self._locked_mask = torch.zeros(cfg.ll_action_dim, device=self.device, dtype=torch.bool)
        self._locked_mask[LOCKED_DOF_INDICES] = True

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
        self._hl_ll_count = 0
        self._gait_phase = {
            aid: torch.zeros(self.num_envs, device=self.device)
            for aid in ALL_ROBOT_IDS
        }

        self._ll_step_counter = 0

        self._waypoints = torch.zeros(self.num_envs, NUM_WAYPOINTS, 2, device=self.device)
        self._prev_dist_to_wp = torch.zeros(self.num_envs, device=self.device)
        self._wp_reached_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        if self.cfg.use_fixed_path and len(self.cfg.path_waypoints) > 0:
            self._path = torch.tensor(
                self.cfg.path_waypoints, dtype=torch.float32, device=self.device
            )
        else:
            self._path = torch.zeros(0, 2, device=self.device)
        self._path_idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if len(self.cfg.obstacle_specs) > 0:
            aabb = []
            for size, pos in self.cfg.obstacle_specs:
                aabb.append([
                    pos[0] - size[0] / 2.0, pos[0] + size[0] / 2.0,
                    pos[1] - size[1] / 2.0, pos[1] + size[1] / 2.0,
                ])
            self._obs_aabb = torch.tensor(aabb, dtype=torch.float32, device=self.device)
        else:
            self._obs_aabb = torch.zeros(0, 4, device=self.device)

        self._lidar_ray_offsets = torch.linspace(
            0.0, 2.0 * math.pi, self.cfg.lidar_num_rays + 1, device=self.device
        )[:-1]

        self._curr_target_radius = (
            self.cfg.target_radius_start
            if self.cfg.use_target_radius_curriculum
            else self.cfg.target_radius
        )
        self._curr_wp_history: list[float] = []
        self._level_step_counter = 0
        self._level_wp_means: list[float] = []

        self._obs_history = {
            aid: torch.zeros(self.num_envs, OBS_TEMPORAL_FRAMES + 1, OBS_CORE_PER_FRAME, device=self.device)
            for aid in self.cfg.possible_agents
        }

    def _setup_scene(self):
        cfg = self.cfg

        half_w = BOARD_WIDTH / 2.0
        half_l = BOARD_LENGTH / 2.0
        offset = cfg.robot_offset
        board_z = 0.97

        sx, sy = cfg.start_xy

        r0_pos = (sx + half_l + offset, sy + 0.0, 0.8)
        r0_rot = _yaw_to_quat(math.pi)

        r1_pos = (sx - (half_l + offset), sy + 0.0, 0.8)
        r1_rot = _yaw_to_quat(0.0)

        r2_pos = (sx + 0.0, sy + half_w + offset, 0.8)
        r2_rot = _yaw_to_quat(-math.pi / 2)

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

        for i, (size, pos) in enumerate(cfg.obstacle_specs):
            obs_spawn = sim_utils.CuboidCfg(
                size=tuple(size),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            )
            obs_spawn.func(f"/World/envs/env_.*/Obstacle_{i}", obs_spawn, translation=tuple(pos))

        cfg.terrain.num_envs = self.scene.cfg.num_envs
        cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = cfg.terrain.class_type(cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        for agent_id in ALL_ROBOT_IDS:
            self.scene.articulations[agent_id] = self.robots[agent_id]
        self.scene.rigid_objects["board"] = self._board

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: dict):
        for agent_id in self.cfg.possible_agents:
            act = actions[agent_id]
            dx_body = act[:, 0].clamp(-self.cfg.max_target_dist, self.cfg.max_target_dist)
            dy_body = act[:, 1].clamp(-self.cfg.max_target_dist, self.cfg.max_target_dist)
            dyaw = act[:, 2].clamp(-self.cfg.max_target_yaw, self.cfg.max_target_yaw)

            robot = self.robots[agent_id]
            current_pos = robot.data.root_pos_w[:, :2]
            current_yaw = _get_heading_yaw(robot.data.root_quat_w)

            cos_yaw = torch.cos(current_yaw)
            sin_yaw = torch.sin(current_yaw)
            dx_world = cos_yaw * dx_body - sin_yaw * dy_body
            dy_world = sin_yaw * dx_body + cos_yaw * dy_body

            final_x = current_pos[:, 0] + dx_world
            final_y = current_pos[:, 1] + dy_world
            final_yaw = current_yaw + dyaw

            if self.cfg.interpolate_target:
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

            actions_scaled = self._filtered_actions[agent_id] * self.cfg.action_scale
            current_pos = robot.data.joint_pos[:, dof_ids]
            current_vel = robot.data.joint_vel[:, dof_ids]
            target_pos = actions_scaled + self._default_dof_pos
            torques_23 = self._kp * (target_pos - current_pos) - self._kd * current_vel
            torques_23 = torques_23.clamp(-self._torque_limits, self._torque_limits)

            full_efforts = torch.zeros_like(robot.data.joint_pos)
            full_efforts[:, dof_ids] = torques_23
            robot.set_joint_effort_target(full_efforts)

        dt = self.cfg.sim.dt
        for agent_id in ALL_ROBOT_IDS:
            self._gait_phase[agent_id] += 2.0 * math.pi * self.cfg.gait_frequency * dt
            self._gait_phase[agent_id] = self._gait_phase[agent_id] % (2.0 * math.pi)

        self._ll_step_counter += 1

    def _get_observations(self) -> dict:
        obs_dict = {}

        board_pos = self._board.data.root_pos_w
        board_quat = self._board.data.root_quat_w
        board_vel = self._board.data.root_lin_vel_w
        board_xy = board_pos[:, :2]

        corners_local = torch.tensor([
            [BOARD_LENGTH / 2, BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [BOARD_LENGTH / 2, -BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [-BOARD_LENGTH / 2, BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [-BOARD_LENGTH / 2, -BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
        ], device=self.device, dtype=torch.float32)

        corners_world = _transform_points(corners_local, board_pos, board_quat)
        board_yaw = _get_heading_yaw(board_quat)

        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)

        for i, agent_id in enumerate(self.cfg.possible_agents):
            robot = self.robots[agent_id]
            root_pos = robot.data.root_pos_w
            root_quat = robot.data.root_quat_w
            root_lin_vel = robot.data.root_lin_vel_b
            robot_xy = root_pos[:, :2]
            robot_yaw = _get_heading_yaw(root_quat)
            cos_yaw = torch.cos(robot_yaw)
            sin_yaw = torch.sin(robot_yaw)

            wp_rel_world = self._waypoints - robot_xy.unsqueeze(1)
            wp_rel_body_x = cos_yaw.unsqueeze(1) * wp_rel_world[:, :, 0] + sin_yaw.unsqueeze(1) * wp_rel_world[:, :, 1]
            wp_rel_body_y = -sin_yaw.unsqueeze(1) * wp_rel_world[:, :, 0] + cos_yaw.unsqueeze(1) * wp_rel_world[:, :, 1]
            psi_task = torch.stack([wp_rel_body_x, wp_rel_body_y], dim=-1).reshape(self.num_envs, -1)

            board_rel_world = board_xy - robot_xy
            board_rel_body_x = cos_yaw * board_rel_world[:, 0] + sin_yaw * board_rel_world[:, 1]
            board_rel_body_y = -sin_yaw * board_rel_world[:, 0] + cos_yaw * board_rel_world[:, 1]
            ego_vel_xy = root_lin_vel[:, :2]
            yaw_diff = board_yaw - robot_yaw
            yaw_diff_wrapped = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff)).unsqueeze(-1)
            com_height = root_pos[:, 2:3]
            gravity_b = quat_rotate_inverse(root_quat, gravity_w)
            torso_pitch = gravity_b[:, 0:1]

            psi_ego = torch.cat([
                board_rel_body_x.unsqueeze(-1), board_rel_body_y.unsqueeze(-1),
                ego_vel_xy, yaw_diff_wrapped, com_height, torso_pitch,
            ], dim=-1)

            partners = [aid for aid in ALL_ROBOT_IDS if aid != agent_id]
            psi_partners = []
            for partner_id in partners:
                p_robot = self.robots[partner_id]
                p_pos = p_robot.data.root_pos_w
                p_quat = p_robot.data.root_quat_w
                p_vel = p_robot.data.root_lin_vel_w

                p_rel_world = p_pos[:, :2] - robot_xy
                p_rel_body_x = cos_yaw * p_rel_world[:, 0] + sin_yaw * p_rel_world[:, 1]
                p_rel_body_y = -sin_yaw * p_rel_world[:, 0] + cos_yaw * p_rel_world[:, 1]
                p_vel_body_x = cos_yaw * p_vel[:, 0] + sin_yaw * p_vel[:, 1]
                p_vel_body_y = -sin_yaw * p_vel[:, 0] + cos_yaw * p_vel[:, 1]
                p_yaw = _get_heading_yaw(p_quat)
                p_yaw_rel = p_yaw - robot_yaw
                p_yaw_rel_wrapped = torch.atan2(torch.sin(p_yaw_rel), torch.cos(p_yaw_rel)).unsqueeze(-1)
                p_height = p_pos[:, 2:3]
                p_gravity_b = quat_rotate_inverse(p_quat, gravity_w)
                p_pitch = p_gravity_b[:, 0:1]

                psi_p = torch.cat([
                    p_rel_body_x.unsqueeze(-1), p_rel_body_y.unsqueeze(-1),
                    p_vel_body_x.unsqueeze(-1), p_vel_body_y.unsqueeze(-1),
                    p_yaw_rel_wrapped, p_height, p_pitch,
                ], dim=-1)
                psi_partners.append(psi_p)

            psi_ptn = torch.cat(psi_partners, dim=-1)

            corners_rel = corners_world - root_pos.unsqueeze(1)
            corners_body_x = cos_yaw.unsqueeze(1) * corners_rel[:, :, 0] + sin_yaw.unsqueeze(1) * corners_rel[:, :, 1]
            corners_body_y = -sin_yaw.unsqueeze(1) * corners_rel[:, :, 0] + cos_yaw.unsqueeze(1) * corners_rel[:, :, 1]
            corners_body_z = corners_rel[:, :, 2]
            corners_body = torch.stack([corners_body_x, corners_body_y, corners_body_z], dim=-1)
            board_com_rel = board_pos - root_pos
            board_com_body_x = cos_yaw * board_com_rel[:, 0] + sin_yaw * board_com_rel[:, 1]
            board_com_body_y = -sin_yaw * board_com_rel[:, 0] + cos_yaw * board_com_rel[:, 1]
            board_com_body_z = board_com_rel[:, 2]
            board_vel_body_x = cos_yaw * board_vel[:, 0] + sin_yaw * board_vel[:, 1]
            board_vel_body_y = -sin_yaw * board_vel[:, 0] + cos_yaw * board_vel[:, 1]
            board_vel_body_z = board_vel[:, 2]

            psi_obj = torch.cat([
                corners_body.reshape(self.num_envs, 12),
                board_com_body_x.unsqueeze(-1), board_com_body_y.unsqueeze(-1), board_com_body_z.unsqueeze(-1),
                board_vel_body_x.unsqueeze(-1), board_vel_body_y.unsqueeze(-1), board_vel_body_z.unsqueeze(-1),
            ], dim=-1)

            robot_xy_local = robot_xy - self.scene.env_origins[:, :2]
            psi_lidar = self._compute_lidar_normalized(robot_xy_local, robot_yaw)
            if self.cfg.mask_lidar_obs:
                psi_lidar = torch.zeros_like(psi_lidar)

            obs_core = torch.cat([
                psi_task, psi_ego, psi_ptn, psi_obj,
            ], dim=-1)

            obs_current = torch.cat([obs_core, psi_lidar], dim=-1)

            self._obs_history[agent_id] = self._obs_history[agent_id].roll(1, dims=1)
            self._obs_history[agent_id][:, 0] = obs_core

            temporal_buffer = self._obs_history[agent_id][:, 1:].reshape(self.num_envs, -1)
            obs_stacked = torch.cat([obs_current, temporal_buffer], dim=-1)

            obs_dict[agent_id] = obs_stacked

        return obs_dict

    def _get_rewards(self) -> dict:
        if self.cfg.use_target_radius_curriculum and not self.cfg.use_fixed_path:
            self._level_step_counter += self.num_envs

        board_pos = self._board.data.root_pos_w
        board_quat = self._board.data.root_quat_w
        board_xy = board_pos[:, :2]
        board_z = board_pos[:, 2]

        active_wp = self._waypoints[:, 0]
        cur_dist = torch.norm(active_wp - board_xy, dim=-1)

        hl_dt = self.cfg.sim.dt * self.cfg.decimation
        v_ref = max(self.cfg.nrew_progress_speed * hl_dt, 1e-6)
        progress = torch.clamp((self._prev_dist_to_wp - cur_dist) / v_ref, -1.0, 1.0)

        reached = cur_dist < WP_REACH_THRESHOLD
        if self.cfg.use_fixed_path and self._path.shape[0] > 0:
            reached = reached & (self._path_idx < self._path.shape[0])
        if reached.any():
            reached_ids = reached.nonzero(as_tuple=False).view(-1)
            self._advance_waypoints(reached_ids)
            self._wp_reached_count[reached_ids] += 1

        active_wp = self._waypoints[:, 0]
        dist_to_wp = torch.where(reached, torch.norm(active_wp - board_xy, dim=-1), cur_dist)
        self._prev_dist_to_wp = dist_to_wp

        corners_local = torch.tensor([
            [BOARD_LENGTH / 2, BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [BOARD_LENGTH / 2, -BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [-BOARD_LENGTH / 2, BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
            [-BOARD_LENGTH / 2, -BOARD_WIDTH / 2, BOARD_THICKNESS / 2],
        ], device=self.device, dtype=torch.float32)
        corners_world = _transform_points(corners_local, board_pos, board_quat)
        corner_z = corners_world[:, :, 2]
        z_mean = corner_z.mean(dim=-1, keepdim=True)
        tilt = (corner_z - z_mean).abs().sum(dim=-1)

        height_loss = torch.clamp(0.97 - board_z, min=0.0)

        board_dying = (board_z < self.cfg.board_drop_height).float()

        collided = self._compute_obstacle_collision().float()

        board_yaw = _get_heading_yaw(board_quat)
        desired_yaws = {
            "robot_0": board_yaw + math.pi,
            "robot_1": board_yaw,
        }
        per_robot_facing_err = []
        for aid in self.cfg.possible_agents:
            robot = self.robots[aid]
            robot_yaw = _get_heading_yaw(robot.data.root_quat_w)
            angle_diff = desired_yaws[aid] - robot_yaw
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            per_robot_facing_err.append(angle_diff.abs())
        per_robot_facing_err = torch.stack(per_robot_facing_err, dim=0)
        facing_error = per_robot_facing_err.mean(dim=0)

        cos_y = torch.cos(board_yaw)
        sin_y = torch.sin(board_yaw)
        anchor_local_x = {
            "robot_0": BOARD_LENGTH / 2.0 + self.cfg.robot_offset,
            "robot_1": -(BOARD_LENGTH / 2.0 + self.cfg.robot_offset),
        }
        per_robot_offset_err = []
        for aid in self.cfg.possible_agents:
            rel = self.robots[aid].data.root_pos_w[:, :2] - board_xy
            local_x = cos_y * rel[:, 0] + sin_y * rel[:, 1]
            local_y = -sin_y * rel[:, 0] + cos_y * rel[:, 1]
            err = torch.sqrt((local_x - anchor_local_x[aid]) ** 2 + local_y ** 2 + 1e-8)
            per_robot_offset_err.append(err)
        per_robot_offset_err = torch.stack(per_robot_offset_err, dim=0)
        offset_error = per_robot_offset_err.mean(dim=0)

        if self.cfg.use_normalized_reward:
            seg = self._waypoints[:, 1] - self._waypoints[:, 0]
            seg_ok = torch.norm(seg, dim=-1, keepdim=True) > 1e-3
            dir_vec = torch.where(seg_ok, seg, active_wp - board_xy)
            desired_heading = torch.atan2(dir_vec[:, 1], dir_vec[:, 0])
            head_err_signed = (board_yaw + math.pi) - desired_heading
            head_err = torch.atan2(torch.sin(head_err_signed), torch.cos(head_err_signed)).abs()
            align = torch.clamp(torch.cos(head_err_signed), 0.0, 1.0)

            progress_gate = torch.where(progress > 0, align, torch.ones_like(align))
            r_progress = self.cfg.nrew_w_progress * progress * progress_gate
            r_wp = self.cfg.nrew_w_wp * reached.float()
            r_tilt = self.cfg.nrew_w_tilt * torch.exp(-((tilt / self.cfg.nrew_sigma_tilt) ** 2))
            r_height = self.cfg.nrew_w_height * torch.exp(-((height_loss / self.cfg.nrew_sigma_height) ** 2))
            r_facing = self.cfg.nrew_w_facing * torch.exp(
                -((per_robot_facing_err / self.cfg.nrew_sigma_facing) ** 2)
            ).mean(dim=0)
            r_offset = self.cfg.nrew_w_offset * torch.exp(
                -((per_robot_offset_err / self.cfg.nrew_sigma_offset) ** 2)
            ).mean(dim=0)
            r_alive = torch.full((self.num_envs,), self.cfg.nrew_w_alive, device=self.device)
            death_pen = board_dying * self.cfg.nrew_w_death
            _w_col = (self.cfg.nrew_w_collision if self.cfg.obstacle_collision_terminates
                      else self.cfg.nrew_w_collision_contact)
            collision_pen = collided * _w_col

            if self._obs_aabb.shape[0] > 0 and self.cfg.nrew_w_obstacle > 0.0:
                origins2 = self.scene.env_origins[:, :2]
                psi_max = torch.zeros(self.num_envs, device=self.device)
                for aid in self.cfg.possible_agents:
                    r = self.robots[aid]
                    rxy = r.data.root_pos_w[:, :2] - origins2
                    ryaw = _get_heading_yaw(r.data.root_quat_w)
                    psi = self._compute_lidar_normalized(rxy, ryaw)
                    psi_max = torch.maximum(psi_max, psi.max(dim=1).values)
                d_min = self.cfg.lidar_max_range * (1.0 - psi_max)
                proximity = torch.clamp(1.0 - d_min / self.cfg.obstacle_penalty_dist, min=0.0)
                obstacle_pen = self.cfg.nrew_w_obstacle * proximity ** 2
            else:
                obstacle_pen = torch.zeros(self.num_envs, device=self.device)

            if self.cfg.use_fixed_path and self._path.shape[0] > 0:
                at_goal = self._path_idx >= self._path.shape[0]
                newly_at_goal = at_goal & (~self._goal_reached)
                self._goal_reached |= at_goal
            else:
                newly_at_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            goal_bonus = newly_at_goal.float() * self.cfg.nrew_w_goal

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
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        for agent_id in ALL_ROBOT_IDS:
            robot = self.robots[agent_id]
            height = robot.data.root_pos_w[:, 2]
            fallen = height < self.cfg.termination_height

            gravity_b = quat_rotate_inverse(robot.data.root_quat_w, gravity_w)
            tilted = (gravity_b[:, 0].abs() > 0.7) | (gravity_b[:, 1].abs() > 0.7)

            terminated = terminated | fallen | tilted

        board_dropped = self._board.data.root_pos_w[:, 2] < self.cfg.board_drop_height
        terminated = terminated | board_dropped

        if self.cfg.obstacle_collision_terminates:
            terminated = terminated | self._compute_obstacle_collision()

        if self.cfg.terminate_on_goal and self.cfg.use_fixed_path:
            terminated = terminated | self._goal_reached

        timed_out = self.episode_length_buf >= self.max_episode_length

        term_dict = {aid: terminated for aid in self.cfg.possible_agents}
        timeout_dict = {aid: timed_out for aid in self.cfg.possible_agents}

        return term_dict, timeout_dict

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        num_reset = len(env_ids)

        for agent_id in ALL_ROBOT_IDS:
            robot = self.robots[agent_id]
            default_root_state = robot.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] += self.scene.env_origins[env_ids]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = torch.zeros_like(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

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
            self._target_start_pos[agent_id][env_ids] = self._target_pos_world[agent_id][env_ids]
            self._target_final_pos[agent_id][env_ids] = self._target_pos_world[agent_id][env_ids]
            self._target_start_yaw[agent_id][env_ids] = self._target_yaw_world[agent_id][env_ids]
            self._target_final_yaw[agent_id][env_ids] = self._target_yaw_world[agent_id][env_ids]

        for agent_id in self.cfg.possible_agents:
            self._obs_history[agent_id][env_ids] = 0.0

        board_default = self._board.data.default_root_state[env_ids].clone()
        board_default[:, :3] += self.scene.env_origins[env_ids]
        self._board.write_root_pose_to_sim(board_default[:, :7], env_ids)
        self._board.write_root_velocity_to_sim(board_default[:, 7:], env_ids)

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

        board_xy = board_default[:, :2]
        self._generate_waypoints(env_ids, board_xy)

        active_wp_reset = self._waypoints[env_ids, 0]
        self._prev_dist_to_wp[env_ids] = torch.norm(active_wp_reset - board_xy, dim=-1)

    def _compute_lidar_normalized(self, origin_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        N = origin_xy.shape[0]
        R = self.cfg.lidar_num_rays
        dmax = self.cfg.lidar_max_range
        M = self._obs_aabb.shape[0]
        if M == 0:
            return torch.zeros(N, R, device=self.device)

        ang = yaw.unsqueeze(1) + self._lidar_ray_offsets.unsqueeze(0)
        dx = torch.cos(ang)
        dy = torch.sin(ang)
        eps = 1e-6
        dx = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
        dy = torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)

        ox = origin_xy[:, 0].view(N, 1, 1)
        oy = origin_xy[:, 1].view(N, 1, 1)
        inv_dx = (1.0 / dx).unsqueeze(-1)
        inv_dy = (1.0 / dy).unsqueeze(-1)

        aabb = self._obs_aabb.view(1, 1, M, 4)
        x0, x1, y0, y1 = aabb[..., 0], aabb[..., 1], aabb[..., 2], aabb[..., 3]

        t1x = (x0 - ox) * inv_dx
        t2x = (x1 - ox) * inv_dx
        t1y = (y0 - oy) * inv_dy
        t2y = (y1 - oy) * inv_dy

        tmin_x = torch.minimum(t1x, t2x)
        tmax_x = torch.maximum(t1x, t2x)
        tmin_y = torch.minimum(t1y, t2y)
        tmax_y = torch.maximum(t1y, t2y)

        t_enter = torch.maximum(tmin_x, tmin_y)
        t_exit = torch.minimum(tmax_x, tmax_y)

        hit = (t_exit >= torch.clamp(t_enter, min=0.0)) & (t_enter <= dmax)
        dist = torch.clamp(t_enter, min=0.0)
        dist = torch.where(hit, dist, torch.full_like(dist, dmax))
        d_min = dist.min(dim=-1).values

        return 1.0 - torch.clamp(d_min, max=dmax) / dmax

    def _compute_obstacle_collision(self) -> torch.Tensor:
        M = self._obs_aabb.shape[0]
        if M == 0 or not self.cfg.use_obstacle_collision_termination:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        origins = self.scene.env_origins[:, :2]

        aabb = self._obs_aabb
        x0 = aabb[:, 0].view(1, M)
        x1 = aabb[:, 1].view(1, M)
        y0 = aabb[:, 2].view(1, M)
        y1 = aabb[:, 3].view(1, M)
        mx = ((aabb[:, 0] + aabb[:, 1]) * 0.5).view(1, M)
        my = ((aabb[:, 2] + aabb[:, 3]) * 0.5).view(1, M)
        hx = ((aabb[:, 1] - aabb[:, 0]) * 0.5).view(1, M)
        hy = ((aabb[:, 3] - aabb[:, 2]) * 0.5).view(1, M)

        hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        c = self._board.data.root_pos_w[:, :2] - origins
        yaw = _get_heading_yaw(self._board.data.root_quat_w)
        co = torch.cos(yaw).unsqueeze(1)
        si = torch.sin(yaw).unsqueeze(1)
        aco = co.abs()
        asi = si.abs()
        m = self.cfg.obstacle_collision_margin
        eu = BOARD_LENGTH / 2.0 + m
        ev = BOARD_WIDTH / 2.0 + m
        dx = c[:, 0:1] - mx
        dy = c[:, 1:2] - my
        ax_wx = dx.abs() <= hx + (eu * aco + ev * asi)
        ax_wy = dy.abs() <= hy + (eu * asi + ev * aco)
        du = (dx * co + dy * si).abs()
        ax_u = du <= eu + (hx * aco + hy * asi)
        dv = (-dx * si + dy * co).abs()
        ax_v = dv <= ev + (hx * asi + hy * aco)
        hit = hit | (ax_wx & ax_wy & ax_u & ax_v).any(dim=1)

        r = self.cfg.robot_collision_radius
        for aid in ALL_ROBOT_IDS:
            p = self.robots[aid].data.root_pos_w[:, :2] - origins
            px = p[:, 0:1]
            py = p[:, 1:2]
            cxp = torch.clamp(px, x0, x1)
            cyp = torch.clamp(py, y0, y1)
            dist2 = (px - cxp) ** 2 + (py - cyp) ** 2
            hit = hit | (dist2 <= r * r).any(dim=1)

        return hit

    def _refresh_path_window(self, env_ids: torch.Tensor):
        P = self._path.shape[0]
        origins = self.scene.env_origins[env_ids, :2]
        base = self._path_idx[env_ids]
        for k in range(NUM_WAYPOINTS):
            idx = torch.clamp(base + k, max=P - 1)
            self._waypoints[env_ids, k] = self._path[idx] + origins

    def _generate_waypoints(self, env_ids: torch.Tensor, board_xy: torch.Tensor):
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
        if self.cfg.use_fixed_path and self._path.shape[0] > 0:
            P = self._path.shape[0]
            self._path_idx[env_ids] = torch.clamp(self._path_idx[env_ids] + 1, max=P)
            self._refresh_path_window(env_ids)
            return

        num = len(env_ids)
        radius = self._curr_target_radius

        self._waypoints[env_ids, :NUM_WAYPOINTS - 1] = self._waypoints[env_ids, 1:].clone()

        last_wp = self._waypoints[env_ids, NUM_WAYPOINTS - 2]
        theta = torch.rand(num, device=self.device) * 2.0 * math.pi
        new_wp_x = last_wp[:, 0] + radius * torch.cos(theta)
        new_wp_y = last_wp[:, 1] + radius * torch.sin(theta)
        self._waypoints[env_ids, NUM_WAYPOINTS - 1, 0] = new_wp_x
        self._waypoints[env_ids, NUM_WAYPOINTS - 1, 1] = new_wp_y


    def _build_ll_obs(self, agent_id: str) -> torch.Tensor:
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

        current_pos = robot.data.root_pos_w[:, :2]
        current_yaw = _get_heading_yaw(root_quat)

        if agent_id in PAYLOAD_TRACKING_ROBOT_IDS:
            board_xy = self._board.data.root_pos_w[:, :2]
            board_vel = self._board.data.root_lin_vel_w[:, :2]
            board_yaw = _get_heading_yaw(self._board.data.root_quat_w)
            half_w = BOARD_WIDTH / 2.0 + self.cfg.robot_offset

            outward_x = -torch.sin(board_yaw)
            outward_y = torch.cos(board_yaw)

            vel_outward = board_vel[:, 0] * outward_x + board_vel[:, 1] * outward_y

            lookahead_t = 0.5
            lookahead_dist = torch.clamp(vel_outward * lookahead_t, min=0.0)

            target_x = board_xy[:, 0] + (half_w + lookahead_dist) * outward_x
            target_y = board_xy[:, 1] + (half_w + lookahead_dist) * outward_y
            target_yaw_val = board_yaw - math.pi / 2
        else:
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

        target_cmd = torch.stack([dx_body, dy_body, dyaw], dim=-1)

        ang_vel_scaled = root_ang_vel * self.cfg.obs_scale_ang_vel
        dof_vel_scaled = dof_vel * self.cfg.obs_scale_dof_vel

        ll_obs = torch.cat([
            self._ll_actions[agent_id],
            ang_vel_scaled,
            dof_pos,
            dof_vel_scaled,
            gait_clock,
            gravity_b,
            target_cmd,
            target_cmd,
        ], dim=-1)

        return ll_obs.clamp(-100.0, 100.0)


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return (w, 0.0, 0.0, z)


def _get_heading_yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_vec_by_quat(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    q_w = quat[:, 0:1]
    q_vec = quat[:, 1:4]
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec, t, dim=-1)


def _transform_points(
    points: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor
) -> torch.Tensor:
    N = pos.shape[0]
    P = points.shape[0]
    pts = points.unsqueeze(0).expand(N, P, 3).reshape(N * P, 3)
    q = quat.unsqueeze(1).expand(N, P, 4).reshape(N * P, 4)
    rotated = _rotate_vec_by_quat(pts, q)
    rotated = rotated.reshape(N, P, 3)
    return rotated + pos.unsqueeze(1)
