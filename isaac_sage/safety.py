from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

FAMILY_NAMES = ("collision", "grasp", "rotation", "workspace")
F_COLLISION, F_GRASP, F_ROTATION, F_WORKSPACE = range(4)


@dataclass
class SafetyConfig:
    board_length: float = 2.0
    board_width: float = 1.0
    robot_radius: float = 0.20
    perimeter_samples: int = 5
    obstacle_margin: float = 0.10
    activation_distance: float = 1.0
    eta: float = 1.0
    curvature_margin: bool = True
    recovery_cap: Optional[float] = 0.05
    use_grasp_rows: bool = True
    grasp_band: float = 0.25
    use_rotation_rows: bool = True
    yaw_step_bound: float = 0.30
    workspace_box: Optional[tuple] = None
    realization_gain: float = 1.0
    dpsi_to_board_yaw: float = 0.0
    max_rows: int = 32
    qp_iterations: int = 40
    qp_tolerance: float = 5e-3


@dataclass
class SafetyState:
    board_xy: Tensor
    board_yaw: Tensor
    robot_xy: Tensor
    robot_yaw: Tensor
    controlled: Tensor
    anchors_local: Tensor
    aabbs: Tensor
    action_scale: Tensor


@dataclass
class RowSet:
    coef: Tensor
    bound: Tensor
    mask: Tensor
    barrier: Tensor
    family: Tensor

    def margins(self, actions: Tensor) -> Tensor:
        slack = torch.einsum("bkd,bd->bk", self.coef, actions) - self.bound
        return torch.where(self.mask > 0, slack, torch.full_like(slack, float("inf")))

    def violated(self, actions: Tensor) -> Tensor:
        return (self.margins(actions) < 0).any(dim=1)

    def detach(self) -> "RowSet":
        return RowSet(self.coef.detach(), self.bound.detach(), self.mask.detach(),
                      self.barrier.detach(), self.family)


def rotation_2d(yaw: Tensor) -> Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2)


def _perpendicular(v: Tensor) -> Tensor:
    return torch.stack((-v[..., 1], v[..., 0]), dim=-1)


def point_box_distance(points: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
    x0, x1, y0, y1 = (boxes[:, i].view(1, 1, -1) for i in range(4))
    px, py = points[..., 0:1], points[..., 1:2]
    closest_x = torch.minimum(torch.maximum(px, x0), x1)
    closest_y = torch.minimum(torch.maximum(py, y0), y1)
    dx, dy = px - closest_x, py - closest_y
    dist_out = torch.sqrt(dx * dx + dy * dy + 1e-12)
    inside = (dx.abs() < 1e-9) & (dy.abs() < 1e-9)
    depth = torch.stack((px - x0, x1 - px, py - y0, y1 - py), dim=-1)
    min_depth, axis = depth.min(dim=-1)
    normal_in = torch.zeros(
        points.shape[0], points.shape[1], boxes.shape[0], 2, device=points.device, dtype=points.dtype
    )
    normal_in[..., 0] = torch.where(axis == 0, -1.0, torch.where(axis == 1, 1.0, 0.0))
    normal_in[..., 1] = torch.where(axis == 2, -1.0, torch.where(axis == 3, 1.0, 0.0))
    normal_out = torch.stack((dx, dy), dim=-1) / dist_out.unsqueeze(-1)
    dist = torch.where(inside, -min_depth, dist_out)
    normal = torch.where(inside.unsqueeze(-1), normal_in, normal_out)
    return dist, normal


class FormationSafetyModel:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        length, width = cfg.board_length, cfg.board_width
        xs = torch.linspace(-length / 2, length / 2, cfg.perimeter_samples)
        top = torch.stack((xs, torch.full_like(xs, width / 2)), 1)
        bottom = torch.stack((xs, torch.full_like(xs, -width / 2)), 1)
        ends = torch.tensor([[length / 2, 0.0], [-length / 2, 0.0]])
        self.perimeter_local = torch.cat((top, bottom, ends), 0)
        self.half = torch.tensor([length / 2, width / 2])

    def displacement_maps(self, state: SafetyState) -> tuple[Tensor, Tensor, Tensor]:
        num_envs = state.board_xy.shape[0]
        num_agents = state.controlled.shape[0]
        act_dim = 3 * num_agents
        device, dtype = state.board_xy.device, state.board_xy.dtype
        gain = self.cfg.realization_gain
        lin_scale = state.action_scale[0]
        yaw_scale = state.action_scale[2]

        robot_maps = torch.zeros(num_envs, num_agents, 2, act_dim, device=device, dtype=dtype)
        offsets = torch.zeros(num_envs, num_agents, 2, device=device, dtype=dtype)
        for i in range(num_agents):
            robot = int(state.controlled[i])
            rot = rotation_2d(state.robot_yaw[:, robot])
            robot_maps[:, i, :, 3 * i:3 * i + 2] = gain * lin_scale * rot
            offsets[:, i] = state.robot_xy[:, robot] - state.board_xy
        vel_map = robot_maps.mean(dim=1)
        rel = robot_maps - vel_map.unsqueeze(1)
        cross = offsets[..., 0:1] * rel[:, :, 1, :] - offsets[..., 1:2] * rel[:, :, 0, :]
        denom = (offsets * offsets).sum(dim=(1, 2)).clamp_min(1e-6)
        yaw_map = cross.sum(dim=1) / denom.unsqueeze(-1)
        if self.cfg.dpsi_to_board_yaw != 0.0:
            for i in range(num_agents):
                yaw_map[:, 3 * i + 2] += self.cfg.dpsi_to_board_yaw * yaw_scale / num_agents
        return robot_maps, vel_map, yaw_map

    @staticmethod
    def point_maps(vel_map: Tensor, yaw_map: Tensor, offsets: Tensor) -> Tensor:
        return vel_map.unsqueeze(1) + _perpendicular(offsets).unsqueeze(-1) * yaw_map[:, None, None, :]

    def rows(self, state: SafetyState) -> RowSet:
        cfg = self.cfg
        num_envs = state.board_xy.shape[0]
        num_agents = state.controlled.shape[0]
        act_dim = 3 * num_agents
        device, dtype = state.board_xy.device, state.board_xy.dtype
        eta = cfg.eta
        robot_maps, vel_map, yaw_map = self.displacement_maps(state)
        board_rot = rotation_2d(state.board_yaw)
        curv_weight = 0.5 * cfg.yaw_step_bound ** 2 if cfg.curvature_margin else 0.0

        coef_list, bound_list, barrier_list, family_list, mask_list = [], [], [], [], []

        def _add(coef, bound, barrier, family, active):
            coef_list.append(coef)
            bound_list.append(bound)
            barrier_list.append(barrier)
            family_list.append(torch.full_like(barrier, family, dtype=torch.long))
            mask_list.append(active.to(dtype))

        perimeter = self.perimeter_local.to(device, dtype)
        perimeter_offsets = torch.einsum("bij,pj->bpi", board_rot, perimeter)
        robot_offsets = state.robot_xy - state.board_xy.unsqueeze(1)
        point_offsets = torch.cat((perimeter_offsets, robot_offsets), dim=1)
        radius = torch.cat((torch.zeros(perimeter_offsets.shape[1], device=device, dtype=dtype),
                            torch.full((robot_offsets.shape[1],), cfg.robot_radius, device=device, dtype=dtype)))
        points = state.board_xy.unsqueeze(1) + point_offsets
        point_map = self.point_maps(vel_map, yaw_map, point_offsets)
        curv = curv_weight * point_offsets.norm(dim=-1)

        num_boxes = state.aabbs.shape[0]
        if num_boxes > 0:
            dist, normal = point_box_distance(points, state.aabbs)
            barrier = dist - radius.view(1, -1, 1) - cfg.obstacle_margin
            coef = torch.einsum("bqmi,bqid->bqmd", normal, point_map)
            bound = -eta * barrier + curv.unsqueeze(-1)
            _add(coef.reshape(num_envs, -1, act_dim), bound.reshape(num_envs, -1), barrier.reshape(num_envs, -1),
                 F_COLLISION, (barrier <= cfg.activation_distance).reshape(num_envs, -1))

            x0, x1, y0, y1 = (state.aabbs[:, i] for i in range(4))
            corners = torch.stack((torch.stack((x0, y0), 1), torch.stack((x1, y0), 1),
                                   torch.stack((x1, y1), 1), torch.stack((x0, y1), 1)), 1)
            corners = corners.reshape(-1, 2)
            rel = corners.view(1, -1, 2) - state.board_xy.unsqueeze(1)
            local = torch.einsum("bji,bqj->bqi", board_rot, rel)
            half = self.half.to(device, dtype)
            closest = torch.maximum(torch.minimum(local, half), -half)
            gap = local - closest
            dist_out = gap.norm(dim=-1)
            inside = dist_out < 1e-9
            over = half - local.abs()
            min_over, axis = over.min(dim=-1)
            normal_in = torch.zeros_like(local)
            sign = torch.sign(local.gather(-1, axis.unsqueeze(-1)).squeeze(-1))
            normal_in.scatter_(-1, axis.unsqueeze(-1), sign.unsqueeze(-1))
            normal_local = torch.where(inside.unsqueeze(-1), normal_in, gap / dist_out.clamp_min(1e-12).unsqueeze(-1))
            dist = torch.where(inside, -min_over, dist_out)
            normal_world = torch.einsum("bij,bqj->bqi", board_rot, normal_local)
            contact_offsets = torch.einsum("bij,bqj->bqi", board_rot, closest)
            contact_map = self.point_maps(vel_map, yaw_map, contact_offsets)
            barrier = dist - cfg.obstacle_margin
            coef = -torch.einsum("bqi,bqid->bqd", normal_world, contact_map)
            bound = -eta * barrier + curv_weight * contact_offsets.norm(dim=-1)
            _add(coef, bound, barrier, F_COLLISION, barrier <= cfg.activation_distance)

        if cfg.use_grasp_rows:
            for i in range(num_agents):
                robot = int(state.controlled[i])
                anchor_offset = torch.einsum("bij,j->bi", board_rot, state.anchors_local[robot].to(device, dtype))
                grasp_error = torch.einsum(
                    "bji,bj->bi", board_rot, state.robot_xy[:, robot] - state.board_xy
                ) - state.anchors_local[robot].to(device, dtype)
                anchor_map = self.point_maps(vel_map, yaw_map, anchor_offset.unsqueeze(1))[:, 0]
                error_map = torch.einsum("bji,bjd->bid", board_rot, robot_maps[:, i] - anchor_map)
                for k in range(2):
                    for side in (1.0, -1.0):
                        barrier = cfg.grasp_band - side * grasp_error[:, k]
                        coef = -side * error_map[:, k, :]
                        bound = -eta * barrier
                        _add(coef.unsqueeze(1), bound.unsqueeze(1), barrier.unsqueeze(1), F_GRASP,
                             torch.ones(num_envs, 1, device=device, dtype=torch.bool))

        if cfg.use_rotation_rows:
            for side in (1.0, -1.0):
                barrier = torch.full((num_envs, 1), cfg.yaw_step_bound, device=device, dtype=dtype)
                _add((side * yaw_map).unsqueeze(1), -barrier, barrier, F_ROTATION,
                     torch.ones(num_envs, 1, device=device, dtype=torch.bool))

        if cfg.workspace_box is not None:
            wx0, wx1, wy0, wy1 = cfg.workspace_box
            for ax, side, limit in ((0, 1.0, wx0), (0, -1.0, wx1), (1, 1.0, wy0), (1, -1.0, wy1)):
                barrier = side * (points[..., ax] - limit) - radius.view(1, -1) - cfg.obstacle_margin
                coef = side * point_map[:, :, ax, :]
                bound = -eta * barrier + curv
                _add(coef, bound, barrier, F_WORKSPACE, barrier <= cfg.activation_distance)

        if not coef_list:
            num_rows = cfg.max_rows
            return RowSet(
                torch.zeros(num_envs, num_rows, act_dim, device=device, dtype=dtype),
                torch.zeros(num_envs, num_rows, device=device, dtype=dtype),
                torch.zeros(num_envs, num_rows, device=device, dtype=dtype),
                torch.zeros(num_envs, num_rows, device=device, dtype=dtype),
                torch.zeros(num_envs, num_rows, device=device, dtype=torch.long),
            )

        coef_all = torch.cat(coef_list, 1)
        bound_all = torch.cat(bound_list, 1)
        barrier_all = torch.cat(barrier_list, 1)
        family_all = torch.cat(family_list, 1)
        mask_all = torch.cat(mask_list, 1)
        mask_all = mask_all * (coef_all.norm(dim=-1) > 1e-9).to(dtype)
        if cfg.recovery_cap is not None:
            bound_all = torch.where(barrier_all < 0, bound_all.clamp(max=cfg.recovery_cap), bound_all)

        num_rows = cfg.max_rows
        pad = num_rows - coef_all.shape[1]
        if pad > 0:
            coef_all = torch.cat((coef_all, coef_all.new_zeros(num_envs, pad, act_dim)), 1)
            bound_all = torch.cat((bound_all, bound_all.new_zeros(num_envs, pad)), 1)
            barrier_all = torch.cat((barrier_all, barrier_all.new_zeros(num_envs, pad)), 1)
            family_all = torch.cat((family_all, family_all.new_zeros(num_envs, pad)), 1)
            mask_all = torch.cat((mask_all, mask_all.new_zeros(num_envs, pad)), 1)
        score = torch.where(mask_all > 0, barrier_all, torch.full_like(barrier_all, float("inf")))
        idx = score.topk(num_rows, dim=1, largest=False).indices
        coef_index = idx.unsqueeze(-1).expand(-1, -1, act_dim)
        return RowSet(coef_all.gather(1, coef_index), bound_all.gather(1, idx), mask_all.gather(1, idx),
                      barrier_all.gather(1, idx), family_all.gather(1, idx))

    @torch.no_grad()
    def exact_projection(self, actions: Tensor, rows: RowSet, fixed: Optional[Tensor] = None,
                         iterations: Optional[int] = None) -> tuple[Tensor, dict]:
        cfg = self.cfg
        num_iters = iterations or cfg.qp_iterations
        num_envs, num_rows, act_dim = rows.coef.shape
        device, dtype = actions.device, actions.dtype
        coef, bound, mask = rows.coef, rows.bound.clone(), rows.mask
        if fixed is not None:
            fixed = fixed.to(device)
            bound = bound - torch.einsum("bkd,bd->bk", coef[:, :, fixed], actions[:, fixed])
            free = ~fixed
            coef = coef[:, :, free]
            free_actions = actions[:, free]
        else:
            free = None
            free_actions = actions
        dim = free_actions.shape[1]
        eye = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0).expand(num_envs, dim, dim)
        lhs = torch.cat((-coef, eye, -eye), dim=1)
        rhs = torch.cat((-bound, torch.ones(num_envs, 2 * dim, device=device, dtype=dtype)), dim=1)
        full_mask = torch.cat((mask, torch.ones(num_envs, 2 * dim, device=device, dtype=dtype)), dim=1)
        gram = torch.bmm(lhs, lhs.transpose(1, 2))
        offset = torch.einsum("bmd,bd->bm", lhs, free_actions) - rhs
        gram_cpu, offset_cpu, mask_cpu = (
            tensor.detach().to("cpu", torch.float64) for tensor in (gram, offset, full_mask)
        )
        diag = torch.diagonal(gram_cpu, dim1=1, dim2=2).clamp_min(1e-12)
        dual_cpu = torch.zeros(num_envs, num_rows + 2 * dim, dtype=torch.float64)
        gram_t = gram_cpu.transpose(0, 1).contiguous()
        for _ in range(num_iters):
            for k in range(num_rows + 2 * dim):
                res = offset_cpu[:, k] - (gram_t[k] * dual_cpu).sum(-1)
                dual_cpu[:, k] = torch.clamp(dual_cpu[:, k] + res / diag[:, k], min=0.0) * mask_cpu[:, k]
        duals = dual_cpu.to(device, dtype)
        solution = (free_actions - torch.einsum("bmd,bm->bd", lhs, duals)).clamp(-1.0, 1.0)
        if free is not None:
            out = actions.clone()
            out[:, free] = solution
        else:
            out = solution
        residual = torch.einsum("bkd,bd->bk", coef, solution) - bound
        residual = torch.where(mask > 0, residual, torch.full_like(residual, float("inf")))
        min_residual = residual.min(dim=1).values
        dual = duals[:, :num_rows]
        active = (residual <= cfg.qp_tolerance) & (dual > 1e-7)
        info = {
            "dual": dual,
            "active": active,
            "min_residual": min_residual,
            "feasible": min_residual >= -cfg.qp_tolerance,
            "correction": (out - actions).norm(dim=-1),
        }
        return out, info


def normalize_rows(rows: RowSet) -> RowSet:
    norms = rows.coef.norm(dim=-1)
    ok = (norms > 1e-9) & (rows.mask > 0)
    scale = torch.where(ok, 1.0 / norms.clamp_min(1e-9), torch.zeros_like(norms))
    return RowSet(rows.coef * scale.unsqueeze(-1), rows.bound * scale, ok.to(rows.mask.dtype),
                  rows.barrier, rows.family)
