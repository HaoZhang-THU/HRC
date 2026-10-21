from __future__ import annotations

from dataclasses import dataclass, field
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
    C: Tensor
    c: Tensor
    mask: Tensor
    h: Tensor
    family: Tensor

    def margins(self, a: Tensor) -> Tensor:
        s = torch.einsum("bkd,bd->bk", self.C, a) - self.c
        return torch.where(self.mask > 0, s, torch.full_like(s, float("inf")))

    def violated(self, a: Tensor) -> Tensor:
        return (self.margins(a) < 0).any(dim=1)

    def detach(self) -> "RowSet":
        return RowSet(self.C.detach(), self.c.detach(), self.mask.detach(),
                      self.h.detach(), self.family)


def rot2(yaw: Tensor) -> Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2)


def _perp(v: Tensor) -> Tensor:
    return torch.stack((-v[..., 1], v[..., 0]), dim=-1)


def point_box_distance(p: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
    x0, x1, y0, y1 = (boxes[:, i].view(1, 1, -1) for i in range(4))
    px, py = p[..., 0:1], p[..., 1:2]
    cx = torch.minimum(torch.maximum(px, x0), x1)
    cy = torch.minimum(torch.maximum(py, y0), y1)
    gx, gy = px - cx, py - cy
    d_out = torch.sqrt(gx * gx + gy * gy + 1e-12)
    inside = (gx.abs() < 1e-9) & (gy.abs() < 1e-9)
    pen = torch.stack((px - x0, x1 - px, py - y0, y1 - py), dim=-1)
    pen_min, axis = pen.min(dim=-1)
    n_in = torch.zeros(p.shape[0], p.shape[1], boxes.shape[0], 2, device=p.device, dtype=p.dtype)
    n_in[..., 0] = torch.where(axis == 0, -1.0, torch.where(axis == 1, 1.0, 0.0))
    n_in[..., 1] = torch.where(axis == 2, -1.0, torch.where(axis == 3, 1.0, 0.0))
    n_out = torch.stack((gx, gy), dim=-1) / d_out.unsqueeze(-1)
    d = torch.where(inside, -pen_min, d_out)
    n = torch.where(inside.unsqueeze(-1), n_in, n_out)
    return d, n


class FormationSafetyModel:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        L, W = cfg.board_length, cfg.board_width
        xs = torch.linspace(-L / 2, L / 2, cfg.perimeter_samples)
        top = torch.stack((xs, torch.full_like(xs, W / 2)), 1)
        bot = torch.stack((xs, torch.full_like(xs, -W / 2)), 1)
        ends = torch.tensor([[L / 2, 0.0], [-L / 2, 0.0]])
        self.perimeter_local = torch.cat((top, bot, ends), 0)
        self.half = torch.tensor([L / 2, W / 2])

    def displacement_maps(self, st: SafetyState) -> tuple[Tensor, Tensor, Tensor]:
        B = st.board_xy.shape[0]
        N = st.controlled.shape[0]
        da = 3 * N
        dev, dt = st.board_xy.device, st.board_xy.dtype
        g = self.cfg.realization_gain
        s_lin = st.action_scale[0]
        s_yaw = st.action_scale[2]

        D = torch.zeros(B, N, 2, da, device=dev, dtype=dt)
        rho = torch.zeros(B, N, 2, device=dev, dtype=dt)
        for i in range(N):
            r = int(st.controlled[i])
            R = rot2(st.robot_yaw[:, r])
            D[:, i, :, 3 * i:3 * i + 2] = g * s_lin * R
            rho[:, i] = st.robot_xy[:, r] - st.board_xy
        V = D.mean(dim=1)
        rel = D - V.unsqueeze(1)
        cross = rho[..., 0:1] * rel[:, :, 1, :] - rho[..., 1:2] * rel[:, :, 0, :]
        denom = (rho * rho).sum(dim=(1, 2)).clamp_min(1e-6)
        Omega = cross.sum(dim=1) / denom.unsqueeze(-1)
        if self.cfg.dpsi_to_board_yaw != 0.0:
            for i in range(N):
                Omega[:, 3 * i + 2] += self.cfg.dpsi_to_board_yaw * s_yaw / N
        return D, V, Omega

    @staticmethod
    def point_maps(V: Tensor, Omega: Tensor, rho: Tensor) -> Tensor:
        return V.unsqueeze(1) + _perp(rho).unsqueeze(-1) * Omega[:, None, None, :]

    def rows(self, st: SafetyState) -> RowSet:
        cfg = self.cfg
        B = st.board_xy.shape[0]
        N = st.controlled.shape[0]
        da = 3 * N
        dev, dt = st.board_xy.device, st.board_xy.dtype
        eta = cfg.eta
        D, V, Omega = self.displacement_maps(st)
        Rb = rot2(st.board_yaw)
        curv_w = 0.5 * cfg.yaw_step_bound ** 2 if cfg.curvature_margin else 0.0

        C_list, c_list, h_list, f_list, m_list = [], [], [], [], []

        def _add(C, c, h, fam, active):
            C_list.append(C)
            c_list.append(c)
            h_list.append(h)
            f_list.append(torch.full_like(h, fam, dtype=torch.long))
            m_list.append(active.to(dt))

        per_local = self.perimeter_local.to(dev, dt)
        per_rho = torch.einsum("bij,pj->bpi", Rb, per_local)
        robot_rho = st.robot_xy - st.board_xy.unsqueeze(1)
        pts_rho = torch.cat((per_rho, robot_rho), dim=1)
        radius = torch.cat((torch.zeros(per_rho.shape[1], device=dev, dtype=dt),
                            torch.full((robot_rho.shape[1],), cfg.robot_radius, device=dev, dtype=dt)))
        pts = st.board_xy.unsqueeze(1) + pts_rho
        Pq = self.point_maps(V, Omega, pts_rho)
        curv = curv_w * pts_rho.norm(dim=-1)

        M = st.aabbs.shape[0]
        if M > 0:
            d, n = point_box_distance(pts, st.aabbs)
            h = d - radius.view(1, -1, 1) - cfg.obstacle_margin
            C = torch.einsum("bqmi,bqid->bqmd", n, Pq)
            c = -eta * h + curv.unsqueeze(-1)
            _add(C.reshape(B, -1, da), c.reshape(B, -1), h.reshape(B, -1),
                 F_COLLISION, (h <= cfg.activation_distance).reshape(B, -1))

            x0, x1, y0, y1 = (st.aabbs[:, i] for i in range(4))
            corners = torch.stack((torch.stack((x0, y0), 1), torch.stack((x1, y0), 1),
                                   torch.stack((x1, y1), 1), torch.stack((x0, y1), 1)), 1)
            corners = corners.reshape(-1, 2)
            rel = corners.view(1, -1, 2) - st.board_xy.unsqueeze(1)
            local = torch.einsum("bji,bqj->bqi", Rb, rel)
            half = self.half.to(dev, dt)
            closest = torch.maximum(torch.minimum(local, half), -half)
            gap = local - closest
            d_out = gap.norm(dim=-1)
            inside = d_out < 1e-9
            over = half - local.abs()
            over_min, axis = over.min(dim=-1)
            n_in = torch.zeros_like(local)
            sgn = torch.sign(local.gather(-1, axis.unsqueeze(-1)).squeeze(-1))
            n_in.scatter_(-1, axis.unsqueeze(-1), sgn.unsqueeze(-1))
            n_local = torch.where(inside.unsqueeze(-1), n_in, gap / d_out.clamp_min(1e-12).unsqueeze(-1))
            d = torch.where(inside, -over_min, d_out)
            n_world = torch.einsum("bij,bqj->bqi", Rb, n_local)
            contact_rho = torch.einsum("bij,bqj->bqi", Rb, closest)
            Pc = self.point_maps(V, Omega, contact_rho)
            h = d - cfg.obstacle_margin
            C = -torch.einsum("bqi,bqid->bqd", n_world, Pc)
            c = -eta * h + curv_w * contact_rho.norm(dim=-1)
            _add(C, c, h, F_COLLISION, h <= cfg.activation_distance)

        if cfg.use_grasp_rows:
            for i in range(N):
                r = int(st.controlled[i])
                anchor_rho = torch.einsum("bij,j->bi", Rb, st.anchors_local[r].to(dev, dt))
                e = torch.einsum("bji,bj->bi", Rb, st.robot_xy[:, r] - st.board_xy) - st.anchors_local[r].to(dev, dt)
                Pa = self.point_maps(V, Omega, anchor_rho.unsqueeze(1))[:, 0]
                E = torch.einsum("bji,bjd->bid", Rb, D[:, i] - Pa)
                for k in range(2):
                    for sgn_ in (1.0, -1.0):
                        h = cfg.grasp_band - sgn_ * e[:, k]
                        C = -sgn_ * E[:, k, :]
                        c = -eta * h
                        _add(C.unsqueeze(1), c.unsqueeze(1), h.unsqueeze(1), F_GRASP,
                             torch.ones(B, 1, device=dev, dtype=torch.bool))

        if cfg.use_rotation_rows:
            for sgn_ in (1.0, -1.0):
                h = torch.full((B, 1), cfg.yaw_step_bound, device=dev, dtype=dt)
                _add((sgn_ * Omega).unsqueeze(1), -h, h, F_ROTATION,
                     torch.ones(B, 1, device=dev, dtype=torch.bool))

        if cfg.workspace_box is not None:
            wx0, wx1, wy0, wy1 = cfg.workspace_box
            for axis_, sgn_, lim in ((0, 1.0, wx0), (0, -1.0, wx1), (1, 1.0, wy0), (1, -1.0, wy1)):
                h = sgn_ * (pts[..., axis_] - lim) - radius.view(1, -1) - cfg.obstacle_margin
                C = sgn_ * Pq[:, :, axis_, :]
                c = -eta * h + curv
                _add(C, c, h, F_WORKSPACE, h <= cfg.activation_distance)

        if not C_list:
            K = cfg.max_rows
            return RowSet(torch.zeros(B, K, da, device=dev, dtype=dt), torch.zeros(B, K, device=dev, dtype=dt),
                          torch.zeros(B, K, device=dev, dtype=dt), torch.zeros(B, K, device=dev, dtype=dt),
                          torch.zeros(B, K, device=dev, dtype=torch.long))

        C_all = torch.cat(C_list, 1)
        c_all = torch.cat(c_list, 1)
        h_all = torch.cat(h_list, 1)
        f_all = torch.cat(f_list, 1)
        m_all = torch.cat(m_list, 1)
        m_all = m_all * (C_all.norm(dim=-1) > 1e-9).to(dt)
        if cfg.recovery_cap is not None:
            c_all = torch.where(h_all < 0, c_all.clamp(max=cfg.recovery_cap), c_all)

        K = cfg.max_rows
        short = K - C_all.shape[1]
        if short > 0:
            C_all = torch.cat((C_all, C_all.new_zeros(B, short, da)), 1)
            c_all = torch.cat((c_all, c_all.new_zeros(B, short)), 1)
            h_all = torch.cat((h_all, h_all.new_zeros(B, short)), 1)
            f_all = torch.cat((f_all, f_all.new_zeros(B, short)), 1)
            m_all = torch.cat((m_all, m_all.new_zeros(B, short)), 1)
        score = torch.where(m_all > 0, h_all, torch.full_like(h_all, float("inf")))
        idx = score.topk(K, dim=1, largest=False).indices
        gidx = idx.unsqueeze(-1).expand(-1, -1, da)
        return RowSet(C_all.gather(1, gidx), c_all.gather(1, idx), m_all.gather(1, idx),
                      h_all.gather(1, idx), f_all.gather(1, idx))

    @torch.no_grad()
    def exact_projection(self, a: Tensor, rows: RowSet, fixed: Optional[Tensor] = None,
                         iterations: Optional[int] = None) -> tuple[Tensor, dict]:
        cfg = self.cfg
        iters = iterations or cfg.qp_iterations
        B, K, da = rows.C.shape
        dev, dt = a.device, a.dtype
        C, c, m = rows.C, rows.c.clone(), rows.mask
        if fixed is not None:
            fixed = fixed.to(dev)
            c = c - torch.einsum("bkd,bd->bk", C[:, :, fixed], a[:, fixed])
            free = ~fixed
            C = C[:, :, free]
            a_free = a[:, free]
        else:
            free = None
            a_free = a
        d = a_free.shape[1]
        eye = torch.eye(d, device=dev, dtype=dt).unsqueeze(0).expand(B, d, d)
        G = torch.cat((-C, eye, -eye), dim=1)
        g = torch.cat((-c, torch.ones(B, 2 * d, device=dev, dtype=dt)), dim=1)
        mk = torch.cat((m, torch.ones(B, 2 * d, device=dev, dtype=dt)), dim=1)
        H = torch.bmm(G, G.transpose(1, 2))
        b = torch.einsum("bmd,bd->bm", G, a_free) - g
        Hc, bc, mc = (t.detach().to("cpu", torch.float64) for t in (H, b, mk))
        diag = torch.diagonal(Hc, dim1=1, dim2=2).clamp_min(1e-12)
        lam_c = torch.zeros(B, K + 2 * d, dtype=torch.float64)
        Ht = Hc.transpose(0, 1).contiguous()
        for _ in range(iters):
            for k in range(K + 2 * d):
                r_k = bc[:, k] - (Ht[k] * lam_c).sum(-1)
                lam_c[:, k] = torch.clamp(lam_c[:, k] + r_k / diag[:, k], min=0.0) * mc[:, k]
        lam = lam_c.to(dev, dt)
        x = (a_free - torch.einsum("bmd,bm->bd", G, lam)).clamp(-1.0, 1.0)
        if free is not None:
            out = a.clone()
            out[:, free] = x
        else:
            out = x
        resid = torch.einsum("bkd,bd->bk", C, x) - c
        resid = torch.where(m > 0, resid, torch.full_like(resid, float("inf")))
        min_resid = resid.min(dim=1).values
        dual = lam[:, :K]
        active = (resid <= cfg.qp_tolerance) & (dual > 1e-7)
        info = {
            "dual": dual,
            "active": active,
            "min_residual": min_resid,
            "feasible": min_resid >= -cfg.qp_tolerance,
            "correction": (out - a).norm(dim=-1),
        }
        return out, info


def normalize_rows(rows: RowSet) -> RowSet:
    nrm = rows.C.norm(dim=-1)
    ok = (nrm > 1e-9) & (rows.mask > 0)
    scale = torch.where(ok, 1.0 / nrm.clamp_min(1e-9), torch.zeros_like(nrm))
    return RowSet(rows.C * scale.unsqueeze(-1), rows.c * scale, ok.to(rows.mask.dtype),
                  rows.h, rows.family)
