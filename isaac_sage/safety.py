"""Joint shared-payload safety model and exact CBF-QP (paper Sec. III-A).

Everything here is batched over parallel environments and written in torch so
the same code serves the rollout filter (Eq. 18), the SAIL training surrogate
(Eq. 27, see ``sail.py``) and the deployment controller (Alg. 2).

Safety state  x = Phi(s)  (Eq. 7 ff.)
    payload planar pose (centre, yaw) and the planar position / heading of
    every carrier, all in the env-local frame, plus the static wall boxes.

One-step predictor  x_{t+1} = F(x_t) + B(x_t) a_t  (Eq. 11)
    carrier i is commanded a body-frame displacement a_i = [dx, dy, dpsi]
    (normalised to [-1, 1]^3, scaled by ``action_scale``).  Within one
    high-level period the whole-body controller realises a fraction
    ``realization_gain`` of it, so carrier i translates by
        d_i = g * S * R(psi_i) a_i[:2].
    The payload is carried rigidly, so its planar twist is affine in a:
        v     = mean_i d_i                               (centre displacement)
        omega = sum_i rho_i x (d_i - v) / sum_i |rho_i|^2 (+ optional dpsi term)
    and any point rigidly attached to the payload at offset rho moves by
        delta q = v + rho_perp * omega  =  P_q(x) a.
    The follower slot (robot_2) tracks the payload and is treated as a point
    attached to it.

Conservative affine rows  C_j(x) a >= c_j(x)  (Eq. 12-16)
    discrete-time barrier condition h_j(x+) >= (1 - eta_j) h_j(x) (Eq. 10)
    with h_j linearised along the current contact normal n_j:
        h_j(x+) ~= h_j(x) + n_j . delta q  =>  n_j^T P_q a >= -eta_j h_j + m_j,
    where m_j is the rotation-curvature margin that bounds the linearisation
    error over the one-step reachable set (Eq. 15).  Barrier families:
        0  collision   payload perimeter / carrier discs vs wall boxes,
                       wall corners vs the payload rectangle
        1  grasp       each carrier's offset from its holding anchor stays
                       inside a box band (payload-drop precursor)
        2  rotation    |payload yaw increment| <= bound
        3  workspace   optional planar box the formation must stay inside

Exact execution filter  Q_CBF(x, a)  (Eq. 18)
    Euclidean projection onto {C a >= c, a in [-1, 1]^da}; batched Hildreth
    dual iteration.  ``fixed`` marks measured (non-optimised) action blocks for
    the human-partner variant (Eq. 19).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

FAMILY_NAMES = ("collision", "grasp", "rotation", "workspace")
F_COLLISION, F_GRASP, F_ROTATION, F_WORKSPACE = range(4)


@dataclass
class SafetyConfig:
    # geometry
    board_length: float = 2.0
    board_width: float = 1.0
    robot_radius: float = 0.20
    perimeter_samples: int = 5        # samples per long payload edge
    # barrier parameters
    obstacle_margin: float = 0.10     # m, folded into h (h = clearance - margin)
    activation_distance: float = 1.0  # rows with h above this are dropped
    eta: float = 1.0                  # discrete-time decay in Eq. 10
    curvature_margin: bool = True     # Eq. 15 rotation-linearisation margin
    # Rows that are already violated (h < 0) would demand a one-step recovery
    # of eta*|h|; capping that demand at recovery_cap (m, in the predictor's
    # displacement units) keeps the joint QP feasible in the overwhelming
    # majority of states while preserving the barrier condition for h >= 0.
    # None reproduces the uncapped condition of Eq. 10.
    recovery_cap: Optional[float] = 0.05
    # grasp band (family 1)
    use_grasp_rows: bool = True
    grasp_band: float = 0.25          # m, half-width of the anchor box
    # rotation bound (family 2)
    use_rotation_rows: bool = True
    yaw_step_bound: float = 0.30      # rad per high-level step
    # optional workspace box (family 3), env-local (x0, x1, y0, y1)
    workspace_box: Optional[tuple] = None
    # predictor
    realization_gain: float = 1.0     # fraction of the command realised in one step
    dpsi_to_board_yaw: float = 0.0    # weight of commanded carrier yaw in payload yaw
    # row bookkeeping
    max_rows: int = 32                # rows kept per env after activation masking
    # QP
    qp_iterations: int = 40
    qp_tolerance: float = 5e-3


@dataclass
class SafetyState:
    """Batched safety-relevant state x = Phi(s), env-local frame."""
    board_xy: Tensor          # [B, 2]
    board_yaw: Tensor         # [B]
    robot_xy: Tensor          # [B, R, 2]   every carrier slot (controlled + follower)
    robot_yaw: Tensor         # [B, R]
    controlled: Tensor        # [N] long, indices into the R slots that are commanded
    anchors_local: Tensor     # [R, 2] holding anchors in the payload frame
    aabbs: Tensor             # [M, 4] (x0, x1, y0, y1) static wall boxes
    action_scale: Tensor      # [3] metres / radians per unit normalised action


@dataclass
class RowSet:
    """Stacked affine safety rows C a >= c (Eq. 16), zero-padded to K."""
    C: Tensor        # [B, K, da]
    c: Tensor        # [B, K]
    mask: Tensor     # [B, K] 1 = row present
    h: Tensor        # [B, K] current barrier value h_j(x)
    family: Tensor   # [B, K] long

    def margins(self, a: Tensor) -> Tensor:
        """Per-row slack C a - c; +inf on padded rows."""
        s = torch.einsum("bkd,bd->bk", self.C, a) - self.c
        return torch.where(self.mask > 0, s, torch.full_like(s, float("inf")))

    def violated(self, a: Tensor) -> Tensor:
        """Proposal-violation indicator (any present row with C a < c)."""
        return (self.margins(a) < 0).any(dim=1)

    def detach(self) -> "RowSet":
        return RowSet(self.C.detach(), self.c.detach(), self.mask.detach(),
                      self.h.detach(), self.family)


def rot2(yaw: Tensor) -> Tensor:
    """[..., 2, 2] planar rotation matrices."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2)


def _perp(v: Tensor) -> Tensor:
    """rho_perp = (-rho_y, rho_x) so that rho_perp * omega is the rotational displacement."""
    return torch.stack((-v[..., 1], v[..., 0]), dim=-1)


def point_box_distance(p: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Signed distance and outward normal from each box to each point.

    p: [B, P, 2]; boxes: [M, 4] -> d [B, P, M], n [B, P, M, 2] (n points from
    the box surface towards the point; inside the box, d < 0 and n is the
    minimum-penetration exit direction).
    """
    x0, x1, y0, y1 = (boxes[:, i].view(1, 1, -1) for i in range(4))
    px, py = p[..., 0:1], p[..., 1:2]
    cx = torch.minimum(torch.maximum(px, x0), x1)
    cy = torch.minimum(torch.maximum(py, y0), y1)
    gx, gy = px - cx, py - cy
    d_out = torch.sqrt(gx * gx + gy * gy + 1e-12)
    inside = (gx.abs() < 1e-9) & (gy.abs() < 1e-9)
    # penetration depth along each face
    pen = torch.stack((px - x0, x1 - px, py - y0, y1 - py), dim=-1)  # [B,P,M,4]
    pen_min, axis = pen.min(dim=-1)
    n_in = torch.zeros(p.shape[0], p.shape[1], boxes.shape[0], 2, device=p.device, dtype=p.dtype)
    n_in[..., 0] = torch.where(axis == 0, -1.0, torch.where(axis == 1, 1.0, 0.0))
    n_in[..., 1] = torch.where(axis == 2, -1.0, torch.where(axis == 3, 1.0, 0.0))
    n_out = torch.stack((gx, gy), dim=-1) / d_out.unsqueeze(-1)
    d = torch.where(inside, -pen_min, d_out)
    n = torch.where(inside.unsqueeze(-1), n_in, n_out)
    return d, n


class FormationSafetyModel:
    """Builds the affine row set (Eq. 16) and solves the exact QP (Eq. 18)."""

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        L, W = cfg.board_length, cfg.board_width
        xs = torch.linspace(-L / 2, L / 2, cfg.perimeter_samples)
        top = torch.stack((xs, torch.full_like(xs, W / 2)), 1)
        bot = torch.stack((xs, torch.full_like(xs, -W / 2)), 1)
        ends = torch.tensor([[L / 2, 0.0], [-L / 2, 0.0]])
        self.perimeter_local = torch.cat((top, bot, ends), 0)   # [P, 2]
        self.half = torch.tensor([L / 2, W / 2])

    # ------------------------------------------------------------------ #
    # predictor (Eq. 11): affine maps from the joint action to displacements
    # ------------------------------------------------------------------ #
    def displacement_maps(self, st: SafetyState) -> tuple[Tensor, Tensor, Tensor]:
        """Returns D [B, N, 2, da] (carrier translation), V [B, 2, da] (payload
        centre), Omega [B, da] (payload yaw increment)."""
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
            R = rot2(st.robot_yaw[:, r])                     # [B, 2, 2]
            D[:, i, :, 3 * i:3 * i + 2] = g * s_lin * R
            rho[:, i] = st.robot_xy[:, r] - st.board_xy
        V = D.mean(dim=1)                                    # [B, 2, da]
        rel = D - V.unsqueeze(1)                             # [B, N, 2, da]
        cross = rho[..., 0:1] * rel[:, :, 1, :] - rho[..., 1:2] * rel[:, :, 0, :]  # [B, N, da]
        denom = (rho * rho).sum(dim=(1, 2)).clamp_min(1e-6)  # [B]
        Omega = cross.sum(dim=1) / denom.unsqueeze(-1)       # [B, da]
        if self.cfg.dpsi_to_board_yaw != 0.0:
            for i in range(N):
                Omega[:, 3 * i + 2] += self.cfg.dpsi_to_board_yaw * s_yaw / N
        return D, V, Omega

    @staticmethod
    def point_maps(V: Tensor, Omega: Tensor, rho: Tensor) -> Tensor:
        """P_q for payload-attached points: rho [B, P, 2] -> [B, P, 2, da]."""
        return V.unsqueeze(1) + _perp(rho).unsqueeze(-1) * Omega[:, None, None, :]

    # ------------------------------------------------------------------ #
    # affine rows (Eq. 12-16)
    # ------------------------------------------------------------------ #
    def rows(self, st: SafetyState) -> RowSet:
        cfg = self.cfg
        B = st.board_xy.shape[0]
        N = st.controlled.shape[0]
        da = 3 * N
        dev, dt = st.board_xy.device, st.board_xy.dtype
        eta = cfg.eta
        D, V, Omega = self.displacement_maps(st)
        Rb = rot2(st.board_yaw)                                  # [B, 2, 2]
        curv_w = 0.5 * cfg.yaw_step_bound ** 2 if cfg.curvature_margin else 0.0

        C_list, c_list, h_list, f_list, m_list = [], [], [], [], []

        def _add(C, c, h, fam, active):
            C_list.append(C)
            c_list.append(c)
            h_list.append(h)
            f_list.append(torch.full_like(h, fam, dtype=torch.long))
            m_list.append(active.to(dt))

        # -- check points rigidly attached to the payload: perimeter samples + carriers
        per_local = self.perimeter_local.to(dev, dt)             # [P, 2]
        per_rho = torch.einsum("bij,pj->bpi", Rb, per_local)     # [B, P, 2]
        robot_rho = st.robot_xy - st.board_xy.unsqueeze(1)       # [B, R, 2]
        pts_rho = torch.cat((per_rho, robot_rho), dim=1)         # [B, P+R, 2]
        radius = torch.cat((torch.zeros(per_rho.shape[1], device=dev, dtype=dt),
                            torch.full((robot_rho.shape[1],), cfg.robot_radius, device=dev, dtype=dt)))
        pts = st.board_xy.unsqueeze(1) + pts_rho                 # [B, Q, 2]
        Pq = self.point_maps(V, Omega, pts_rho)                  # [B, Q, 2, da]
        curv = curv_w * pts_rho.norm(dim=-1)                     # [B, Q]

        M = st.aabbs.shape[0]
        if M > 0:
            # family 0a: payload points / carrier discs vs wall boxes
            d, n = point_box_distance(pts, st.aabbs)             # [B,Q,M], [B,Q,M,2]
            h = d - radius.view(1, -1, 1) - cfg.obstacle_margin
            C = torch.einsum("bqmi,bqid->bqmd", n, Pq)           # n^T P_q
            c = -eta * h + curv.unsqueeze(-1)
            _add(C.reshape(B, -1, da), c.reshape(B, -1), h.reshape(B, -1),
                 F_COLLISION, (h <= cfg.activation_distance).reshape(B, -1))

            # family 0b: wall corners vs the payload rectangle
            x0, x1, y0, y1 = (st.aabbs[:, i] for i in range(4))
            corners = torch.stack((torch.stack((x0, y0), 1), torch.stack((x1, y0), 1),
                                   torch.stack((x1, y1), 1), torch.stack((x0, y1), 1)), 1)
            corners = corners.reshape(-1, 2)                      # [4M, 2]
            rel = corners.view(1, -1, 2) - st.board_xy.unsqueeze(1)      # [B, 4M, 2]
            local = torch.einsum("bji,bqj->bqi", Rb, rel)        # R^T rel
            half = self.half.to(dev, dt)
            closest = torch.maximum(torch.minimum(local, half), -half)
            gap = local - closest
            d_out = gap.norm(dim=-1)
            inside = d_out < 1e-9
            over = half - local.abs()                             # [B, 4M, 2]
            over_min, axis = over.min(dim=-1)
            n_in = torch.zeros_like(local)
            sgn = torch.sign(local.gather(-1, axis.unsqueeze(-1)).squeeze(-1))
            n_in.scatter_(-1, axis.unsqueeze(-1), sgn.unsqueeze(-1))
            n_local = torch.where(inside.unsqueeze(-1), n_in, gap / d_out.clamp_min(1e-12).unsqueeze(-1))
            d = torch.where(inside, -over_min, d_out)
            n_world = torch.einsum("bij,bqj->bqi", Rb, n_local)  # from payload surface to corner
            contact_rho = torch.einsum("bij,bqj->bqi", Rb, closest)
            Pc = self.point_maps(V, Omega, contact_rho)          # [B, 4M, 2, da]
            h = d - cfg.obstacle_margin
            C = -torch.einsum("bqi,bqid->bqd", n_world, Pc)      # payload point moving towards the corner closes the gap
            c = -eta * h + curv_w * contact_rho.norm(dim=-1)
            _add(C, c, h, F_COLLISION, h <= cfg.activation_distance)

        # family 1: grasp band  |e_i + E_i a| <= band  (per axis, both signs)
        if cfg.use_grasp_rows:
            for i in range(N):
                r = int(st.controlled[i])
                anchor_rho = torch.einsum("bij,j->bi", Rb, st.anchors_local[r].to(dev, dt))  # [B, 2]
                e = torch.einsum("bji,bj->bi", Rb, st.robot_xy[:, r] - st.board_xy) - st.anchors_local[r].to(dev, dt)
                Pa = self.point_maps(V, Omega, anchor_rho.unsqueeze(1))[:, 0]              # [B, 2, da]
                E = torch.einsum("bji,bjd->bid", Rb, D[:, i] - Pa)                           # R^T (D_i - P_anchor)
                for k in range(2):
                    for sgn_ in (1.0, -1.0):
                        h = cfg.grasp_band - sgn_ * e[:, k]
                        C = -sgn_ * E[:, k, :]
                        c = -eta * h
                        _add(C.unsqueeze(1), c.unsqueeze(1), h.unsqueeze(1), F_GRASP,
                             torch.ones(B, 1, device=dev, dtype=torch.bool))

        # family 2: payload yaw increment bound  -bound <= Omega a <= bound
        if cfg.use_rotation_rows:
            for sgn_ in (1.0, -1.0):
                h = torch.full((B, 1), cfg.yaw_step_bound, device=dev, dtype=dt)
                _add((sgn_ * Omega).unsqueeze(1), -h, h, F_ROTATION,
                     torch.ones(B, 1, device=dev, dtype=torch.bool))

        # family 3: optional workspace box for every check point
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
        # rows with a vanishing coefficient vector carry no information
        m_all = m_all * (C_all.norm(dim=-1) > 1e-9).to(dt)
        if cfg.recovery_cap is not None:
            c_all = torch.where(h_all < 0, c_all.clamp(max=cfg.recovery_cap), c_all)

        # keep the max_rows most critical (smallest h) present rows per env;
        # always return exactly max_rows (zero-padded) so rollout buffers have a fixed shape
        K = cfg.max_rows
        short = K - C_all.shape[1]
        if short > 0:
            C_all = torch.cat((C_all, C_all.new_zeros(B, short, da)), 1)
            c_all = torch.cat((c_all, c_all.new_zeros(B, short)), 1)
            h_all = torch.cat((h_all, h_all.new_zeros(B, short)), 1)
            f_all = torch.cat((f_all, f_all.new_zeros(B, short)), 1)
            m_all = torch.cat((m_all, m_all.new_zeros(B, short)), 1)
        score = torch.where(m_all > 0, h_all, torch.full_like(h_all, float("inf")))
        idx = score.topk(K, dim=1, largest=False).indices           # [B, K]
        gidx = idx.unsqueeze(-1).expand(-1, -1, da)
        return RowSet(C_all.gather(1, gidx), c_all.gather(1, idx), m_all.gather(1, idx),
                      h_all.gather(1, idx), f_all.gather(1, idx))

    # ------------------------------------------------------------------ #
    # exact execution filter (Eq. 18 / 19)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def exact_projection(self, a: Tensor, rows: RowSet, fixed: Optional[Tensor] = None,
                         iterations: Optional[int] = None) -> tuple[Tensor, dict]:
        """Q_CBF(x, a): Euclidean projection of ``a`` onto {C a >= c, |a| <= 1}.

        ``fixed`` ([da] bool) marks action coordinates held at their input
        value (measured human block, Eq. 19); their contribution is moved into
        the right-hand side and they are excluded from the optimisation.
        Batched Hildreth dual iteration over rows G x <= g with
        G = [-C; I; -I], g = [-c; 1; 1].  Returns the projected action and a
        dict with dual values, active mask, min residual, feasibility flag.
        """
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
        G = torch.cat((-C, eye, -eye), dim=1)                   # [B, M, d], M = K+2d
        g = torch.cat((-c, torch.ones(B, 2 * d, device=dev, dtype=dt)), dim=1)
        mk = torch.cat((m, torch.ones(B, 2 * d, device=dev, dtype=dt)), dim=1)
        # Hildreth in dual coordinates: with x = a - G^T lam the row residual is
        # G_k x - g_k = b_k - H_k lam, H = G G^T, b = G a - g.  The sequential
        # sweep runs on small CPU tensors (one GPU round trip per call instead of
        # ~10^4 tiny kernels); the iterates are those of the primal sweep.
        H = torch.bmm(G, G.transpose(1, 2))
        b = torch.einsum("bmd,bd->bm", G, a_free) - g
        Hc, bc, mc = (t.detach().to("cpu", torch.float64) for t in (H, b, mk))
        diag = torch.diagonal(Hc, dim1=1, dim2=2).clamp_min(1e-12)
        lam_c = torch.zeros(B, K + 2 * d, dtype=torch.float64)
        Ht = Hc.transpose(0, 1).contiguous()                    # [M, B, M]: row k for every batch
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
    """Divide every present row and its right-hand side by ||C_j||_2 (Sec. III-C)."""
    nrm = rows.C.norm(dim=-1)
    ok = (nrm > 1e-9) & (rows.mask > 0)
    scale = torch.where(ok, 1.0 / nrm.clamp_min(1e-9), torch.zeros_like(nrm))
    return RowSet(rows.C * scale.unsqueeze(-1), rows.c * scale, ok.to(rows.mask.dtype),
                  rows.h, rows.family)
