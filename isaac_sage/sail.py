"""Shield-Annealed Internalization Layer (paper Sec. III-C, Eq. 27).

    a_bar = Pi_kappa(x, a_ST)
          = argmin_a  1/2 ||a - a_ST||^2 + kappa/2 * sum_j [c_j(x) - C_j(x) a]_+^2

The squared-hinge objective is strongly convex and piecewise quadratic.  A
fixed number ``K_Pi`` of unrolled damped Newton steps approximates the
minimiser and autograd is carried through the iterations, so the training
proposal centre keeps a non-zero constraint-normal sensitivity
(Eq. 29: d a_bar / d a_ST = (I + kappa C_act^T C_act)^-1) where the exact
projection (Eq. 28) would annihilate it.  Rows are normalised by ||C_j|| before
entering the penalty (Sec. III-C), which leaves each feasible half-space
unchanged.

SAIL only shapes the *training* proposal centre.  Every executed action still
comes from the exact QP in ``safety.exact_projection`` (Eq. 32).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .safety import RowSet, normalize_rows


def sail_penalty_map(a_st: Tensor, rows: RowSet, kappa: float,
                     newton_steps: int = 8, damping: float = 1.0) -> Tensor:
    """Differentiable penalty map Pi_kappa (Eq. 27), batched.

    a_st: [B, da] stacked hard-forward ST nominal action
    rows: affine safety rows for the same states (padded rows masked)
    kappa: penalty coefficient kappa_n of the current iteration
    newton_steps: K_Pi unrolled damped Newton iterations
    damping: step length in (0, 1]; 1.0 solves the current quadratic piece
             exactly, smaller values damp active-set switching
    """
    rn = normalize_rows(rows)
    Cn, cn, m = rn.C, rn.c, rn.mask
    u = a_st
    eye = torch.eye(a_st.shape[-1], dtype=a_st.dtype, device=a_st.device)
    for _ in range(newton_steps):
        s = cn - torch.einsum("bkd,bd->bk", Cn, u)            # > 0 : row violated
        act = F.relu(s) * m                                    # [c_j - C_j u]_+
        grad = (u - a_st) - kappa * torch.einsum("bk,bkd->bd", act, Cn)
        w = (s > 0).to(a_st.dtype) * m
        hess = eye + kappa * torch.einsum("bk,bkd,bke->bde", w, Cn, Cn)
        step = torch.linalg.solve(hess, grad.unsqueeze(-1)).squeeze(-1)
        u = u - damping * step
    return u


def sail_jacobian_normal_gain(kappa: float) -> float:
    """Closed-form constraint-normal gain 1 / (1 + kappa) for one isolated
    normalised active row (Sec. III-C); used by the smoke test to check the
    unrolled map against Eq. 29."""
    return 1.0 / (1.0 + kappa)
