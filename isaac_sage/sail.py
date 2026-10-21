from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .safety import RowSet, normalize_rows


def sail_penalty_map(a_st: Tensor, rows: RowSet, kappa: float,
                     newton_steps: int = 8, damping: float = 1.0) -> Tensor:
    rn = normalize_rows(rows)
    Cn, cn, m = rn.C, rn.c, rn.mask
    u = a_st
    eye = torch.eye(a_st.shape[-1], dtype=a_st.dtype, device=a_st.device)
    for _ in range(newton_steps):
        s = cn - torch.einsum("bkd,bd->bk", Cn, u)
        act = F.relu(s) * m
        grad = (u - a_st) - kappa * torch.einsum("bk,bkd->bd", act, Cn)
        w = (s > 0).to(a_st.dtype) * m
        hess = eye + kappa * torch.einsum("bk,bkd,bke->bde", w, Cn, Cn)
        step = torch.linalg.solve(hess, grad.unsqueeze(-1)).squeeze(-1)
        u = u - damping * step
    return u


def sail_jacobian_normal_gain(kappa: float) -> float:
    return 1.0 / (1.0 + kappa)
