from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .safety import RowSet, normalize_rows


def sail_penalty_map(actions: Tensor, rows: RowSet, kappa: float,
                     newton_steps: int = 8, damping: float = 1.0) -> Tensor:
    norm_rows = normalize_rows(rows)
    coef, bound, mask = norm_rows.coef, norm_rows.bound, norm_rows.mask
    point = actions
    eye = torch.eye(actions.shape[-1], dtype=actions.dtype, device=actions.device)
    for _ in range(newton_steps):
        slack = bound - torch.einsum("bkd,bd->bk", coef, point)
        active = F.relu(slack) * mask
        grad = (point - actions) - kappa * torch.einsum("bk,bkd->bd", active, coef)
        weights = (slack > 0).to(actions.dtype) * mask
        hessian = eye + kappa * torch.einsum("bk,bkd,bke->bde", weights, coef, coef)
        step = torch.linalg.solve(hessian, grad.unsqueeze(-1)).squeeze(-1)
        point = point - damping * step
    return point


def sail_jacobian_normal_gain(kappa: float) -> float:
    return 1.0 / (1.0 + kappa)
