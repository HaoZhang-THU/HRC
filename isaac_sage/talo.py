from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def _grads(objective: Tensor, params: Sequence[Tensor], create_graph: bool) -> list[Tensor]:
    gs = torch.autograd.grad(objective, params, create_graph=create_graph, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for g, p in zip(gs, params)]


def _dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> Tensor:
    return sum((x * y).sum() for x, y in zip(a, b))


def talo_update_field(L_ind: Sequence[Tensor], L_team: Tensor, params: Sequence[Tensor],
                      sigma: float, eps: float, enabled: bool = True) -> tuple[list[Tensor], dict]:
    need_graph = bool(enabled)
    g_ind = _grads(sum(L_ind), params, create_graph=need_graph)
    g_team = _grads(L_team, params, create_graph=need_graph)
    W = 0.5 * sum(((gi - gt) ** 2).sum() for gi, gt in zip(g_ind, g_team))

    with torch.no_grad():
        ni = torch.sqrt(sum((gi.detach() ** 2).sum() for gi in g_ind))
        nt = torch.sqrt(sum((gt.detach() ** 2).sum() for gt in g_team))
        cos = _dot([g.detach() for g in g_ind], [g.detach() for g in g_team]) / (ni * nt + 1e-12)

    stats = {"W_gap": float(W.detach()), "cos_align": float(cos),
             "g_ind_norm": float(ni), "g_team_norm": float(nt),
             "talo_lambda": 0.0, "talo_active": 0.0, "delta_gap": 0.0}

    if not enabled:
        return [g.detach() for g in g_team], stats
    if float(W.detach()) <= 0.0:
        return [g.detach() for g in g_ind], stats

    gradW = _grads(W, params, create_graph=False)
    with torch.no_grad():
        delta = _dot(gradW, [g.detach() for g in g_ind]) + sigma * W.detach()
        hn2 = sum((h * h).sum() for h in gradW)
        lam = torch.clamp(delta, min=0.0) / (hn2 + eps)
        field = [gi.detach() - lam * h for gi, h in zip(g_ind, gradW)]
    stats.update({"talo_lambda": float(lam), "talo_active": float(delta > 0), "delta_gap": float(delta)})
    return field, stats
