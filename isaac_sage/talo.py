from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def _grads(objective: Tensor, params: Sequence[Tensor], create_graph: bool) -> list[Tensor]:
    grads = torch.autograd.grad(objective, params, create_graph=create_graph, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


def _dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> Tensor:
    return sum((x * y).sum() for x, y in zip(a, b))


def talo_update_field(agent_objectives: Sequence[Tensor], team_objective: Tensor, params: Sequence[Tensor],
                      sigma: float, eps: float, enabled: bool = True) -> tuple[list[Tensor], dict]:
    need_graph = bool(enabled)
    agent_grads = _grads(sum(agent_objectives), params, create_graph=need_graph)
    team_grads = _grads(team_objective, params, create_graph=need_graph)
    gap = 0.5 * sum(((agent_g - team_g) ** 2).sum() for agent_g, team_g in zip(agent_grads, team_grads))

    with torch.no_grad():
        agent_norm = torch.sqrt(sum((agent_g.detach() ** 2).sum() for agent_g in agent_grads))
        team_norm = torch.sqrt(sum((team_g.detach() ** 2).sum() for team_g in team_grads))
        cos = _dot([g.detach() for g in agent_grads], [g.detach() for g in team_grads]) / (
            agent_norm * team_norm + 1e-12
        )

    stats = {"W_gap": float(gap.detach()), "cos_align": float(cos),
             "g_ind_norm": float(agent_norm), "g_team_norm": float(team_norm),
             "talo_lambda": 0.0, "talo_active": 0.0, "delta_gap": 0.0}

    if not enabled:
        return [g.detach() for g in team_grads], stats
    if float(gap.detach()) <= 0.0:
        return [g.detach() for g in agent_grads], stats

    gap_grads = _grads(gap, params, create_graph=False)
    with torch.no_grad():
        delta = _dot(gap_grads, [g.detach() for g in agent_grads]) + sigma * gap.detach()
        gap_grad_sq = sum((gap_g * gap_g).sum() for gap_g in gap_grads)
        lam = torch.clamp(delta, min=0.0) / (gap_grad_sq + eps)
        field = [agent_g.detach() - lam * gap_g for agent_g, gap_g in zip(agent_grads, gap_grads)]
    stats.update({"talo_lambda": float(lam), "talo_active": float(delta > 0), "delta_gap": float(delta)})
    return field, stats
