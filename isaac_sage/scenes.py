from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

BOARD_LENGTH, BOARD_WIDTH = 2.0, 1.0
ROBOT_OFFSET, ROBOT_R = 0.20, 0.20
WALL_H = 0.71
SEC_PER_M = 18.0


def box(x0: float, x1: float, y0: float, y1: float, h: float = WALL_H):
    return ((round(x1 - x0, 4), round(y1 - y0, 4), h),
            (round((x0 + x1) / 2, 4), round((y0 + y1) / 2, 4), h / 2))


def resample(vertices, spacing: float, keep_vertices: bool) -> list[tuple[float, float]]:
    v = np.asarray(vertices, dtype=float)
    if keep_vertices:
        out = []
        for a, b in zip(v[:-1], v[1:]):
            n = max(1, int(round(np.linalg.norm(b - a) / spacing)))
            out += [tuple(a + (k / n) * (b - a)) for k in range(n)]
        out.append(tuple(v[-1]))
    else:
        seg = np.linalg.norm(np.diff(v, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        n = max(1, int(round(s[-1] / spacing)))
        out = [tuple(np.array([np.interp(t, s, v[:, 0]), np.interp(t, s, v[:, 1])]))
               for t in np.linspace(0.0, s[-1], n + 1)]
    return [(round(float(x), 3), round(float(y), 3)) for x, y in out]


def _robot_centres(cx, cy, h):
    dx, dy, px, py = math.cos(h), math.sin(h), -math.sin(h), math.cos(h)
    ax, ay = BOARD_LENGTH / 2 + ROBOT_OFFSET, BOARD_WIDTH / 2 + ROBOT_OFFSET
    return [(cx + ax * dx, cy + ax * dy), (cx - ax * dx, cy - ax * dy), (cx + ay * px, cy + ay * py)]


def _board_corners(cx, cy, h):
    dx, dy, px, py = math.cos(h), math.sin(h), -math.sin(h), math.cos(h)
    hl, hw = BOARD_LENGTH / 2, BOARD_WIDTH / 2
    return [(cx + sl * hl * dx + sw * hw * px, cy + sl * hl * dy + sw * hw * py)
            for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def footprint_hits(cx, cy, h, rects) -> bool:
    poly = np.asarray(_board_corners(cx, cy, h))
    for x0, x1, y0, y1 in rects:
        rect = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        sep = False
        for ax in (np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                   np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])):
            a, b = poly @ ax, rect @ ax
            if a.max() < b.min() or b.max() < a.min():
                sep = True
                break
        if not sep:
            return True
    for rx, ry in _robot_centres(cx, cy, h):
        for x0, x1, y0, y1 in rects:
            qx, qy = min(max(rx, x0), x1), min(max(ry, y0), y1)
            if (rx - qx) ** 2 + (ry - qy) ** 2 <= ROBOT_R ** 2:
                return True
    return False


def footprint_clearance(cx, cy, h, rects) -> float:
    if footprint_hits(cx, cy, h, rects):
        return -1.0
    corners = _board_corners(cx, cy, h)
    dx, dy = math.cos(h), math.sin(h)
    best = math.inf
    for x0, x1, y0, y1 in rects:
        for px, py in corners:
            qx, qy = min(max(px, x0), x1), min(max(py, y0), y1)
            best = min(best, math.hypot(px - qx, py - qy))
        for qx, qy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            u, v = (qx - cx) * dx + (qy - cy) * dy, -(qx - cx) * dy + (qy - cy) * dx
            eu, ev = max(abs(u) - BOARD_LENGTH / 2, 0.0), max(abs(v) - BOARD_WIDTH / 2, 0.0)
            best = min(best, math.hypot(eu, ev))
        for rx, ry in _robot_centres(cx, cy, h):
            qx, qy = min(max(rx, x0), x1), min(max(ry, y0), y1)
            best = min(best, math.hypot(rx - qx, ry - qy) - ROBOT_R)
    return best


def hinted_yaw(x, y, hints, default: float = 0.0) -> float:
    for x0, x1, y0, y1, yaw in hints:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return yaw
    return default


def add_start_curriculum(pool: list[dict], rects, hints, margin: float = 0.10) -> None:
    for e in pool:
        yaws, ok = [], []
        for x, y in e["path"]:
            h = hinted_yaw(x, y, hints)
            yaws.append(round(h, 4))
            ok.append(bool(footprint_clearance(x, y, h, rects) >= margin + 0.01))
        e["curriculum_yaw"], e["curriculum_ok"] = yaws, ok


def sweep(points, rects, step: float = 0.05):
    p = np.asarray(points, dtype=float)
    dense = []
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        dense += [a + (k / n) * (b - a) for k in range(n)]
    dense.append(p[-1])
    dense = np.asarray(dense)
    hits = []
    for i in range(len(dense)):
        d = dense[min(i + 2, len(dense) - 1)] - dense[max(i - 2, 0)]
        h = math.atan2(d[1], d[0]) + math.pi
        if footprint_hits(dense[i, 0], dense[i, 1], h, rects):
            hits.append((round(float(dense[i, 0]), 2), round(float(dense[i, 1]), 2)))
    return len(hits) / len(dense), hits


def path_length(points) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())


def crossing_width(phi: float) -> float:
    h = math.pi / 2 - phi
    xs = [c[0] for c in _board_corners(0.0, 0.0, h)]
    lo, hi = min(xs), max(xs)
    for rx, _ in _robot_centres(0.0, 0.0, h):
        lo, hi = min(lo, rx - ROBOT_R), max(hi, rx + ROBOT_R)
    return hi - lo


def max_carry_angle(opening: float) -> float:
    return max(d for d in np.arange(0.0, 90.01, 0.5) if crossing_width(math.radians(d)) <= opening)


def resample_fixed(vertices, n_points: int) -> list[tuple[float, float]]:
    v = np.asarray(vertices, dtype=float)
    legs = np.linalg.norm(np.diff(v, axis=0), axis=1)
    n_seg = n_points - 1
    if n_seg < len(legs):
        raise ValueError("fewer points than polyline corners")
    alloc = np.maximum(1, np.floor(legs / legs.sum() * n_seg)).astype(int)
    while alloc.sum() < n_seg:
        alloc[np.argmax(legs / alloc)] += 1
    while alloc.sum() > n_seg:
        alloc[np.argmin(np.where(alloc > 1, legs / alloc, np.inf))] -= 1
    out = []
    for a, b, n in zip(v[:-1], v[1:], alloc):
        out += [tuple(a + (k / n) * (b - a)) for k in range(n)]
    out.append(tuple(v[-1]))
    return [(round(float(x), 3), round(float(y), 3)) for x, y in out]


def spawn_pool(rects, start, goal, via, start_jitter, goal_jitter, n_points: int,
               size: int = 512, seed: int = 0, max_draws: int = 200_000) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(max_draws):
        if len(pool) >= size:
            break
        s = np.asarray(start, float) + rng.uniform(-1, 1, 3) * np.asarray(start_jitter, float)
        g = np.asarray(goal, float) + rng.uniform(-1, 1, 2) * np.asarray(goal_jitter, float)
        if footprint_hits(s[0], s[1], s[2], rects):
            continue
        path = resample_fixed([tuple(s[:2])] + list(via) + [tuple(g)], n_points)
        if sweep(path, rects)[0] > 0.0:
            continue
        pool.append({"pose": [round(float(s[0]), 3), round(float(s[1]), 3), round(float(s[2]), 4)],
                     "path": [list(p) for p in path]})
    if len(pool) < size:
        raise RuntimeError(f"only {len(pool)} of {size} spawn draws are collision-free")
    return pool


def make_scene(name, rects, vertices, sparse_spacing: float | None, note: str = "",
               min_budget_s: float = 300.0, wall_h: float = WALL_H,
               reference: list | None = None) -> dict:
    if reference is None:
        reference = resample(vertices, 0.5, keep_vertices=True)
    visible = reference if sparse_spacing is None else resample(vertices, sparse_spacing, keep_vertices=False)
    xs = [v for r in rects for v in r[:2]] + [p[0] for p in reference]
    ys = [v for r in rects for v in r[2:]] + [p[1] for p in reference]
    length = path_length(reference)
    return {
        "name": name,
        "note": note,
        "aabbs": [list(r) for r in rects],
        "obstacle_specs": [box(*r, h=wall_h) for r in rects],
        "reference_path": reference,
        "path_waypoints": visible,
        "start_xy": visible[0],
        "path_length_m": round(length, 2),
        "episode_length_s": float(max(min_budget_s, math.ceil(length * SEC_PER_M / 30) * 30)),
        "env_spacing": float(math.ceil(max(max(xs) - min(xs), max(ys) - min(ys)) + 6)),
    }


TILT_GATE_NAME = "tilt_gate_2m2"
TILT_GATE_OPENING = 2.2
TILT_GATE_WALL_H = 1.2
TILT_GATE_START, TILT_GATE_GOAL = (0.0, 0.0, 0.0), (0.0, 5.6)
TILT_GATE_VIA = [(0.0, 1.4), (0.0, 3.5)]
TILT_GATE_JITTER = ((0.933, 0.35, 0.25), (0.933, 0.35))
TILT_GATE_POINTS = 14
TILT_GATE_RECTS = [
    (-4.95, -4.7, -2.15, 7.75),
    (4.7, 4.95, -2.15, 7.75),
    (-4.7, 4.7, -2.15, -1.9),
    (-4.7, 4.7, 7.5, 7.75),
    (-4.7, -TILT_GATE_OPENING / 2, 1.4, 3.5),
    (TILT_GATE_OPENING / 2, 4.7, 1.4, 3.5),
]
TILT_GATE_VERTICES = [TILT_GATE_START[:2], *TILT_GATE_VIA, TILT_GATE_GOAL]
TILT_GATE_HINT_YAW = 0.97
TILT_GATE_YAW_HINTS = [
    (-4.7, 4.7, -1.9, 1.05, TILT_GATE_HINT_YAW),
    (-TILT_GATE_OPENING / 2, TILT_GATE_OPENING / 2, 1.05, 5.02, TILT_GATE_HINT_YAW),
    (-4.7, 4.7, 5.02, 7.5, 0.0),
]


MAZE_NAME = "maze"
MAZE_SPECS = [
    ((5.05, 0.25, 0.71), (1.55, -0.695, 0.355)),
    ((7.7, 0.25, 0.71), (0.225, -3.945, 0.355)),
    ((0.25, 1.675, 0.71), (-1.1, 0.0175, 0.355)),
    ((0.25, 8.075, 0.71), (-3.75, -0.0325, 0.355)),
    ((5.05, 0.25, 0.71), (-1.1, 3.88, 0.355)),
    ((0.25, 3.15, 0.71), (1.55, 3.505, 0.355)),
    ((2.4, 0.25, 0.71), (2.875, 4.955, 0.355)),
    ((0.25, 9.15, 0.71), (4.2, 0.505, 0.355)),
]
MAZE_PATH = [
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
MAZE_RECTS = [(c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2) for s, c in MAZE_SPECS]
MAZE_START, MAZE_GOAL = (2.0, -2.32, 0.0), (2.65, 3.0)
MAZE_VIA = [(-1.65, -2.32), (-2.14, -1.82), (-2.14, 1.62), (-1.6, 2.23), (-0.7, 2.23), (1.2, 0.97),
            (2.2, 0.97), (2.65, 1.67)]
MAZE_JITTER = ((0.5, 0.4, 0.2), (0.3, 0.3))


def add_path_curriculum(pool: list[dict], rects, margin: float = 0.10) -> None:
    for entry in pool:
        path = np.asarray(entry["path"], dtype=float)
        yaws, ok = [], []
        for k in range(len(path)):
            d = path[min(k + 1, len(path) - 1)] - path[max(k - 1, 0)]
            yaw = math.remainder(math.atan2(d[1], d[0]) + math.pi, 2 * math.pi)
            yaws.append(round(yaw, 4))
            ok.append(bool(footprint_clearance(path[k, 0], path[k, 1], yaw, rects) >= margin + 0.01))
        entry["curriculum_yaw"], entry["curriculum_ok"] = yaws, ok


def build_tilt_gate() -> dict:
    opening = TILT_GATE_OPENING
    sc = make_scene(
        TILT_GATE_NAME, TILT_GATE_RECTS, TILT_GATE_VERTICES, None,
        note=(f"{opening} m deep gate entered broadside; long axis must be within "
              f"{max_carry_angle(opening):.1f} deg of travel to fit; randomised spawn; "
              f"{TILT_GATE_WALL_H} m walls"),
        min_budget_s=150.0, wall_h=TILT_GATE_WALL_H,
        reference=resample_fixed(TILT_GATE_VERTICES, TILT_GATE_POINTS))
    sc["spawn_pool"] = spawn_pool(TILT_GATE_RECTS, TILT_GATE_START, TILT_GATE_GOAL, TILT_GATE_VIA,
                                  *TILT_GATE_JITTER, n_points=TILT_GATE_POINTS)
    sc["yaw_hints"] = [list(h) for h in TILT_GATE_YAW_HINTS]
    sc["goal_yaw"], sc["yaw_symmetry"] = 0.0, math.pi
    add_start_curriculum(sc["spawn_pool"], TILT_GATE_RECTS, TILT_GATE_YAW_HINTS)
    return sc


def build_maze() -> dict:
    sc = {
        "name": MAZE_NAME,
        "note": "the task's default maze; randomised spawn (use --fixed_spawn for the default start)",
        "aabbs": [list(r) for r in MAZE_RECTS],
        "obstacle_specs": [[list(s), list(c)] for s, c in MAZE_SPECS],
        "reference_path": [list(p) for p in MAZE_PATH],
        "path_waypoints": [list(p) for p in MAZE_PATH],
        "start_xy": list(MAZE_PATH[0]),
        "path_length_m": round(path_length(MAZE_PATH), 2),
        "episode_length_s": 300.0,
        "env_spacing": 10.0,
    }
    sc["spawn_pool"] = spawn_pool(MAZE_RECTS, MAZE_START, MAZE_GOAL, MAZE_VIA, *MAZE_JITTER,
                                  n_points=len(MAZE_PATH))
    add_path_curriculum(sc["spawn_pool"], MAZE_RECTS)
    sc["yaw_hints"] = []
    sc["goal_yaw"], sc["yaw_symmetry"] = -math.pi / 2, math.pi
    return sc


COURSE_WALL_H = TILT_GATE_WALL_H
ARENA_T = 0.25
WAYPOINT_SPACING = 0.43


def arena(x0: float, x1: float, y0: float, y1: float) -> list[tuple]:
    t = ARENA_T
    return [(x0 - t, x0, y0 - t, y1 + t), (x1, x1 + t, y0 - t, y1 + t),
            (x0, x1, y0 - t, y0), (x0, x1, y1, y1 + t)]


def footprint_shift(h: float) -> tuple[float, float]:
    xs, ys = map(list, zip(*_board_corners(0.0, 0.0, h)))
    for rx, ry in _robot_centres(0.0, 0.0, h):
        xs += [rx - ROBOT_R, rx + ROBOT_R]
        ys += [ry - ROBOT_R, ry + ROBOT_R]
    return round(-(min(xs) + max(xs)) / 2, 3), round(-(min(ys) + max(ys)) / 2, 3)


def course_points(vertices) -> int:
    return int(round(path_length(vertices) / WAYPOINT_SPACING)) + 1


LENGTHWISE_DY = footprint_shift(0.0)[1]


LENGTHWISE_DY_BACK = footprint_shift(math.pi)[1]
LENGTHWISE_DX_UP = footprint_shift(-math.pi / 2)[0]


def arc(cx: float, cy: float, r: float, t0: float, t1: float, k: int) -> list[tuple[float, float]]:
    return [(round(cx + r * math.cos(t), 3), round(cy + r * math.sin(t), 3)) for t in np.linspace(t0, t1, k)]


def arc_band_hints(cx: float, cy: float, r: float, t0: float, t1: float, n_bands: int = 4,
                   pad: float = 0.15) -> list[tuple]:
    out = []
    for k in range(n_bands):
        a0, a1 = t0 + k * (t1 - t0) / n_bands, t0 + (k + 1) * (t1 - t0) / n_bands
        ts = np.linspace(a0, a1, 9)
        xs, ys = cx + r * np.cos(ts), cy + r * np.sin(ts)
        out.append((round(xs.min() - pad, 3), round(xs.max() + pad, 3), round(ys.min() - pad, 3),
                    round(ys.max() + pad, 3), round(math.remainder((a0 + a1) / 2 + math.pi / 2, 2 * math.pi), 4)))
    return out


S_SHAPED_LANE, S_SHAPED_GATE, S_SHAPED_TURN_GAP = 3.0, 2.0, 2.2
_SS_H, _SS_TIP, _SS_XR, _SS_XG = 0.2335, -2.5, 5.6, 0.3
_SS_LANE_C = -(_SS_H + S_SHAPED_LANE / 2)
S_SHAPED_RECTS = arena(_SS_TIP - S_SHAPED_TURN_GAP, _SS_XR, -S_SHAPED_LANE - _SS_H, S_SHAPED_LANE + _SS_H) + [
    (_SS_TIP, _SS_XR, -_SS_H, _SS_H),
    (_SS_XG, _SS_XG + 0.467, -S_SHAPED_LANE - _SS_H, _SS_LANE_C - S_SHAPED_GATE / 2),
    (_SS_XG, _SS_XG + 0.467, _SS_LANE_C + S_SHAPED_GATE / 2, -_SS_H),
]
_SS_B0, _SS_B1 = _SS_LANE_C + LENGTHWISE_DY, -_SS_LANE_C + LENGTHWISE_DY_BACK
_SS_TC, _SS_TY, _SS_TR = _SS_TIP + 0.75, (_SS_B0 + _SS_B1) / 2, (_SS_B1 - _SS_B0) / 2
S_SHAPED_START, S_SHAPED_GOAL = (3.5, _SS_B0, 0.0), (3.5, _SS_B1)
S_SHAPED_VIA = ([(_SS_XG + 1.467, _SS_B0), (_SS_XG + 0.467, _SS_B0), (_SS_XG, _SS_B0), (_SS_TC, _SS_B0)]
                + arc(_SS_TC, _SS_TY, _SS_TR, -math.pi / 2, -1.5 * math.pi, 17)[1:-1]
                + [(_SS_TC, _SS_B1)])
S_SHAPED_JITTER = ((0.5, 0.6, 0.25), (0.5, 0.6))
S_SHAPED_YAW_HINTS = arc_band_hints(_SS_TC, _SS_TY, _SS_TR, -math.pi / 2, -1.5 * math.pi, n_bands=6) + [
    (_SS_TIP - S_SHAPED_TURN_GAP, _SS_XR, -S_SHAPED_LANE - _SS_H, -_SS_H, 0.0),
    (_SS_TIP - S_SHAPED_TURN_GAP, _SS_XR, _SS_H, S_SHAPED_LANE + _SS_H, math.pi),
]

U_SHAPED_WIDTH = 2.5
_US_X0, _US_X1, _US_Y0, _US_Y1 = -2.5, 5.5, -2.5, 2.5
U_SHAPED_RECTS = arena(_US_X0 - U_SHAPED_WIDTH, _US_X1, _US_Y0 - U_SHAPED_WIDTH, _US_Y1 + U_SHAPED_WIDTH) + [
    (_US_X0, _US_X1, _US_Y0, _US_Y1),
]
_US_BL = _US_Y0 - U_SHAPED_WIDTH / 2 + LENGTHWISE_DY
_US_BS = _US_X0 - U_SHAPED_WIDTH / 2 + LENGTHWISE_DX_UP
_US_BU = _US_Y1 + U_SHAPED_WIDTH / 2 + LENGTHWISE_DY_BACK
U_SHAPED_TURN_R = 1.3
U_SHAPED_START, U_SHAPED_GOAL = (3.5, _US_BL, 0.0), (3.5, _US_BU)
U_SHAPED_VIA = ([(_US_X0 + 1.0, _US_BL)] + arc(_US_X0, _US_Y0, U_SHAPED_TURN_R, -math.pi / 2, -math.pi, 13)
                + [(_US_BS, _US_Y0 + 1.0), (_US_BS, _US_Y1 - 1.0)]
                + arc(_US_X0, _US_Y1, U_SHAPED_TURN_R, math.pi, math.pi / 2, 13) + [(_US_X0 + 1.0, _US_BU)])
U_SHAPED_JITTER = ((0.5, 0.4, 0.2), (0.5, 0.4))
U_SHAPED_YAW_HINTS = (
    arc_band_hints(_US_X0, _US_Y0, U_SHAPED_TURN_R, -math.pi / 2, -math.pi)
    + arc_band_hints(_US_X0, _US_Y1, U_SHAPED_TURN_R, math.pi, math.pi / 2)
    + [(_US_X0, _US_X1, _US_Y0 - U_SHAPED_WIDTH, _US_Y0, 0.0),
       (_US_X0 - U_SHAPED_WIDTH, _US_X0, _US_Y0, _US_Y1, -math.pi / 2),
       (_US_X0, _US_X1, _US_Y1, _US_Y1 + U_SHAPED_WIDTH, math.pi)]
)

L_CORRIDOR_WIDTH = 2.2
_LC_C, _LC_H, _LC_T = -2.8, L_CORRIDOR_WIDTH / 2, 0.467
L_CORRIDOR_RECTS = [
    (_LC_C + _LC_H, 4.667, _LC_C + _LC_H, _LC_C + _LC_H + _LC_T),
    (_LC_C + _LC_H, _LC_C + _LC_H + _LC_T, _LC_C + _LC_H, 4.667),
    (_LC_C - _LC_H - _LC_T, 4.667, _LC_C - _LC_H - _LC_T, _LC_C - _LC_H),
    (_LC_C - _LC_H - _LC_T, _LC_C - _LC_H, _LC_C - _LC_H, 4.667),
    (4.667, 4.667 + _LC_T, _LC_C - _LC_H - _LC_T, _LC_C + _LC_H + _LC_T),
    (_LC_C - _LC_H - _LC_T, _LC_C + _LC_H + _LC_T, 4.667, 4.667 + _LC_T),
]
_LC_B = _LC_C + LENGTHWISE_DY
_LC_IN = round(_LC_C + _LC_H, 3)
L_CORRIDOR_TURN_R = 1.15
L_CORRIDOR_START, L_CORRIDOR_GOAL = (1.867, _LC_B, 0.0), (_LC_B, 1.867)
L_CORRIDOR_VIA = ([(_LC_IN + 1.0, _LC_B)] + arc(_LC_IN, _LC_IN, L_CORRIDOR_TURN_R, -math.pi / 2, -math.pi, 13)
                  + [(_LC_B, _LC_IN + 1.0)])
L_CORRIDOR_JITTER = ((1.05, 0.233, 0.2), (0.233, 1.05))
L_CORRIDOR_YAW_HINTS = arc_band_hints(_LC_IN, _LC_IN, L_CORRIDOR_TURN_R, -math.pi / 2, -math.pi) + [
    (_LC_IN, 4.667, _LC_C - _LC_H, _LC_IN, 0.0),
    (_LC_C - _LC_H, _LC_IN, _LC_IN, 4.667, -math.pi / 2),
]


def course(name, rects, start, goal, via, jitter, hints, goal_yaw: float, note: str,
           sparse_spacing: float | None = None) -> dict:
    vertices = [start[:2], *via, goal]
    n_points = course_points(vertices)
    sc = make_scene(name, rects, vertices, sparse_spacing, note=note, min_budget_s=150.0,
                    wall_h=COURSE_WALL_H, reference=resample_fixed(vertices, n_points))
    sc["spawn_pool"] = spawn_pool(rects, start, goal, via, *jitter, n_points=n_points)
    sc["yaw_hints"] = [list(h) for h in hints]
    sc["goal_yaw"], sc["yaw_symmetry"] = goal_yaw, math.pi
    add_start_curriculum(sc["spawn_pool"], rects, hints)
    return sc


def build_s_shaped() -> dict:
    return course(
        "s_shaped", S_SHAPED_RECTS, S_SHAPED_START, S_SHAPED_GOAL,
        S_SHAPED_VIA, S_SHAPED_JITTER, S_SHAPED_YAW_HINTS, 0.0,
        f"{S_SHAPED_LANE} m lanes, {S_SHAPED_GATE} m gate, 180 deg turn through a {S_SHAPED_TURN_GAP} m "
        f"gap; carried lengthwise along -X then +X; randomised spawn")


def build_u_shaped() -> dict:
    return course(
        "u_shaped", U_SHAPED_RECTS, U_SHAPED_START, U_SHAPED_GOAL,
        U_SHAPED_VIA, U_SHAPED_JITTER, U_SHAPED_YAW_HINTS, 0.0,
        f"{U_SHAPED_WIDTH} m corridor round three sides of an 8.0 x 5.0 m block; carried lengthwise along -X, "
        "then +Y, then +X; randomised spawn")


def build_l_corridor() -> dict:
    return course(
        "l_corridor", L_CORRIDOR_RECTS, L_CORRIDOR_START, L_CORRIDOR_GOAL,
        L_CORRIDOR_VIA, L_CORRIDOR_JITTER, L_CORRIDOR_YAW_HINTS, -math.pi / 2,
        f"closed {L_CORRIDOR_WIDTH} m corridor, one 90 deg corner; carried lengthwise along -X then +Y; "
        "randomised spawn")


def build() -> dict:
    return {TILT_GATE_NAME: build_tilt_gate(), MAZE_NAME: build_maze(), "s_shaped": build_s_shaped(),
            "u_shaped": build_u_shaped(), "l_corridor": build_l_corridor()}


def _render(scene: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(10, 7))
    for x0, x1, y0, y1 in scene["aabbs"]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="firebrick", alpha=0.6))
    ref = np.asarray(scene["reference_path"])
    vis = np.asarray(scene["path_waypoints"])
    ax.plot(ref[:, 0], ref[:, 1], "g-", lw=1, alpha=0.8, label="reference path (collision-free)")
    ax.plot(vis[:, 0], vis[:, 1], "b--o", ms=5, lw=1, label="visible waypoints")
    for i in range(0, len(ref), 3):
        j0, j1 = max(i - 1, 0), min(i + 1, len(ref) - 1)
        d = ref[j1] - ref[j0]
        h = math.atan2(d[1], d[0]) + math.pi
        ax.add_patch(Polygon(_board_corners(ref[i, 0], ref[i, 1], h), closed=True,
                             facecolor="limegreen", alpha=0.08, edgecolor="green", lw=0.3))
        for rx, ry in _robot_centres(ref[i, 0], ref[i, 1], h):
            ax.add_patch(Circle((rx, ry), ROBOT_R, facecolor="green", alpha=0.08))
    for k, entry in enumerate(scene.get("spawn_pool", [])[:24]):
        pth, (px, py, pyaw) = np.asarray(entry["path"]), entry["pose"]
        ax.plot(pth[:, 0], pth[:, 1], "-", color="purple", lw=0.6, alpha=0.4,
                label="randomised spawns / paths (sample)" if k == 0 else None)
        ax.add_patch(Polygon(_board_corners(px, py, pyaw), closed=True, fill=False,
                             edgecolor="purple", lw=0.4, alpha=0.4))
    sx, sy = scene["start_xy"]
    ax.add_patch(Polygon(_board_corners(sx, sy, 0.0), closed=True, facecolor="orange", alpha=0.35,
                         edgecolor="darkorange", lw=1, label="spawn pose"))
    for rx, ry in _robot_centres(sx, sy, 0.0):
        ax.add_patch(Circle((rx, ry), ROBOT_R, facecolor="darkorange", alpha=0.35))
    _, hits = sweep(scene["path_waypoints"], scene["aabbs"])
    if hits:
        hx, hy = zip(*hits)
        ax.plot(hx, hy, "rx", ms=3, alpha=0.5, label="straight-line carry overlaps a wall")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{scene['name']}  ({scene['path_length_m']} m, {scene['episode_length_s']:.0f} s)")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "scenes"))
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, sc in build().items():
        ref_frac, ref_hits = sweep(sc["reference_path"], sc["aabbs"])
        ok &= ref_frac == 0.0
        print(f"[{'OK ' if ref_frac == 0 else 'HIT'}] {name}: reference {sc['path_length_m']} m, "
              f"budget {sc['episode_length_s']:.0f} s, {len(sc['path_waypoints'])} visible waypoints, "
              f"{len(sc['spawn_pool'])} spawn draws; "
              f"reference overlaps walls {ref_frac:.1%}{' at ' + str(ref_hits[:5]) if ref_hits else ''}")
        (out / f"{name}.json").write_text(json.dumps(sc, indent=1))
        if args.png:
            _render(sc, out / f"{name}.png")
    print("reference paths collision-free" if ok else "reference path clips a wall -- fix the geometry")


if __name__ == "__main__":
    main()
