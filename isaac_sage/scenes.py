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


def build() -> dict:
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
    return {TILT_GATE_NAME: sc}


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
