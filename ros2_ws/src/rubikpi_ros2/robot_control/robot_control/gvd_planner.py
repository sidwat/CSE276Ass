"""
Maximum-clearance planner: generalised Voronoi diagram on a clearance grid + Dijkstra (no ROS).

Representation. The arena is discretised into square cells. Each cell stores its clearance d
(distance to the nearest obstacle) and the label of that obstacle (the brushfire / exact
Euclidean distance transform; with five simple obstacles it is computed analytically from
World.clearance_and_label). The GVD is the set of cells where the nearest-obstacle label changes
between 4-neighbours: the points equally far from their two nearest obstacles. It is the locus
of locally maximal clearance (moving off it brings the robot closer to something) and a
deformation retract of the free space (every free point reaches it by climbing d), so it is a
complete roadmap that maximises the distance to the obstacles.

Search. Start and goal are retracted onto the GVD by steepest ascent of d; Dijkstra on the
8-connected GVD cells then picks the shortest route along the roadmap (which way round the box,
and no detours into dead-end corner spokes). The cell chain is simplified with RDP (epsilon);
clearance drops by at most epsilon.

plan_weighted() is the tunable alternative for the safety/time trade-off: grid A* minimising
integral (1 + alpha / d) ds; alpha = 0 is the shortest grid path, alpha -> inf tends to the GVD.
"""

from dataclasses import dataclass, field
import heapq
import math

import numpy as np

from robot_control.path_tools import drop_short_legs, rdp

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2), (0, -1, 1.0), (0, 1, 1.0),
              (1, -1, SQRT2), (1, 0, 1.0), (1, 1, SQRT2)]


@dataclass
class Grid:
    x0: float     # x of the first cell centre
    y0: float
    res: float
    nx: int
    ny: int
    D: np.ndarray       # (ny, nx) clearance
    L: np.ndarray       # (ny, nx) nearest-obstacle label

    def centre(self, cell):
        i, j = cell
        return self.x0 + j * self.res, self.y0 + i * self.res

    def cell_of(self, x, y):
        j = int(round((x - self.x0) / self.res))
        i = int(round((y - self.y0) / self.res))
        return min(max(i, 0), self.ny - 1), min(max(j, 0), self.nx - 1)

    def extent(self):
        h = self.res / 2.0
        return (self.x0 - h, self.x0 + (self.nx - 1) * self.res + h,
                self.y0 - h, self.y0 + (self.ny - 1) * self.res + h)


def build_grid(world, res):
    nx = int(math.floor((world.x_max - world.x_min) / res))
    ny = int(math.floor((world.y_max - world.y_min) / res))
    # cell centres, inset half a cell from the walls
    x0 = world.x_min + ((world.x_max - world.x_min) - (nx - 1) * res) / 2.0
    y0 = world.y_min + ((world.y_max - world.y_min) - (ny - 1) * res) / 2.0
    X, Y = np.meshgrid(x0 + np.arange(nx) * res, y0 + np.arange(ny) * res)
    D, L = world.clearance_and_label(X, Y)
    return Grid(x0, y0, res, nx, ny, D, L)


def gvd_mask(grid, free):
    """Cells where the nearest-obstacle label differs from a 4-neighbour's (both sides)."""
    L = grid.L
    mask = np.zeros_like(free)
    dx = L[:, 1:] != L[:, :-1]
    dy = L[1:, :] != L[:-1, :]
    mask[:, 1:] |= dx
    mask[:, :-1] |= dx
    mask[1:, :] |= dy
    mask[:-1, :] |= dy
    return mask & free


def retract(grid, cell, target, free):
    """Steepest ascent of the clearance from cell until a target cell (or a local maximum)."""
    chain = [cell]
    seen = {cell}
    while not target[chain[-1]]:
        i, j = chain[-1]
        best, best_d = None, grid.D[i, j]
        for di, dj, _ in NEIGHBOURS:
            n = (i + di, j + dj)
            if 0 <= n[0] < grid.ny and 0 <= n[1] < grid.nx and free[n] and n not in seen:
                if grid.D[n] > best_d + 1e-12:
                    best, best_d = n, grid.D[n]
        if best is None:
            break
        chain.append(best)
        seen.add(best)
    return chain


def grid_search(grid, allowed, start, goal, alpha=0.0, heuristic=True):
    """
    Dijkstra / A* over 8-connected allowed cells.

    Step cost = length * (1 + alpha / d), d the clearance of the cell stepped into. The
    Euclidean heuristic is admissible because the cost per metre is at least 1.
    Returns (cells, cost, expanded) or (None, inf, expanded).
    """
    ny, nx = allowed.shape
    res = grid.res
    D = grid.D
    gx, gy = goal[1], goal[0]
    g = {start: 0.0}
    parent = {start: None}
    heap = [(0.0, 0.0, start)]
    closed = np.zeros_like(allowed)
    expanded = 0
    while heap:
        _, gc, cell = heapq.heappop(heap)
        if closed[cell]:
            continue
        closed[cell] = True
        expanded += 1
        if cell == goal:
            seq = []
            while cell is not None:
                seq.append(cell)
                cell = parent[cell]
            return seq[::-1], gc, expanded
        i, j = cell
        for di, dj, step in NEIGHBOURS:
            n = (i + di, j + dj)
            if not (0 <= n[0] < ny and 0 <= n[1] < nx) or not allowed[n] or closed[n]:
                continue
            c = step * res
            if alpha > 0.0:
                c *= 1.0 + alpha / max(float(D[n]), 1e-3)
            ng = gc + c
            if ng < g.get(n, math.inf) - 1e-12:
                g[n] = ng
                parent[n] = cell
                h = math.hypot(n[1] - gx, n[0] - gy) * res if heuristic else 0.0
                heapq.heappush(heap, (ng + h, ng, n))
    return None, math.inf, expanded


@dataclass
class SafePlan:
    path: list                 # simplified (x, y) points, start .. goal
    raw_path: list             # cell-centre chain (with exact start/goal) before RDP
    grid: Grid
    free: np.ndarray
    gvd: np.ndarray
    retract_start: list = field(default_factory=list)
    retract_goal: list = field(default_factory=list)
    fallback: bool = False
    expanded: int = 0


def _chain_to_points(grid, cells, world):
    pts = [grid.centre(c) for c in cells]
    pts[0] = tuple(world.start[:2])
    pts[-1] = tuple(world.goal[:2])
    return pts


def simplify(raw, cfg):
    """RDP (epsilon), then drop legs shorter than min_leg while within 2 epsilon of raw."""
    path = rdp(raw, cfg['rdp_epsilon'])
    if cfg.get('min_leg', 0.0) > 0.0:
        path = drop_short_legs(path, raw, cfg['min_leg'], 2.0 * cfg['rdp_epsilon'])
    return path


def plan_safe(world):
    cfg = world.safe
    grid = build_grid(world, cfg['resolution'])
    free = grid.D >= world.robot_radius + cfg['min_margin']
    gvd = gvd_mask(grid, free)
    s = grid.cell_of(*world.start[:2])
    g = grid.cell_of(*world.goal[:2])
    allowed_free = free.copy()
    allowed_free[s] = allowed_free[g] = True  # endpoints may sit a little closer than min_margin
    rs = retract(grid, s, gvd, allowed_free)
    rg = retract(grid, g, gvd, allowed_free)
    allowed = gvd.copy()
    for c in rs + rg:
        allowed[c] = True
    cells, _, expanded = grid_search(grid, allowed, s, g, alpha=0.0, heuristic=False)
    fallback = False
    if cells is None:  # GVD broken by the discretisation: strongly clearance-weighted A*
        fallback = True
        cells, _, expanded = grid_search(grid, allowed_free, s, g, alpha=5.0)
        if cells is None:
            raise RuntimeError('no collision-free path from start to goal')
    raw = _chain_to_points(grid, cells, world)
    path = simplify(raw, cfg)
    return SafePlan(path=path, raw_path=raw, grid=grid, free=free, gvd=gvd,
                    retract_start=rs, retract_goal=rg, fallback=fallback, expanded=expanded)


def plan_weighted(world, alpha=None, grid=None):
    """Grid A* with cost (1 + alpha / d) per metre, simplified with RDP; returns (path, raw)."""
    cfg = world.weighted
    alpha = cfg['alpha'] if alpha is None else alpha
    grid = grid or build_grid(world, cfg['resolution'])
    free = grid.D >= world.robot_radius + world.safe['min_margin']
    s = grid.cell_of(*world.start[:2])
    g = grid.cell_of(*world.goal[:2])
    free[s] = free[g] = True
    cells, _, _ = grid_search(grid, free, s, g, alpha=alpha)
    if cells is None:
        raise RuntimeError('no collision-free path from start to goal')
    raw = _chain_to_points(grid, cells, world)
    return simplify(raw, cfg), raw
