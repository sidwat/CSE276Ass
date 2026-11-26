"""
Maximum-safety planner: generalized Voronoi diagram (GVD) on a clearance grid + Dijkstra (no ROS).

Representation: the arena is discretised into cells; each cell stores its distance to the
nearest obstacle (wall or box) and which obstacle that is, i.e. the result of the brushfire
wavefront. For axis-aligned boxes the distance is computed exactly (an exact Euclidean distance
transform). Cells where the nearest obstacle changes between neighbours are equidistant from
two obstacles: they form the GVD, the roadmap of locally maximal clearance.

Search: start and goal are retracted onto the GVD by climbing the clearance field; Dijkstra
then finds the shortest route along the GVD cells. The cell chain is simplified with
Ramer-Douglas-Peucker, never letting a segment cut closer to an obstacle than allowed.
"""

from collections import deque
import heapq
import math

import numpy as np

from robot_control.path_tools import Plan, PlanningError, rdp

NEIGHBOURS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class ClearanceGrid:
    """Cell (row, col) has centre (xs[col], xs[row]); D = clearance, label = nearest obstacle."""

    def __init__(self, world, resolution):
        self.res = float(resolution)
        self.n = int(round(world.size / self.res))
        self.xs = (np.arange(self.n) + 0.5) * (world.size / self.n)
        X, Y = np.meshgrid(self.xs, self.xs)
        self.D, self.label = world.clearance(X, Y)

    def cell(self, p):
        q = min(self.n - 1, max(0, int(p[0] / self.res)))
        r = min(self.n - 1, max(0, int(p[1] / self.res)))
        return r, q

    def point(self, c):
        return float(self.xs[c[1]]), float(self.xs[c[0]])

    def inside(self, c):
        return 0 <= c[0] < self.n and 0 <= c[1] < self.n


def gvd_mask(grid, min_clearance):
    """Cells whose 4-neighbour has a different nearest obstacle, and clearance >= min."""
    lab, n = grid.label, grid.n
    g = np.zeros((n, n), dtype=bool)
    for dr, dc in ((0, 1), (1, 0)):
        diff = lab[:n - dr, :n - dc] != lab[dr:, dc:]
        g[:n - dr, :n - dc] |= diff
        g[dr:, dc:] |= diff
    return g & (grid.D >= min_clearance)


def retract(grid, gvd, start, min_free):
    """
    Return the cells from `start` to the nearest Voronoi cell.

    Climbs the clearance field (gradient ascent of the distance transform); if that stalls on a
    plateau, finishes with a breadth-first search through cells with clearance >= min_free.
    """
    trail = [start]
    c = start
    while not gvd[c]:
        nbrs = [(c[0] + a, c[1] + b) for a, b in NEIGHBOURS8 if grid.inside((c[0] + a, c[1] + b))]
        on_gvd = [t for t in nbrs if gvd[t]]
        nxt = max(on_gvd or nbrs, key=lambda t: grid.D[t])
        if not on_gvd and grid.D[nxt] <= grid.D[c]:
            break
        c = nxt
        trail.append(c)
    if gvd[c]:
        return trail
    # plateau / local maximum off the GVD: BFS to the nearest GVD cell
    parent = {c: None}
    queue = deque([c])
    while queue:
        u = queue.popleft()
        if gvd[u]:
            chain = []
            while u is not None:
                chain.append(u)
                u = parent[u]
            return trail[:-1] + chain[::-1]
        for a, b in NEIGHBOURS8:
            v = (u[0] + a, u[1] + b)
            if grid.inside(v) and v not in parent and grid.D[v] >= min_free:
                parent[v] = u
                queue.append(v)
    raise PlanningError('cannot reach the Voronoi diagram from the start/goal')


def dijkstra_cells(grid, allowed, s, t, weight=None):
    """
    Shortest 8-connected path through allowed cells (step h or h*sqrt(2), times weight).

    With weight (an array), a step into cell c costs length * weight[c].
    """
    dist = {s: 0.0}
    parent = {s: None}
    heap = [(0.0, s)]
    while heap:
        d0, c = heapq.heappop(heap)
        if c == t:
            break
        if d0 > dist[c]:
            continue
        for a, b in NEIGHBOURS8:
            v = (c[0] + a, c[1] + b)
            if not grid.inside(v) or not allowed[v]:
                continue
            step = grid.res * (math.sqrt(2.0) if a and b else 1.0)
            nd = d0 + step * (weight[v] if weight is not None else 1.0)
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                parent[v] = c
                heapq.heappush(heap, (nd, v))
    if t not in parent:
        raise PlanningError('start and goal are not connected through the free cells')
    chain = []
    c = t
    while c is not None:
        chain.append(c)
        c = parent[c]
    return chain[::-1]


def simplify(world, raw, eps, required):
    """
    Simplify with RDP, never letting a segment cut closer to an obstacle than `required`.

    A segment whose own endpoints are closer than `required` may come as close as they are.
    """
    def ok(a, b):
        need = min(required, world.clearance_at(*a), world.clearance_at(*b)) - 1e-9
        return world.segment_clearance(a, b) >= need
    return rdp(raw, eps, ok)


def plan_safe(world, timing=None, **overrides):
    """Plan the maximum-clearance path along the GVD; returns a Plan."""
    cfg = {'resolution': 0.02, 'min_clearance': 0.25, 'simplify_tolerance': 0.03}
    cfg.update(world.planner_section('safe'))
    cfg.update(overrides)
    rho = world.robot_radius
    world.check_endpoints(rho)
    grid = ClearanceGrid(world, cfg['resolution'])
    min_clear = max(float(cfg['min_clearance']), rho)
    gvd = gvd_mask(grid, min_clear)
    if not gvd.any():
        raise PlanningError(f'no Voronoi cell has clearance >= {min_clear:.2f} m')
    s_trail = retract(grid, gvd, grid.cell(world.start), rho)
    g_trail = retract(grid, gvd, grid.cell(world.goal), rho)
    chain = dijkstra_cells(grid, gvd, s_trail[-1], g_trail[-1])
    cells = s_trail[:-1] + chain + g_trail[-2::-1]
    raw = ([tuple(world.start[:2])] + [grid.point(c) for c in cells[1:-1]]
           + [tuple(world.goal[:2])])
    path = simplify(world, raw, float(cfg['simplify_tolerance']), min_clear)
    guaranteed = min(world.path_clearance(path)[0], min_clear)
    return Plan('safe', path, guaranteed, raw,
                {'grid': grid, 'gvd': gvd, 'config': cfg, 'min_clearance': min_clear})


def plan_weighted(world, timing=None, **overrides):
    """
    Grid A*-style search with cost integral of (1 + alpha / clearance) ds.

    alpha = 0 is the shortest grid path; large alpha approaches the GVD. Used in the report to
    show the trade-off between the two planners. Dijkstra is used (no heuristic needed here).
    """
    cfg = {'resolution': 0.04, 'alpha': 0.3, 'simplify_tolerance': 0.03}
    cfg.update(world.planner_section('weighted'))
    cfg.update(overrides)
    rho = world.robot_radius
    world.check_endpoints(rho)
    grid = ClearanceGrid(world, cfg['resolution'])
    allowed = grid.D >= rho
    weight = 1.0 + float(cfg['alpha']) / np.maximum(grid.D, 1e-3)
    s, t = grid.cell(world.start), grid.cell(world.goal)
    allowed[s] = allowed[t] = True
    chain = dijkstra_cells(grid, allowed, s, t, weight)
    raw = ([tuple(world.start[:2])] + [grid.point(c) for c in chain[1:-1]]
           + [tuple(world.goal[:2])])
    path = simplify(world, raw, float(cfg['simplify_tolerance']), rho)
    return Plan('weighted', path, world.path_clearance(path)[0], raw,
                {'grid': grid, 'config': cfg})
