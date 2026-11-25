"""
Minimum-time planner: visibility graph in configuration space + A* with a time cost (no ROS).

Representation. The robot is a disk of radius r, so the configuration space is the plane and
the C-obstacle of the box is its Minkowski sum with a disk of radius rho = r + margin: a rounded
rectangle. Each quarter-circle corner is replaced by k circumscribed segments (vertices at
radius rho / cos(pi / 4k)), so the C-obstacle becomes a convex polygon that contains the true
one (errs safe by at most rho * (sec(pi / 4k) - 1)). In a polygonal free space every shortest
path is a taut string that bends only at convex obstacle vertices, so the visibility graph on
{start, goal, polygon vertices} contains the exact shortest path.

Search. The robot stops and turns in place at each vertex, so time is not proportional to
length. The edge cost charges the turn too:

    c((k -> i) -> j) = l_ij / v + |wrap(psi_ij - psi_ki)| / w + t_vertex (if i is a vertex)

which depends on the incoming leg, so the A* state is (node, predecessor). The heuristic
|n - goal| / v is admissible and consistent (turns and stops only add time).
"""

from dataclasses import dataclass, field
import heapq
import itertools
import math

from robot_control.path_tools import estimate_time, make_waypoints
from robot_control.waypoint_controller import wrap_angle


def c_obstacle_polygon(box, rho, k=2):
    """Convex polygon (counter-clockwise) containing the box grown by rho."""
    r_v = rho / math.cos(math.pi / (4 * k))
    corner_offsets = [(box.hx, box.hy, 0.0), (-box.hx, box.hy, math.pi / 2),
                      (-box.hx, -box.hy, math.pi), (box.hx, -box.hy, 1.5 * math.pi)]
    verts = []
    for ox, oy, a0 in corner_offsets:
        for j in range(k):
            a = a0 + (j + 0.5) * (math.pi / 2) / k
            verts.append((box.cx + ox + r_v * math.cos(a), box.cy + oy + r_v * math.sin(a)))
    return verts


def point_in_convex_polygon(p, poly, tol=1e-9):
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -tol:
            return False
    return True


def segment_crosses_polygon(p, q, poly, step=0.005):
    """Check whether the open segment p -> q passes through the polygon's interior."""
    n = max(2, int(math.ceil(math.dist(p, q) / step)))
    for i in range(1, n):
        t = i / n
        m = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
        if point_in_convex_polygon(m, poly, tol=-1e-7):  # strictly inside
            return True
    return False


@dataclass
class FastPlan:
    path: list                      # final (x, y) points, start .. goal
    raw_path: list                  # A* result before corner merging
    nodes: list                     # visibility graph nodes (0 = start, 1 = goal)
    edges: list                     # (i, j) pairs, i < j
    polygon: list                   # C-obstacle polygon
    rho: float
    cost_s: float
    expanded: int
    merges: list = field(default_factory=list)


def _edge_ok(world, p, q, poly, rho):
    # Walls: plain clearance test against the walls only (the box is the polygon's job, which
    # is conservative); the polygon test keeps the path out of the grown box.
    x, y = world.segment_points(p, q)
    wall = min(float(min(x.min() - world.x_min, world.x_max - x.max())),
               float(min(y.min() - world.y_min, world.y_max - y.max())))
    end_wall = min(_wall_clearance(world, p), _wall_clearance(world, q))
    if wall < min(rho, end_wall) - 1e-6:
        return False
    return not segment_crosses_polygon(p, q, poly)


def _wall_clearance(world, p):
    return min(p[0] - world.x_min, world.x_max - p[0], p[1] - world.y_min, world.y_max - p[1])


def build_visibility_graph(world, rho, k):
    poly = c_obstacle_polygon(world.box, rho, k)
    start, goal = world.start[:2], world.goal[:2]
    nodes = [tuple(start), tuple(goal)]
    for v in poly:  # vertices pushed into a wall are unusable
        if _wall_clearance(world, v) >= rho - 1e-9:
            nodes.append(v)
    for name, p in (('start', start), ('goal', goal)):
        if point_in_convex_polygon(p, poly, tol=-1e-9):
            raise ValueError(f'the {name} is inside the grown obstacle (rho = {rho:.3f} m): '
                             f'move it or lower fast.margin')
    edges = []
    adj = {i: [] for i in range(len(nodes))}
    for i, j in itertools.combinations(range(len(nodes)), 2):
        if _edge_ok(world, nodes[i], nodes[j], poly, rho):
            L = math.dist(nodes[i], nodes[j])
            edges.append((i, j))
            adj[i].append((j, L))
            adj[j].append((i, L))
    return nodes, edges, adj, poly


def astar_time(nodes, adj, start_heading, goal_heading, timing, t_vertex):
    """Run A* over (node, predecessor) states; return (node indices, cost, expanded)."""
    v, w = timing['v'], timing['w']
    goal = nodes[1]

    def h(n):
        return math.dist(nodes[n], goal) / v

    def heading_in(state):
        node, prev = state
        if prev is None:
            return start_heading
        a, b = nodes[prev], nodes[node]
        return math.atan2(b[1] - a[1], b[0] - a[0])

    start_state = (0, None)
    g = {start_state: 0.0}
    parent = {start_state: None}
    tie = itertools.count()
    heap = [(h(0), next(tie), start_state)]
    closed = set()
    expanded = 0
    while heap:
        _, _, state = heapq.heappop(heap)
        if state in closed:
            continue
        closed.add(state)
        expanded += 1
        node, _ = state
        if node == 1:
            seq = []
            s = state
            while s is not None:
                seq.append(s[0])
                s = parent[s]
            return seq[::-1], g[state], expanded
        psi_in = heading_in(state)
        for j, L in adj[node]:
            nxt = (j, node)
            if nxt in closed:
                continue
            psi = math.atan2(nodes[j][1] - nodes[node][1], nodes[j][0] - nodes[node][0])
            cost = L / v + abs(wrap_angle(psi - psi_in)) / w
            if node != 0:
                cost += t_vertex
            if j == 1 and not math.isnan(goal_heading):
                cost += abs(wrap_angle(goal_heading - psi)) / w
            new_g = g[state] + cost
            if new_g < g.get(nxt, math.inf) - 1e-12:
                g[nxt] = new_g
                parent[nxt] = state
                heapq.heappush(heap, (new_g + h(j), next(tie), nxt))
    raise RuntimeError('no path from start to goal in the visibility graph')


def path_time(world, points, intermediate):
    wps = make_waypoints(points, world.goal[2], intermediate)
    return estimate_time(world.start, wps, world.timing)['total_s']


def line_intersection(a, b, c, d):
    """Intersection p = a + t (b - a) of lines a-b and c-d, with t and s (p = c + s (d - c))."""
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]
    ba = (b[0] - a[0], b[1] - a[1])
    dc = (d[0] - c[0], d[1] - c[1])
    den = cross(ba, dc)
    if abs(den) < 1e-12:
        return None
    ca = (c[0] - a[0], c[1] - a[1])
    t = cross(ca, dc) / den
    s = cross(ca, ba) / den
    return (a[0] + t * ba[0], a[1] + t * ba[1]), t, s


def merge_corners(world, path, poly, rho, intermediate):
    """
    Replace runs of vertices b..c by the intersection of lines a->b and c->d.

    Each polygon corner contributes k vertices, i.e. k stops; one corner point costs a little
    length but saves stops. A merge is kept only if it lies beyond b and before c, both new
    segments are collision-free, and the modelled time goes down.
    """
    path = list(path)
    merges = []
    changed = True
    while changed:
        changed = False
        best = None
        base = path_time(world, path, intermediate)
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path) - 1):
                a, b, c, d = path[i - 1], path[i], path[j], path[j + 1]
                hit = line_intersection(a, b, c, d)
                if hit is None:
                    continue
                p, t, s = hit
                if t < 1.0 - 1e-9 or s > 1e-9:
                    continue
                if not (_edge_ok(world, a, p, poly, rho) and _edge_ok(world, p, d, poly, rho)):
                    continue
                cand = path[:i] + [p] + path[j + 1:]
                gain = base - path_time(world, cand, intermediate)
                if gain > 1e-6 and (best is None or gain > best[0]):
                    best = (gain, cand, (i, j, p))
        if best is not None:
            path = best[1]
            merges.append(best[2])
            changed = True
    return path, merges


def plan_fast(world):
    cfg = world.fast
    rho = world.robot_radius + cfg['margin']
    intermediate = 'pass' if cfg['pass_through'] else 'stop'
    t_vertex = world.timing['t_pass'] if cfg['pass_through'] else world.timing['t_stop']
    nodes, edges, adj, poly = build_visibility_graph(world, rho, cfg['polygon_k'])
    seq, cost, expanded = astar_time(nodes, adj, world.start[2], world.goal[2],
                                     world.timing, t_vertex)
    raw = [nodes[i] for i in seq]
    path, merges = raw, []
    if cfg['merge_corners']:
        path, merges = merge_corners(world, raw, poly, rho, intermediate)
    return FastPlan(path=[tuple(map(float, p)) for p in path], raw_path=raw, nodes=nodes,
                    edges=edges, polygon=poly, rho=rho, cost_s=cost, expanded=expanded,
                    merges=merges)
