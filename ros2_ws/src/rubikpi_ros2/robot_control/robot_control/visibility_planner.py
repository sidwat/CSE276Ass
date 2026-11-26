"""
Minimum-time / minimum-distance planner: visibility graph in configuration space + A* (no ROS).

Representation: every box is grown by rho = robot_radius + margin (Minkowski sum with the robot
disk), its rounded corners replaced by a circumscribed polygon. In a polygonal free space every
shortest path is a polyline that bends only at convex obstacle vertices, so the graph of
{start, goal, C-obstacle vertices} with an edge wherever the straight segment stays free contains
the exact shortest path.

Search: A* over (vertex, previous vertex) so that the edge cost can include the turn the robot
makes at the vertex: cost = length / v + |turn| / w + t_pass. The heuristic, straight-line
distance / v, never overestimates (turning and stopping only add time), so A* is optimal.
"""

import heapq
import math

from robot_control.path_tools import Plan, PlanningError, Timing
from robot_control.waypoint_controller import wrap_angle


def build_graph(world, rho, segments=2):
    """Nodes (start, goal, free C-obstacle vertices) and their visibility edges."""
    nodes = [tuple(world.start[:2]), tuple(world.goal[:2])]
    for box in world.boxes:
        # grow the polygon a hair so its own vertices pass the clearance test
        for p in world.c_obstacle(box, rho * 1.0005 + 1e-6, segments):
            if world.clearance_at(*p) >= rho:
                nodes.append(p)
    edges = {i: [] for i in range(len(nodes))}
    pairs = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if world.segment_free(nodes[i], nodes[j], rho):
                d = math.dist(nodes[i], nodes[j])
                edges[i].append((j, d))
                edges[j].append((i, d))
                pairs.append((i, j))
    return nodes, edges, pairs


def astar(nodes, edges, start_heading, timing, turn_cost=True, goal_heading=float('nan')):
    """
    Search from node 0 to node 1 with A*; return the node indices of the best path.

    With turn_cost the cost is travel time including turning at each vertex (the state is
    (node, predecessor), because the turn depends on the incoming heading); without it, length.
    A finite goal_heading also charges the final turn to it on arrival at the goal.
    """
    goal = nodes[1]

    def h(i):
        d = math.dist(nodes[i], goal)
        return d / timing.v if turn_cost else d

    start = (0, -1)
    best = {start: 0.0}
    parent = {start: None}
    heap = [(h(0), 0.0, start)]
    found = None
    while heap:
        _, g, state = heapq.heappop(heap)
        if g > best.get(state, math.inf) + 1e-12:
            continue
        i, prev = state
        if i == 1:
            found = state
            break
        if prev < 0:
            heading_in = start_heading
        else:
            heading_in = math.atan2(nodes[i][1] - nodes[prev][1], nodes[i][0] - nodes[prev][0])
        for j, d in edges[i]:
            if j == prev:
                continue
            if turn_cost:
                heading_out = math.atan2(nodes[j][1] - nodes[i][1], nodes[j][0] - nodes[i][0])
                c = (d / timing.v + abs(wrap_angle(heading_out - heading_in)) / timing.w
                     + (timing.t_pass if prev >= 0 else 0.0))
                if j == 1 and math.isfinite(goal_heading):
                    c += abs(wrap_angle(goal_heading - heading_out)) / timing.w
            else:
                c = d
            ng = g + c
            nxt = (j, i)
            if ng < best.get(nxt, math.inf) - 1e-12:
                best[nxt] = ng
                parent[nxt] = state
                heapq.heappush(heap, (ng + h(j), ng, nxt))
    if found is None:
        raise PlanningError('no collision-free path: the C-obstacles block every route from '
                            'start to goal (reduce the margin or the robot radius?)')
    out = []
    state = found
    while state is not None:
        out.append(state[0])
        state = parent[state]
    return out[::-1]


def _intersection(a, b, c, d):
    """Return (point, t) where line a->b meets line c->d (p = a + t (b - a)), or None."""
    den = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
    if abs(den) < 1e-12:
        return None
    t = ((c[0] - a[0]) * (d[1] - c[1]) - (c[1] - a[1]) * (d[0] - c[0])) / den
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])), t


def merge_corners(world, path, rho, max_extra=0.10):
    """
    Replace runs of consecutive vertices by one, where the incoming and outgoing legs meet.

    Wrapping a rounded corner gives several vertices a few cm apart, and every waypoint costs
    a stop and a turn. The legs a->b and c->d extended meet at p; vertices b..c are replaced by
    p if both new legs stay free and the path grows by at most max_extra.
    """
    path = [tuple(p) for p in path]
    while len(path) > 3:
        best = None
        for i in range(1, len(path) - 1):
            for j in range(i + 1, len(path) - 1):
                hit = _intersection(path[i - 1], path[i], path[j], path[j + 1])
                if hit is None:
                    continue
                p, t = hit
                if t < 1.0 - 1e-9:
                    continue
                old = sum(math.dist(path[k], path[k + 1]) for k in range(i - 1, j + 1))
                extra = math.dist(path[i - 1], p) + math.dist(p, path[j + 1]) - old
                if extra > max_extra:
                    continue
                if not (world.clearance_at(*p) >= rho
                        and world.segment_free(path[i - 1], p, rho)
                        and world.segment_free(p, path[j + 1], rho)):
                    continue
                key = (j - i, -extra)
                if best is None or key > best[0]:
                    best = (key, i, j, p)
        if best is None:
            break
        _, i, j, p = best
        path = path[:i] + [p] + path[j + 1:]
    return path


def plan_fast(world, timing=None, **overrides):
    """Plan the minimum-time path; returns a Plan (path[0] = start, path[-1] = goal)."""
    cfg = {'margin': 0.08, 'corner_segments': 2, 'turn_cost': True, 'merge_corners': True,
           'max_merge_extra': 0.10}
    cfg.update(world.planner_section('fast'))
    cfg.update(overrides)
    timing = timing or Timing.from_world(world)
    rho = world.robot_radius + float(cfg['margin'])
    world.check_endpoints(rho)
    nodes, edges, pairs = build_graph(world, rho, int(cfg['corner_segments']))
    order = astar(nodes, edges, world.start[2], timing, bool(cfg['turn_cost']), world.goal[2])
    raw = [nodes[i] for i in order]
    path = merge_corners(world, raw, rho, float(cfg['max_merge_extra'])) \
        if cfg['merge_corners'] else list(raw)
    polygons = [world.c_obstacle(b, rho, int(cfg['corner_segments'])) for b in world.boxes]
    return Plan('fast', path, rho, raw,
                {'nodes': nodes, 'edges': pairs, 'polygons': polygons, 'config': cfg})
