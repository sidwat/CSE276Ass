import math
import os

import numpy as np
import pytest
from robot_control.gvd_planner import plan_safe, plan_weighted
from robot_control.hw4_tool import load_params, simulate
from robot_control.path_tools import (
    estimate_time, path_length, rdp, Timing, to_waypoints, write_waypoints)
from robot_control.visibility_planner import plan_fast
from robot_control.waypoint_controller import (
    Gains, load_waypoints, WaypointController, wrap_angle)
from robot_control.world_model import Box, load_world, World, world_from_dict, WorldError

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')
L = 3.048


def make_world(side=0.6096, start=(2.59, 0.46, math.pi / 2), goal=(0.46, 2.59, -math.pi / 4),
               radius=0.13, **planner):
    boxes = [Box(L / 2, L / 2, side / 2, side / 2)] if side else []
    return World(L, boxes, radius, start, goal, {}, planner, 'test')


# --- world model -------------------------------------------------------------------------------

def test_box_distance_is_zero_inside_and_euclidean_outside():
    b = Box(1.0, 1.0, 0.5, 0.25)
    assert b.distance(1.2, 1.1) == 0.0
    assert b.distance(2.0, 1.0) == pytest.approx(0.5)            # east of the east face
    assert b.distance(1.8, 1.55) == pytest.approx(math.hypot(0.3, 0.3))  # off the NE corner


def test_clearance_is_nearest_wall_or_box_with_its_label():
    w = make_world()
    d, lab = w.clearance(0.1, 1.5)
    assert float(d) == pytest.approx(0.1) and int(lab) == 0      # west wall
    d, lab = w.clearance(1.524 + 0.3048 + 0.05, 1.524)
    assert float(d) == pytest.approx(0.05) and int(lab) == 4     # the box


def test_c_obstacle_polygon_contains_the_rounded_box():
    w = make_world()
    rho = 0.2
    poly = np.array(w.c_obstacle(w.boxes[0], rho, segments=2))
    # every point at distance < rho from the box lies inside the (convex) polygon
    ang = np.linspace(0, 2 * np.pi, 400)
    for a in ang:
        for r in (0.0, 0.5, 0.99):
            # walk out from the box centre until distance r*rho from the box surface
            t = np.linspace(0, 1.5, 3000)
            px, py = L / 2 + t * np.cos(a), L / 2 + t * np.sin(a)
            k = np.argmax(w.boxes[0].distance(px, py) >= r * rho)
            p = np.array([px[k], py[k]])
            edges = np.roll(poly, -1, axis=0) - poly
            cross = edges[:, 0] * (p[1] - poly[:, 1]) - edges[:, 1] * (p[0] - poly[:, 0])
            assert np.all(cross >= -1e-9)


def test_world_yaml_loads_and_endpoints_are_valid():
    w = load_world(os.path.join(CONFIG, 'hw4_world.yaml'))
    assert len(w.boxes) == 1 and len(w.landmarks) == 11
    assert w.landmarks['Standard41h12:1'] == (1.12, 1.60, 3.1416)
    assert w.landmarks['Standard41h12:2'] == (1.44, 1.92, 1.5708)
    assert w.landmarks['Standard41h12:3'] == (1.75, 1.60, 0.0)
    assert w.landmarks[1][:2] == (2.00, 3.04)
    assert w.landmarks[3][:2] == (0.96, 3.04)
    assert w.start[0] + w.start[1] == pytest.approx(w.size, abs=0.01)  # on the diagonal
    w.check_endpoints(w.robot_radius + 0.08)


def test_world_rejects_start_inside_the_box():
    data = {'arena': {'size': L}, 'obstacles': [{'center': [1.5, 1.5], 'size': [0.6, 0.6]}],
            'robot': {'radius': 0.13}, 'start': [1.5, 1.5, 0.0], 'goal': [0.4, 2.6, None]}
    with pytest.raises(WorldError):
        world_from_dict(data)


# --- fast planner --------------------------------------------------------------------------------

def test_fast_without_obstacle_is_the_straight_line():
    w = make_world(side=0)
    plan = plan_fast(w, margin=0.08)
    assert len(plan.path) == 2
    assert path_length(plan.path) == pytest.approx(math.dist(w.start[:2], w.goal[:2]))


def test_fast_plan_is_free_near_optimal_and_has_few_waypoints():
    w = make_world()
    plan = plan_fast(w, margin=0.08)
    rho = w.robot_radius + 0.08
    for p, q in zip(plan.path[:-1], plan.path[1:]):
        assert w.segment_free(p, q, rho)
    # lower bound: tangent path around a disk of radius rho at the box corner
    corner = (L / 2 + 0.3048, L / 2 + 0.3048)
    lower = math.dist(w.start[:2], corner) + math.dist(corner, w.goal[:2])
    assert lower < path_length(plan.path) < lower + 0.25
    assert len(plan.path) - 1 <= 3


def test_time_cost_picks_the_side_the_robot_is_already_facing():
    north = plan_fast(make_world(start=(2.59, 0.46, math.pi / 2)), margin=0.08)
    west = plan_fast(make_world(start=(2.59, 0.46, math.pi)), margin=0.08)
    # facing north it goes round the north-east of the box, facing west round the south-west
    assert north.path[1][0] > L / 2 and north.path[1][1] > L / 2
    assert west.path[1][0] < L / 2 and west.path[1][1] < L / 2


def test_fast_plan_fails_cleanly_when_the_margin_blocks_the_start():
    with pytest.raises(WorldError):
        plan_fast(make_world(), margin=0.4)


# --- safe planner --------------------------------------------------------------------------------

def test_safe_plan_follows_the_voronoi_band_between_box_and_wall():
    w = make_world()
    plan = plan_safe(w, resolution=0.02, min_clearance=0.25, simplify_tolerance=0.03)
    band = (L + L / 2 + 0.3048) / 2   # equidistant from the east wall and the box east face
    pts = np.array(plan.raw_path)
    # facing north at the start, the route goes up the east side of the box
    on_band = pts[(pts[:, 0] > L / 2) & (pts[:, 1] > 1.3) & (pts[:, 1] < 1.7)]
    assert len(on_band) > 5
    assert np.all(np.abs(on_band[:, 0] - band) < 0.04)
    dmin, dmean, d = w.path_clearance(plan.path)
    # away from the start and goal (0.46 m from the walls) clearance stays near the band's
    assert dmin >= 0.45
    assert dmean > 0.55


def test_safe_is_safer_and_longer_than_fast_and_weighted_is_in_between():
    w = make_world()
    fast = plan_fast(w, margin=0.08)
    safe = plan_safe(w)
    weighted = plan_weighted(w, alpha=0.3)
    m = {n: w.path_clearance(p.path)[1] for n, p in
         (('fast', fast), ('safe', safe), ('weighted', weighted))}
    assert m['fast'] < m['weighted'] < m['safe']
    assert path_length(fast.path) < path_length(weighted.path) < path_length(safe.path)


def test_rdp_keeps_endpoints_bounds_error_and_is_idempotent():
    t = np.linspace(0, np.pi, 200)
    pts = [(float(x), float(np.sin(x))) for x in t]
    out = rdp(pts, 0.02)
    assert out[0] == pts[0] and out[-1] == pts[-1]
    for p in pts:  # every original point is within eps of the simplified polyline
        d = min(_seg_dist(p, a, b) for a, b in zip(out[:-1], out[1:]))
        assert d <= 0.02 + 1e-9
    assert rdp(out, 0.02) == out


def _seg_dist(p, a, b):
    a, b, p = np.array(a), np.array(b), np.array(p)
    t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


# --- waypoints, controller -----------------------------------------------------------------------

def test_waypoints_mark_intermediate_vertices_pass_through(tmp_path):
    path = [(0, 0), (1, 0), (1, 1)]
    wps = to_waypoints(path, goal_theta=0.5, pass_through=True)
    assert math.isnan(wps[0][2]) and wps[1] == (1.0, 1.0, 0.5)
    wps = to_waypoints(path, pass_through=False)
    assert wps[0][2] == pytest.approx(0.0) and wps[1][2] == pytest.approx(math.pi / 2)
    f = tmp_path / 'wp.txt'
    write_waypoints(str(f), to_waypoints(path, 0.5, True), ['header'])
    loaded = load_waypoints(str(f))
    assert math.isnan(loaded[0][2]) and loaded[1] == pytest.approx((1.0, 1.0, 0.5))


def test_time_estimate_counts_stops_only_where_the_robot_settles():
    path = [(0, 0), (1, 0), (1, 1)]
    tm = Timing(v=0.5, w=1.0, t_stop=2.0, t_pass=0.5)
    assert estimate_time(path, 0.0, tm, pass_through=True) == pytest.approx(
        2 / 0.5 + (math.pi / 2) / 1.0 + 2.0 + 0.5)
    assert estimate_time(path, 0.0, tm, pass_through=False) == pytest.approx(
        2 / 0.5 + (math.pi / 2) / 1.0 + 2 * 2.0)


def test_controller_drives_through_pass_through_waypoints_without_settling():
    wps = [(1.0, 0.0, float('nan')), (1.0, 1.0, float('nan')), (0.0, 1.0, math.pi)]
    c = WaypointController(wps, Gains(settle_time=0.5, pass_radius=0.06))
    x = y = th = 0.0
    dt = 0.05
    phases, reached = set(), []
    for _ in range(4000):
        v, w, r = c.update(x, y, th, dt)
        if r is not None:
            reached.append((r, c.phase))
        if c.index < 2:
            phases.add(c.phase)
        if c.done:
            break
        x += v * math.cos(th + w * dt / 2) * dt
        y += v * math.sin(th + w * dt / 2) * dt
        th = wrap_angle(th + w * dt)
    assert c.done
    assert [r for r, _ in reached] == [0, 1, 2]
    assert WaypointController.SETTLE not in phases          # never settled before the goal
    assert math.hypot(x, y - 1.0) < c.gains.pos_tolerance + 0.01


# --- end to end ----------------------------------------------------------------------------------

@pytest.mark.parametrize('mode', ['fast', 'safe'])
def test_simulated_run_reaches_the_goal_without_touching_anything(mode):
    w = load_world(os.path.join(CONFIG, 'hw4_world.yaml'))
    plan = plan_fast(w) if mode == 'fast' else plan_safe(w)
    params = load_params(os.path.join(CONFIG, 'hw4.yaml'))
    result = simulate(w, to_waypoints(plan.path, w.goal[2], True), params, seed=1)
    assert result['done']
    Q = np.array([r[1:3] for r in result['truth']])
    d, _ = w.clearance(Q[:, 0], Q[:, 1])
    assert d.min() > w.robot_radius                         # no contact in simulation
    assert math.hypot(Q[-1, 0] - w.goal[0], Q[-1, 1] - w.goal[1]) < 0.1


def test_fast_plan_is_within_3_cm_of_the_exact_shortest_path():
    # exact optimum round the NE corner: tangent + arc (outer side) + tangent, radius rho
    w = make_world()
    rho = w.robot_radius + 0.08
    c = (L / 2 + 0.3048, L / 2 + 0.3048)
    S, G = w.start[:2], w.goal[:2]
    dS, dG = math.dist(S, c), math.dist(G, c)
    aS = math.atan2(S[1] - c[1], S[0] - c[0])
    aG = math.atan2(G[1] - c[1], G[0] - c[0])
    outer = 2 * math.pi - abs(wrap_angle(aG - aS))
    exact = (math.sqrt(dS ** 2 - rho ** 2) + math.sqrt(dG ** 2 - rho ** 2)
             + rho * (outer - math.acos(rho / dS) - math.acos(rho / dG)))
    plan = plan_fast(w, margin=0.08)
    assert exact <= path_length(plan.path) < exact + 0.03
    fine = plan_fast(w, margin=0.08, corner_segments=32, merge_corners=False, turn_cost=False)
    assert path_length(fine.path) == pytest.approx(exact, abs=0.002)


def test_sharp_pass_through_corner_is_reached_within_pos_tolerance():
    # a 90 degree bend: the robot turns in place there, so it must not turn 6 cm early
    wps = [(1.0, 0.0, float('nan')), (1.0, 1.0, math.pi / 2)]
    c = WaypointController(wps, Gains(pass_radius=0.06, pos_tolerance=0.02, realign_angle=0.3))
    assert c.update(0.95, 0.0, 0.0, 0.05)[2] is None       # 5 cm short: not yet
    v, w, reached = c.update(0.985, 0.0, 0.0, 0.05)       # 1.5 cm short: counts
    assert reached == 0
    # a gentle bend keeps the larger pass radius
    wps = [(1.0, 0.0, float('nan')), (2.0, 0.1, 0.0)]
    c = WaypointController(wps, Gains(pass_radius=0.06, pos_tolerance=0.02, realign_angle=0.3))
    assert c.update(0.95, 0.0, 0.0, 0.05)[2] == 0
