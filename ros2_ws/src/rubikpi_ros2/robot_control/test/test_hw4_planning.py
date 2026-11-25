import math
import os

import numpy as np
import pytest
from robot_control.gvd_planner import plan_safe
from robot_control.hw4_plan import make_plan
from robot_control.hw4_sim import load_params_yaml, simulate
from robot_control.hw4_world import Box, load_world, number, world_from_dict
from robot_control.path_tools import (
    drop_short_legs, estimate_time, make_waypoints, max_deviation, polyline_length, rdp)
from robot_control.visibility_planner import (
    c_obstacle_polygon, plan_fast, point_in_convex_polygon)

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')


def make_world(**over):
    data = {
        'arena': {'x_min': 0.0, 'x_max': 3.0, 'y_min': 0.0, 'y_max': 3.0},
        'obstacle': {'center': [1.5, 1.5], 'size': [0.61, 0.61],
                     'face_tags': {'east': 10, 'north': 11, 'west': 12, 'south': 13}},
        'robot': {'radius': 0.13},
        'start': [2.54, 0.46, 'pi/2'],
        'goal': [0.46, 2.54, None],
        'landmarks': [[0, 3.0, 0.96, 'pi'], [7, 3.0, 2.03, 'pi'], [1, 2.0, 3.0, '-pi/2'],
                      [3, 0.96, 3.0, '-pi/2'], [4, 0.0, 2.04, 0], [8, 0.0, 0.96, 0],
                      [9, 0.97, 0.0, 'pi/2'], [5, 1.98, 0.0, 'pi/2']],
    }
    for k, v in over.items():
        data[k] = v
    return world_from_dict(data)


# ---- world model --------------------------------------------------------------------------
def test_number_parses_expressions_and_rejects_code():
    assert number('pi/2') == pytest.approx(math.pi / 2)
    assert number('90*deg') == pytest.approx(math.pi / 2)
    assert number(-1) == -1.0
    assert math.isnan(number(None))
    with pytest.raises(ValueError):
        number('__import__("os")')


def test_box_distance_is_zero_inside_and_euclidean_outside():
    b = Box(1.0, 1.0, 0.4, 0.2)
    assert b.distance(1.0, 1.0) == 0.0
    assert b.distance(1.5, 1.0) == pytest.approx(0.3)           # east face
    assert b.distance(1.0, 0.5) == pytest.approx(0.4)           # south face
    assert b.distance(1.5, 1.5) == pytest.approx(math.hypot(0.3, 0.4))  # NE corner


def test_face_tags_are_generated_and_measured_entries_win():
    w = make_world()
    assert w.landmarks[10] == pytest.approx((1.805, 1.5, 0.0))
    assert w.landmarks[13] == pytest.approx((1.5, 1.195, -math.pi / 2))
    w2 = make_world(landmarks=[[10, 1.81, 1.52, 0.05]])
    assert w2.landmarks[10] == pytest.approx((1.81, 1.52, 0.05))


def test_config_world_file_loads_cleanly():
    w = load_world(os.path.join(CONFIG, 'hw4_world.yaml'))
    assert w.check() == []
    assert len(w.landmarks) == 12


def test_c_obstacle_polygon_contains_the_rounded_rectangle():
    w = make_world()
    rho = 0.23
    for k in (1, 2, 4):
        poly = c_obstacle_polygon(w.box, rho, k)
        assert len(poly) == 4 * k
        bulge = rho * (1.0 / math.cos(math.pi / (4 * k)) - 1.0)
        for vx, vy in poly:  # every vertex is outside, by at most the bulge
            d = float(w.box.distance(vx, vy))
            assert rho - 1e-9 <= d <= rho + bulge + 1e-9
        for a in np.linspace(0, 2 * math.pi, 400):  # points at distance rho are inside
            # walk out from the centre along direction a until the box distance is rho
            lo, hi = 0.0, 2.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if w.box.distance(1.5 + mid * math.cos(a), 1.5 + mid * math.sin(a)) < rho:
                    lo = mid
                else:
                    hi = mid
            p = (1.5 + lo * math.cos(a), 1.5 + lo * math.sin(a))
            assert point_in_convex_polygon(p, poly, tol=1e-7)


# ---- fast planner -------------------------------------------------------------------------
def test_fast_path_is_straight_when_nothing_is_in_the_way():
    w = make_world(obstacle={'center': [2.6, 2.6], 'size': 0.2})
    plan = plan_fast(w)
    assert len(plan.path) == 2
    assert polyline_length(plan.path) == pytest.approx(math.dist(w.start[:2], w.goal[:2]))


def test_fast_path_is_collision_free_and_near_the_exact_shortest_length():
    w = make_world()
    plan = plan_fast(w)
    for a, b in zip(plan.path[:-1], plan.path[1:]):
        assert w.segment_clearance(a, b) >= plan.rho - 1e-3
    w.fast['polygon_k'] = 16  # nearly the exact rounded C-obstacle
    w.fast['merge_corners'] = False
    fine = polyline_length(plan_fast(w).path)
    # the octagon errs long by ~1 cm; merging the corner into one vertex costs ~2 cm more
    assert fine - 1e-6 <= polyline_length(plan.raw_path) <= fine + 0.02
    assert polyline_length(plan.path) <= fine + 0.05


def test_fast_route_side_follows_the_start_heading():
    north = plan_fast(make_world()).path
    assert north[1][0] > 1.5 and north[1][1] > 1.5             # round the NE corner
    west = plan_fast(make_world(start=[2.54, 0.46, 'pi'])).path
    assert west[1][0] < 1.5 and west[1][1] < 1.5               # round the SW corner


def test_corner_merge_leaves_one_pass_through_vertex():
    plan = make_plan(make_world(), 'fast')
    assert len(plan.waypoints) == 2
    assert math.isnan(plan.waypoints[0][2])
    assert not math.isnan(plan.waypoints[-1][2])


# ---- safe planner -------------------------------------------------------------------------
def test_gvd_band_lies_midway_between_wall_and_box():
    w = make_world()
    plan = plan_safe(w)
    g = plan.grid
    y_star = (0.0 + (1.5 - 0.305)) / 2.0                       # south band
    cols = [j for j in range(g.nx) if 1.3 < g.x0 + j * g.res < 1.7]
    for j in cols:
        rows = [i for i in range(g.ny) if plan.gvd[i, j] and g.y0 + i * g.res < 1.0]
        assert rows
        ys = [g.y0 + i * g.res for i in rows]
        assert min(abs(y - y_star) for y in ys) <= g.res


def test_safe_path_keeps_band_clearance_from_the_box():
    w = make_world()
    plan = make_plan(w, 'safe')
    band = (1.5 - 0.305) / 2.0                                 # wall-box gap / 2
    eps = w.safe['rdp_epsilon']
    assert plan.metrics['min_box_gap_m'] >= band - w.robot_radius - 2 * eps - w.safe['resolution']
    assert plan.metrics['min_clearance_m'] >= 0.46 - 2 * eps   # start/goal are the minimum
    assert plan.metrics['min_body_gap_m'] > 0.25
    legs = [math.dist(a, b) for a, b in zip(plan.path[:-1], plan.path[1:])]
    assert min(legs) >= w.safe['min_leg'] - 1e-9


def test_safe_is_safer_and_fast_is_faster():
    w = make_world()
    fast, safe = make_plan(w, 'fast'), make_plan(w, 'safe')
    assert safe.metrics['min_box_gap_m'] > fast.metrics['min_box_gap_m'] + 0.2
    assert safe.metrics['mean_clearance_m'] > fast.metrics['mean_clearance_m']
    assert fast.metrics['length_m'] < safe.metrics['length_m']
    assert fast.metrics['est_time_s'] < safe.metrics['est_time_s']


def test_stop_headings_see_a_tag_when_one_can():
    w = make_world()
    for mode in ('fast', 'safe'):
        for x, y, th in make_plan(w, mode).waypoints:
            if not math.isnan(th):
                assert w.tags_in_view((x, y, th)), (mode, x, y, th)


# ---- path tools ---------------------------------------------------------------------------
def test_rdp_keeps_endpoints_bounds_deviation_and_is_idempotent():
    xs = np.linspace(0, 2, 81)
    pts = list(zip(xs, 0.1 * np.sin(3 * xs)))
    out = rdp(pts, 0.02)
    assert out[0] == pytest.approx(pts[0]) and out[-1] == pytest.approx(pts[-1])
    assert max_deviation(pts, out) <= 0.02 + 1e-9
    assert rdp(out, 0.02) == out
    assert len(out) < len(pts)


def test_drop_short_legs_respects_the_deviation_bound():
    raw = [(0, 0), (1, 0), (1.05, 0.02), (2, 0)]
    out = drop_short_legs(raw, raw, 0.15, 0.05)
    assert len(out) == 3 and max_deviation(raw, out) <= 0.05
    assert drop_short_legs(raw, raw, 0.15, 0.001) == raw    # too strict: unchanged


def test_make_waypoints_headings():
    pts = [(0, 0), (1, 0), (1, 1)]
    stop = make_waypoints(pts, math.nan, 'stop')
    assert stop[0][2] == pytest.approx(math.pi / 2)            # outgoing leg
    assert stop[1][2] == pytest.approx(math.pi / 2)            # last leg when goal is free
    passing = make_waypoints(pts, 0.3, 'pass')
    assert math.isnan(passing[0][2]) and passing[1][2] == 0.3


def test_estimate_time_counts_drive_turns_and_stops():
    timing = {'v': 0.25, 'w': 1.0, 't_stop': 1.5, 't_pass': 0.3}
    t = estimate_time((0, 0, 0), [(1, 0, math.nan), (1, 1, math.pi / 2)], timing)
    assert t['drive_s'] == pytest.approx(8.0)
    assert t['turn_s'] == pytest.approx(math.pi / 2)
    assert t['vertex_s'] == pytest.approx(1.8)


# ---- simulation ---------------------------------------------------------------------------
@pytest.mark.parametrize('mode', ['fast', 'safe'])
def test_simulated_runs_reach_the_goal_without_contact(mode):
    w = load_world(os.path.join(CONFIG, 'hw4_world.yaml'))
    params = load_params_yaml(os.path.join(CONFIG, 'hw4.yaml'))
    plan = make_plan(w, mode)
    res = simulate(w, plan.waypoints, params, seed=0)
    assert res['done']
    assert not res['collided']
    assert math.dist(res['true_final'][:2], w.goal[:2]) < 0.06
    assert res['time'] < 2.0 * plan.metrics['est_time_s'] + 10.0
