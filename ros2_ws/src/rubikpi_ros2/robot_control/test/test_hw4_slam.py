import math
import os

import numpy as np
import pytest
from robot_control.hw4_slam_tool import simulate
from robot_control.hw4_tool import load_params
from robot_control.localization import NoiseParams
from robot_control.slam import EKFSLAM, SLAMParams
from robot_control.slam_obstacle import (
    estimate_obstacle, prior_estimate, prune_short_legs, seed_known_walls, SlamConfig,
    SlamPlanner)
from robot_control.waypoint_controller import Gains, WaypointController
from robot_control.world_model import load_world

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')
WORLD = os.path.join(CONFIG, 'hw4_world_slam.yaml')
H = 0.0865  # tag half width


def world():
    return load_world(WORLD)


def lm(x, y, psi, std=0.01, psi_std=0.05):
    return ((x, y, psi), np.diag([std ** 2, std ** 2]), psi_std ** 2)


def true_bounds(w):
    b = w.boxes[0]
    return (b.cx - b.hx, b.cx + b.hx, b.cy - b.hy, b.cy + b.hy)


def face_tags(w, faces, flush=True):
    """Map entries for tags on the given faces of the true box (two per face)."""
    x0, x1, y0, y1 = true_bounds(w)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    off = (x1 - x0) / 2 - H if flush else 0.0
    out, k = {}, 100
    for face in faces:
        for s in (-1, 1):
            if face == 'east':
                out[k] = lm(x1, cy + s * off, 0.0)
            elif face == 'west':
                out[k] = lm(x0, cy + s * off, math.pi)
            elif face == 'north':
                out[k] = lm(cx + s * off, y1, math.pi / 2)
            else:
                out[k] = lm(cx + s * off, y0, -math.pi / 2)
            k += 1
    return out


def contains(est, bounds, tol=1e-9):
    x0, x1, y0, y1 = bounds
    return (est.x_min <= x0 + tol and est.x_max >= x1 - tol
            and est.y_min <= y0 + tol and est.y_max >= y1 - tol)


def inside_prior(est, cfg):
    p = prior_estimate(cfg)
    return (est.x_min >= p.x_min - 1e-9 and est.x_max <= p.x_max + 1e-9
            and est.y_min >= p.y_min - 1e-9 and est.y_max <= p.y_max + 1e-9)


# ---- obstacle estimate ---------------------------------------------------------------------
def test_no_obstacle_tags_gives_the_prior():
    w = world()
    cfg = SlamConfig.from_world(w)
    walls = {0: lm(3.05, 0.96, math.pi), 4: lm(-0.04, 2.04, 0.0)}  # wall tags are ignored
    est = estimate_obstacle(walls, w, cfg)
    assert est.source == 'prior' and est.bounds() == prior_estimate(cfg).bounds()


@pytest.mark.parametrize('faces', [['east'], ['south'], ['east', 'south'],
                                   ['east', 'west', 'north', 'south']])
def test_estimate_contains_the_box_and_stays_inside_the_prior(faces):
    w = world()
    cfg = SlamConfig.from_world(w)
    est = estimate_obstacle(face_tags(w, faces), w, cfg)
    assert est.source == 'slam' and sorted(est.faces) == sorted(faces)
    assert contains(est, true_bounds(w))
    assert inside_prior(est, cfg)


def test_flush_tags_on_all_faces_pin_the_box():
    w = world()
    cfg = SlamConfig.from_world(w)
    est = estimate_obstacle(face_tags(w, ['east', 'west', 'north', 'south']), w, cfg)
    widen = cfg.k_sigma * 0.01 + cfg.bound_margin
    for e, t in zip(est.bounds(), true_bounds(w)):
        assert abs(e - t) <= widen + 1e-6


def test_more_faces_never_grow_the_estimate():
    w = world()
    cfg = SlamConfig.from_world(w)
    one = estimate_obstacle(face_tags(w, ['east']), w, cfg)
    two = estimate_obstacle(face_tags(w, ['east', 'north']), w, cfg)
    assert two.x_min >= one.x_min - 1e-9 and two.x_max <= one.x_max + 1e-9
    assert two.y_min >= one.y_min - 1e-9 and two.y_max <= one.y_max + 1e-9


def test_a_very_uncertain_tag_cannot_grow_the_estimate_beyond_the_prior():
    w = world()
    cfg = SlamConfig.from_world(w)
    est = estimate_obstacle({50: lm(1.75, 1.6, 0.0, std=0.5)}, w, cfg)
    assert inside_prior(est, cfg)


def test_contradicting_faces_fall_back_to_the_prior_on_that_axis():
    w = world()
    cfg = SlamConfig.from_world(w)
    tags = {50: lm(1.2, 1.6, 0.0), 51: lm(1.8, 1.6, math.pi)}  # 'east' face west of 'west'
    est = estimate_obstacle(tags, w, cfg)
    p = prior_estimate(cfg)
    assert est.conflicts == ['x'] and (est.x_min, est.x_max) == (p.x_min, p.x_max)


def test_tags_with_uncertain_orientation_are_not_used():
    w = world()
    cfg = SlamConfig.from_world(w)
    est = estimate_obstacle({50: lm(1.75, 1.6, 0.0, psi_std=0.6)}, w, cfg)
    assert est.faces == {} and est.tags == {50: None}


# ---- planning ------------------------------------------------------------------------------
def test_planner_plans_around_the_prior_then_replans_on_the_map():
    w = world()
    p = SlamPlanner(w, 'fast')
    wps = p.update(0.0, w.start, {})
    assert wps and p.history[-1]['reason'] == 'initial'
    assert p.history[-1]['estimate']['source'] == 'prior'
    tags = face_tags(w, ['east', 'south'])
    assert p.update(1.0, w.start, tags, remaining=wps) is None   # replan_interval not over
    wps2 = p.update(3.0, w.start, tags, remaining=wps)
    assert wps2 and p.history[-1]['reason'] == 'map'
    assert p.history[-1]['length'] < p.history[0]['length']     # smaller obstacle, shorter path


def test_planner_replans_at_once_when_the_path_is_blocked():
    w = world()
    p = SlamPlanner(w, 'fast')
    p.update(0.0, w.start, face_tags(w, ['east', 'south']))
    blocked = [(1.44, 1.6, float('nan')), w.goal]               # through the middle of the box
    assert p.update(0.5, w.start, face_tags(w, ['east', 'south']), remaining=blocked)
    assert p.history[-1]['reason'] == 'blocked'


def test_no_replanning_near_the_goal():
    w = world()
    p = SlamPlanner(w, 'safe')
    p.update(0.0, w.start, {})
    near_goal = (w.goal[0] + 0.1, w.goal[1], 0.0)
    assert p.update(10.0, near_goal, face_tags(w, ['east'])) is None


def test_prune_short_legs_merges_short_legs_without_losing_clearance():
    w = world()
    path = [(2.6, 0.4), (2.6, 1.0), (2.62, 1.05), (2.6, 2.6)]   # a 5 cm leg, away from the box
    out = prune_short_legs(w, path, 0.12)
    assert out[0] == path[0] and out[-1] == path[-1] and len(out) == 3
    assert min(math.dist(a, b) for a, b in zip(out[:-1], out[1:])) >= 0.12
    assert w.path_clearance(out)[0] >= w.path_clearance(path)[0] - 0.01


def test_prune_short_legs_keeps_a_vertex_whose_removal_costs_clearance():
    w = world()
    path = [(2.35, 0.65), (2.4, 2.4), (0.75, 2.4), (0.70, 2.35)]  # merging cuts towards the box
    assert prune_short_legs(w, path, 0.12) == path


def test_slam_starts_at_a_pose_and_takes_known_walls():
    w = world()
    w.slam = dict(w.slam, prior_landmarks='walls')
    slam = EKFSLAM(NoiseParams(), SLAMParams())
    slam.set_pose(w.start)
    added = seed_known_walls(slam, w, SlamConfig.from_world(w))
    assert slam.pose == pytest.approx(w.start)
    assert sorted(added) == [0, 1, 3, 4, 5, 7, 8, 9]            # box tags are never given
    assert len(slam.mu) == 3 + 3 * 8
    assert np.allclose(slam.sigma[3:6, 3:6], np.diag([0.02 ** 2, 0.02 ** 2, 0.05 ** 2]))
    assert np.allclose(slam.sigma[:3, 3:], 0.0)


def test_pass_through_waypoint_inside_replan_distance_does_not_spin():
    # the first vertex of a GVD plan can be < replan_distance from the start, in any direction
    c = WaypointController([(2.29, 0.71, float('nan')), (2.37, 1.09, math.pi / 2)], Gains())
    x, y, th = 2.35, 0.65, 2.3562
    for _ in range(400):
        v, w, _ = c.update(x, y, th, 0.05)
        assert math.isfinite(v) and math.isfinite(w)
        x += v * math.cos(th) * 0.05
        y += v * math.sin(th) * 0.05
        th += w * 0.05
        if c.index >= 1:
            break
    assert c.index >= 1


# ---- simulation ----------------------------------------------------------------------------
@pytest.mark.parametrize('mode', ['fast', 'safe'])
def test_simulated_slam_runs_map_the_box_and_reach_the_goal(mode):
    w = world()
    res = simulate(w, mode, load_params(os.path.join(CONFIG, 'hw4_slam.yaml')), seed=1)
    assert res['done']
    assert res['min_true_gap'] > 0.05
    assert math.dist(res['true'][:2], w.goal[:2]) < 0.10
    est = res['planner'].latest
    assert est.source == 'slam' and contains(est, true_bounds(w))
