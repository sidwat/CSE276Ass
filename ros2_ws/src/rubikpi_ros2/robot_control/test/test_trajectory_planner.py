import math
import os

import numpy as np
import pytest
from robot_control.simulate_trajectory import simulate_open_loop
from robot_control.trajectory_planner import plan_trajectory, PlannerConfig
from robot_control.waypoint_controller import load_waypoints, wrap_angle

WAYPOINTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'config', 'waypoints.txt')


@pytest.fixture(scope='module')
def plan():
    waypoints = load_waypoints(WAYPOINTS_FILE)
    config = PlannerConfig()
    return waypoints, config, plan_trajectory(waypoints, config)


def test_meets_every_waypoint_pose_exactly(plan):
    waypoints, _, traj = plan
    assert len(traj.waypoint_times) == len(waypoints)
    for (wx, wy, wt), t in zip(waypoints, traj.waypoint_times):
        s = traj.sample(min(t, traj.duration))
        assert math.hypot(s.x - wx, s.y - wy) < 0.005
        assert abs(wrap_angle(s.heading - wt)) < 0.01


def test_starts_and_ends_at_rest(plan):
    _, _, traj = plan
    assert traj.sample(0.0).v == pytest.approx(0.0, abs=1e-9)
    assert traj.sample(traj.duration).v == pytest.approx(0.0, abs=1e-9)


def test_aligned_headings_give_a_straight_segment(plan):
    _, _, traj = plan
    assert traj.report.straight[0]
    assert traj.report.max_deviation[0] < 1e-6


def test_direction_changes_only_where_forced(plan):
    _, _, traj = plan
    assert traj.report.directions == [1, 1, 1, -1, 1, -1]
    assert traj.report.stops == [3, 4, 5]
    for t in traj.waypoint_times[1:3]:
        assert abs(traj.sample(t).v) == pytest.approx(0.25)


def test_constraints_hold(plan):
    _, config, traj = plan
    report = traj.report
    assert max(report.max_deviation) <= config.corridor_width + 1e-3
    assert report.max_curvature <= report.curvature_limit * 1.02
    assert not report.violations


def test_speed_is_constant_between_ramps(plan):
    _, config, traj = plan
    for leg in traj.legs:
        ts = leg.start_time + np.linspace(config.ramp_time, leg.duration - config.ramp_time, 200)
        speeds = [abs(traj.sample(t).v) for t in ts]
        assert speeds == pytest.approx([leg.peak_speed] * len(ts), abs=1e-6)
    assert traj.legs[0].peak_speed == pytest.approx(config.cruise_speed)


def test_heading_and_curvature_are_continuous(plan):
    _, _, traj = plan
    for leg in traj.legs:
        ts = np.linspace(leg.start_time, leg.start_time + leg.duration, 4000)[1:-1]
        samples = [traj.sample(t) for t in ts]
        headings = np.unwrap([s.heading for s in samples])
        curvature = np.array([s.curvature for s in samples])
        assert np.max(np.abs(np.diff(headings))) < 0.05
        assert np.max(np.abs(np.diff(curvature))) < 0.3


def test_wheel_commands_reproduce_the_plan(plan):
    waypoints, config, traj = plan
    _, poses, cmds = simulate_open_loop(traj, config, dt=0.05)
    assert np.max(np.abs(cmds)) <= config.max_wheel_cmd + 1e-9
    assert math.hypot(*(poses[-1, :2] - np.array(waypoints[-1][:2]))) < 0.03


def test_smooth_waypoints_need_no_stops():
    config = PlannerConfig(arena_min_x=-1.0, arena_min_y=-1.0, arena_size=4.0)
    traj = plan_trajectory([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.5, math.pi / 2)], config)
    assert len(traj.legs) == 1
    assert traj.report.straight == [True, False]
    assert not traj.report.violations
    end = traj.sample(traj.duration)
    assert math.hypot(end.x - 1.5, end.y - 0.5) < 0.005
    assert abs(wrap_angle(end.heading - math.pi / 2)) < 0.01
