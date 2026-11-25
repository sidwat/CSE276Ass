import math
import os

import pytest
from robot_control.waypoint_controller import (
    body_to_wheel_cmds, Gains, load_waypoints, WaypointController, wheel_cmds_to_body, wrap_angle)

WAYPOINTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'config', 'waypoints.txt')


def simulate(waypoints, dt=0.05, max_steps=20000):
    """Run the controller against ideal unicycle kinematics; return the pose at each waypoint."""
    controller = WaypointController(waypoints)
    x = y = theta = 0.0
    reached_poses = []
    for _ in range(max_steps):
        v, w, reached = controller.update(x, y, theta)
        if reached is not None:
            reached_poses.append((x, y, theta))
        if controller.done:
            return controller, reached_poses
        x += v * math.cos(theta + w * dt / 2) * dt
        y += v * math.sin(theta + w * dt / 2) * dt
        theta = wrap_angle(theta + w * dt)
    pytest.fail(f'controller did not finish; stuck at waypoint {controller.index}')


def test_assignment_waypoints_are_all_reached():
    waypoints = load_waypoints(WAYPOINTS_FILE)
    controller, reached_poses = simulate(waypoints)
    assert len(reached_poses) == len(waypoints)
    for (wx, wy, wt), (x, y, theta) in zip(waypoints, reached_poses):
        assert math.hypot(wx - x, wy - y) < controller.gains.pos_tolerance
        assert abs(wrap_angle(wt - theta)) < controller.gains.heading_tolerance


def test_load_waypoints_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / 'wp.txt'
    path.write_text('# x, y, theta\n0,0,0\n\n1.5, -2, 3.14  # last\n')
    assert load_waypoints(str(path)) == [(0.0, 0.0, 0.0), (1.5, -2.0, 3.14)]


def test_wheel_conversion_round_trips_and_saturates_proportionally():
    left, right = body_to_wheel_cmds(0.1, 0.5, speed_per_unit=1.0, wheel_base=0.2, max_cmd=0.5)
    assert wheel_cmds_to_body(left, right, 1.0, 0.2) == pytest.approx((0.1, 0.5))

    left, right = body_to_wheel_cmds(1.0, 2.0, speed_per_unit=1.0, wheel_base=0.2, max_cmd=0.5)
    assert max(abs(left), abs(right)) == pytest.approx(0.5)
    assert right / left == pytest.approx(1.2 / 0.8)  # turning radius preserved


def test_settle_waits_and_turns_again_if_the_heading_is_corrected():
    gains = Gains(heading_tolerance=0.05, settle_time=1.0)
    controller = WaypointController([(0.0, 0.0, 0.0)], gains)
    assert controller.update(0.0, 0.0, 0.01, dt=0.1) == (0.0, 0.0, None)  # settling, not done
    assert controller.phase == WaypointController.SETTLE

    v, w, reached = controller.update(0.0, 0.0, 0.3, dt=0.1)  # camera: still 0.3 rad off
    assert reached is None and v == 0.0 and w < 0.0  # turns back

    for _ in range(30):
        _, _, reached = controller.update(0.0, 0.0, 0.0, dt=0.1)
        if reached is not None:
            break
    assert reached == 0 and controller.done


def test_unconfirmed_heading_searches_in_the_last_turn_direction_until_a_tag_is_seen():
    gains = Gains(heading_tolerance=0.05, settle_time=0.5, min_w=1.0,
                  search_turn_time=0.3, search_look_time=1.0)
    controller = WaypointController([(0.0, 0.0, 1.0)], gains)
    assert controller.update(0.0, 0.0, 0.0, dt=0.1, heading_confirmed=False)[1] > 0  # turn left
    for _ in range(10):  # the estimate says it got there, but no tag has confirmed it
        _, w, reached = controller.update(0.0, 0.0, 1.0, dt=0.1, heading_confirmed=False)
        assert reached is None
        if controller.phase == WaypointController.SEARCH:
            break
    assert controller.phase == WaypointController.SEARCH
    assert w > 0  # keeps turning left, the way it was turning

    # A tag is seen: the corrected estimate is checked again, then the waypoint counts
    for _ in range(10):
        _, _, reached = controller.update(0.0, 0.0, 1.0, dt=0.1, heading_confirmed=True)
        if reached is not None:
            break
    assert reached == 0


def test_near_the_waypoint_it_drives_straight_instead_of_turning_around():
    controller = WaypointController([(1.0, 0.0, 0.0)], Gains(pos_tolerance=0.03))
    controller.phase = WaypointController.DRIVE
    # A correction puts the estimate 4 cm short and 4 cm to the side: no turn in place
    v, w, _ = controller.update(0.96, 0.04, 0.0)
    assert v > 0 and w == 0.0
    # Once it is no longer ahead, the 4 cm sideways error is accepted (heading already right)
    assert controller.update(1.0, 0.04, 0.0) == (0.0, 0.0, 0)


def run_phases(waypoints, gains=None, dt=0.05, max_steps=20000):
    """Like simulate(), with dt passed on, recording (index, phase) at every step."""
    controller = WaypointController(waypoints, gains or Gains(settle_time=0.5))
    x = y = theta = 0.0
    trace, reached_poses = [], []
    for _ in range(max_steps):
        trace.append((controller.index, controller.phase))
        v, w, reached = controller.update(x, y, theta, dt)
        if reached is not None:
            reached_poses.append((reached, x, y, theta))
        if controller.done:
            return controller, trace, reached_poses
        x += v * math.cos(theta + w * dt / 2) * dt
        y += v * math.sin(theta + w * dt / 2) * dt
        theta = wrap_angle(theta + w * dt)
    pytest.fail(f'controller did not finish; stuck at waypoint {controller.index}')


def test_pass_through_waypoint_never_stops_and_the_last_one_settles():
    waypoints = [(1.0, 0.0, math.nan), (1.0, 1.0, math.pi / 2)]
    controller, trace, reached = run_phases(waypoints)
    phases_at_0 = {p for i, p in trace if i == 0}
    assert WaypointController.SETTLE not in phases_at_0
    assert WaypointController.ROTATE_TO_HEADING not in phases_at_0
    assert WaypointController.SETTLE in {p for i, p in trace if i == 1}
    assert [r[0] for r in reached] == [0, 1]
    _, x, y, _ = reached[0]
    assert math.hypot(x - 1.0, y) < max(controller.gains.pass_radius,
                                        controller.gains.replan_distance)
    _, x, y, theta = reached[1]
    assert math.hypot(x - 1.0, y - 1.0) < controller.gains.pos_tolerance + 1e-6
    assert abs(wrap_angle(theta - math.pi / 2)) < controller.gains.heading_tolerance


def test_straight_pass_through_keeps_driving():
    waypoints = [(0.5, 0.0, math.nan), (1.0, 0.0, 0.0)]
    _, trace, _ = run_phases(waypoints)
    # no turning in place at all: after the first DRIVE only DRIVE until the final waypoint
    first_drive = next(k for k, (_, p) in enumerate(trace) if p == WaypointController.DRIVE)
    middle = [p for i, p in trace[first_drive:] if i == 0]
    assert set(middle) == {WaypointController.DRIVE}


def test_final_waypoint_without_heading_stops_where_it_is():
    controller, trace, reached = run_phases([(0.0, 0.8, math.nan)])
    assert WaypointController.SETTLE in {p for _, p in trace}
    _, x, y, theta = reached[0]
    assert math.hypot(x, y - 0.8) < controller.gains.pos_tolerance + 1e-6
    assert abs(wrap_angle(theta - math.pi / 2)) < 0.2  # still facing along the leg


def test_load_waypoints_reads_nan_and_dash_as_no_heading(tmp_path):
    path = tmp_path / 'wp.txt'
    path.write_text('1, 2, nan\n3, 4, -\n5, 6, 0.5\n')
    wps = load_waypoints(str(path))
    assert math.isnan(wps[0][2]) and math.isnan(wps[1][2]) and wps[2] == (5.0, 6.0, 0.5)
