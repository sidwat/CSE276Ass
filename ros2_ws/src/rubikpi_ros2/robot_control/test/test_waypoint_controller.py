import math
import os

import pytest
from robot_control.waypoint_controller import (
    apply_wheel_trim, body_to_wheel_cmds, load_waypoints, WaypointController, wheel_cmds_to_body,
    wrap_angle)

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


def test_wheel_trim_only_changes_forward_motion():
    # Turning in place is untouched
    assert apply_wheel_trim(-0.3, 0.3, 0.06) == pytest.approx((-0.3, 0.3))
    # Driving straight shifts speed from the left wheel to the right
    assert apply_wheel_trim(0.4, 0.4, 0.06) == pytest.approx((0.376, 0.424))
    # Driving while steering gets the same correction as driving straight at that forward speed
    left, right = apply_wheel_trim(0.3, 0.5, 0.06)
    assert (left - 0.3, right - 0.5) == pytest.approx((0.376 - 0.4, 0.424 - 0.4))
