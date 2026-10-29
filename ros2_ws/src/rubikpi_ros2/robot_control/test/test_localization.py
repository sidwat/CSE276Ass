import math
import os
import random

import pytest
from robot_control.localization import load_landmarks, PoseEKF, tag_measurement
from robot_control.waypoint_controller import (
    Gains, load_waypoints, WaypointController, wrap_angle)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')


def observe(pose, landmark):
    """Exact (range, bearing, orientation) of a landmark seen from pose."""
    x, y, theta = pose
    lx, ly, lyaw = landmark
    return (math.hypot(lx - x, ly - y),
            wrap_angle(math.atan2(ly - y, lx - x) - theta),
            wrap_angle(lyaw - theta))


def visible(pose, landmark, half_fov=0.9, max_range=3.0):
    r, bearing, orientation = observe(pose, landmark)
    # The face must point back at the robot, within 70 degrees
    facing = abs(wrap_angle(orientation - (bearing + math.pi))) < math.radians(70)
    return r < max_range and abs(bearing) < half_fov and facing


def test_tag_measurement_from_optical_frame():
    # 2 m straight ahead, facing the camera; either sign of the tag z axis gives the same answer
    for normal in ((0, 0, -1), (0, 0, 1)):
        assert tag_measurement((0, 0, 2), normal, (0, 0, 0)) == pytest.approx((2, 0, math.pi))

    # 1 m ahead and 1 m left (optical -x)
    r, bearing, _ = tag_measurement((-1, 0.05, 1), (0, 0, -1), (0, 0, 0))
    assert (r, bearing) == pytest.approx((math.sqrt(2), math.pi / 4))

    # Camera 0.1 m ahead of the rotation centre adds to the range
    assert tag_measurement((0, 0, 2), (0, 0, -1), (0.1, 0, 0))[0] == pytest.approx(2.1)

    # Tag rotated 30 deg about the vertical (optical y) axis
    a = math.radians(30)
    normal = (math.sin(a), 0, -math.cos(a))
    orientation = tag_measurement((0, 0, 2), normal, (0, 0, 0))[2]
    assert orientation == pytest.approx(wrap_angle(math.pi + a))  # optical y is down: turns right


def test_ekf_corrects_drifted_pose_and_rejects_outliers():
    landmark = (2.0, 0.0, math.pi)
    ekf = PoseEKF()
    true_pose = (0.3, 0.1, 0.1)
    ekf.sigma *= 100  # the estimate (0, 0, 0) is badly off; let it know
    for _ in range(30):
        assert ekf.correct(observe(true_pose, landmark), landmark)[0]
    assert ekf.pose == pytest.approx(true_pose, abs=0.01)  # the prior keeps a little weight

    accepted, _ = ekf.correct((1.0, 0.5, math.pi), landmark)  # 1 m and 30 deg off
    assert not accepted
    assert ekf.pose == pytest.approx(true_pose, abs=0.01)


def test_ekf_uses_pose_at_image_time():
    landmark = (2.0, 0.0, math.pi)
    ekf = PoseEKF()
    ekf.predict(0.0, 0.0, 0.05, stamp=10.0)
    seen_at_10 = observe((0.0, 0.0, 0.0), landmark)
    ekf.predict(0.0, 1.0, 0.2, stamp=10.2)  # turned 0.2 rad since the image was taken
    accepted, _ = ekf.correct(seen_at_10, landmark, stamp=10.0)
    assert accepted
    assert ekf.pose[2] == pytest.approx(0.2, abs=1e-6)  # a late image does not undo the turn


def test_rotate_to_heading_drives_back_when_correction_moves_robot():
    controller = WaypointController([(1.0, 0.0, math.pi / 2)], Gains())
    controller.update(1.0, 0.0, 0.0)
    assert controller.phase == WaypointController.ROTATE_TO_HEADING
    controller.update(0.8, 0.0, 0.0)  # a tag sighting moved the estimate 20 cm back
    assert controller.phase != WaypointController.ROTATE_TO_HEADING


def run_course(use_vision, seed=1, dt=0.05):
    """Drive the HW2 course with miscalibrated odometry; return (true, goal) at each waypoint."""
    rng = random.Random(seed)
    waypoints = load_waypoints(os.path.join(CONFIG_DIR, 'waypoints_hw2.txt'))
    landmarks = load_landmarks(os.path.join(CONFIG_DIR, 'landmarks.txt'))
    gains = Gains(pos_tolerance=0.03, heading_tolerance=0.03, max_v=0.25, max_w=0.8)
    controller = WaypointController(waypoints, gains)
    ekf = PoseEKF()
    true = [0.0, 0.0, 0.0]
    results = []
    for step in range(20000):
        v, w, reached = controller.update(*ekf.pose)
        if reached is not None:
            results.append((tuple(true), waypoints[reached]))
        if controller.done:
            return results
        # The real robot goes 10% faster, turns 8% less and veers left, compared to the model
        tv, tw = 1.1 * v, 0.92 * w + 0.03 * v
        true[0] += tv * math.cos(true[2]) * dt
        true[1] += tv * math.sin(true[2]) * dt
        true[2] = wrap_angle(true[2] + tw * dt)
        ekf.predict(v, w, dt, stamp=step * dt)
        if use_vision and step % 2 == 0:  # 10 Hz camera
            for landmark in landmarks.values():
                if visible(true, landmark):
                    r, b, o = observe(true, landmark)
                    noisy = (r * (1 + rng.gauss(0, 0.02)), b + rng.gauss(0, 0.01),
                             o + rng.gauss(0, 0.1))
                    ekf.correct(noisy, landmark, stamp=step * dt)
    pytest.fail('controller did not finish the course')


def test_vision_keeps_waypoint_error_small_where_dead_reckoning_drifts():
    blind = run_course(use_vision=False)
    final_blind = math.hypot(blind[-1][0][0], blind[-1][0][1])
    assert final_blind > 0.2  # the miscalibration really does matter without the camera

    for (x, y, theta), (gx, gy, gt) in run_course(use_vision=True):
        assert math.hypot(x - gx, y - gy) < 0.06
        # No tag is in view while turning to pi at (1,2), so the 8% turn error shows there
        assert abs(wrap_angle(theta - gt)) < math.radians(8)
