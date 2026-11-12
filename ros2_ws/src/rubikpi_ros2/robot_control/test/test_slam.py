import math
import os
import random

import numpy as np
import pytest
from robot_control.localization import NoiseParams
from robot_control.slam import (
    EKFSLAM, inverse_measurement, is_lap_end, measurement_model, SLAMParams)
from robot_control.slam_sim import (
    arena_landmarks, centred_start, FT, octagon_waypoints, SimConfig, simulate, square_waypoints)
from robot_control.waypoint_controller import load_waypoints, wrap_angle

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')

# Odometry noise the filter assumes in the closed-loop runs (the simulated robot is better than
# this, plus a systematic error the filter does not know about)
ROBOT_NOISE = NoiseParams(odom_dist_std=0.1, odom_turn_std=0.3, odom_drift_std=0.05)


def run_square(laps=3, dof=3, seed=1, **cfg):
    size = 4 * FT
    return simulate(square_waypoints(size, laps), arena_landmarks(), centred_start('square', size),
                    noise=ROBOT_NOISE, params=SLAMParams(landmark_dof=dof),
                    cfg=SimConfig(seed=seed, **cfg))


def numeric_jacobian(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    cols = []
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = eps
        diff = np.asarray(f(x + d)) - np.asarray(f(x - d))
        cols.append(np.array([wrap_angle(v) for v in diff]) / (2 * eps))
    return np.array(cols).T


@pytest.mark.parametrize('dof', [2, 3])
def test_jacobians_match_finite_differences(dof):
    robot = np.array([0.3, -0.2, 0.7])
    landmark = np.array([1.9, 1.1, -2.5])[:dof]
    _, H = measurement_model(robot, landmark, dof)
    numeric = numeric_jacobian(
        lambda s: measurement_model(s[:3], s[3:], dof)[0], np.r_[robot, landmark])
    assert H == pytest.approx(numeric, abs=1e-6)

    z = np.array([1.7, 0.4, 2.9])[:dof]
    _, Jx, Jz = inverse_measurement(robot, z, dof)
    assert Jx == pytest.approx(
        numeric_jacobian(lambda r: inverse_measurement(r, z, dof)[0], robot), abs=1e-6)
    assert Jz == pytest.approx(
        numeric_jacobian(lambda zz: inverse_measurement(robot, zz, dof)[0], z), abs=1e-6)

    # g inverts h: the landmark implied by the predicted measurement is the landmark
    h, _ = measurement_model(robot, landmark, dof)
    assert inverse_measurement(robot, h, dof)[0] == pytest.approx(landmark)


def test_state_grows_only_with_tags_actually_seen():
    landmarks = arena_landmarks()
    landmarks[99] = (6.0, 1.5, math.pi)  # beyond camera range from everywhere on the path
    size = 4 * FT
    result = simulate(square_waypoints(size, 1), landmarks, centred_start('square', size),
                      noise=ROBOT_NOISE)
    slam = result.slam
    # N starts at 0 and grows one landmark per 'added' event, each a different id
    added = [tag for _, tag, _, obs, _, _ in result.events if obs.event == 'added']
    assert len(added) == len(set(added)) == slam.n_landmarks
    assert [n for _, n in result.n_history] == sorted(n for _, n in result.n_history)
    assert list(slam.index) == added
    assert set(slam.index) == result.seen_ids == set(arena_landmarks())
    assert 99 not in slam.index
    assert len(slam.mu) == 3 + 3 * slam.n_landmarks
    assert slam.sigma.shape == (len(slam.mu),) * 2


@pytest.mark.parametrize('dof, tolerance', [(3, 0.05), (2, 0.05)])
def test_three_square_laps_map_better_than_dead_reckoning(dof, tolerance):
    result = run_square(laps=3, dof=dof)
    errors = result.landmark_errors()
    dead_reckoning = result.dead_reckoning_errors()
    assert len(errors) == 8
    assert max(errors.values()) < 2 * tolerance
    assert np.mean(list(errors.values())) < tolerance
    assert np.mean(list(errors.values())) < np.mean(list(dead_reckoning.values())) / 3
    assert [lap for lap, *_ in result.snapshots] == [1, 2, 3]


def test_octagon_map():
    size = 2 * FT
    result = simulate(octagon_waypoints(size, 2), arena_landmarks(),
                      centred_start('octagon', size), noise=ROBOT_NOISE)
    errors = result.landmark_errors()
    assert len(errors) == 8
    assert np.mean(list(errors.values())) < 0.05


def test_reobservation_shrinks_covariance():
    """Drive away (uncertainty grows), then see the first landmark again: a loop closure."""
    slam = EKFSLAM(NoiseParams(odom_dist_std=0.1, odom_turn_std=0.2, odom_drift_std=0.05))
    a = (2.0, 0.0, math.pi)
    b = (1.0, 3.0, -math.pi / 2)

    def sight(pose, lm):
        h, _ = measurement_model(pose, lm)
        return h

    pose = np.zeros(3)
    for _ in range(3):
        slam.observe(1, sight(pose, a))
    assert slam.n_landmarks == 1
    a_cov_start = slam.landmarks()[1][1][:2, :2].copy()
    # Drive a loop out of sight of tag 1, seeing tag 2 on the way
    for v, w, steps in ((0.2, 0.0, 40), (0.0, 1.0, 16), (0.2, 0.0, 40), (0.0, 1.0, 16)):
        for _ in range(steps):
            slam.predict(v, w, 0.1)
            pose = slam.mu[:3].copy()  # noise-free: the estimate is the truth
        if slam.n_landmarks < 2:
            for _ in range(3):
                slam.observe(2, sight(pose, b))
    assert slam.n_landmarks == 2
    trace_before = np.trace(slam.sigma)
    robot_before = np.trace(slam.sigma[:3, :3])
    b_before = np.trace(slam.landmarks()[2][1][:2, :2])

    event = slam.observe(1, sight(pose, a))
    assert event.event == 'updated'
    assert np.trace(slam.sigma) < trace_before
    assert np.trace(slam.sigma[:3, :3]) < 0.5 * robot_before
    assert np.trace(slam.landmarks()[2][1][:2, :2]) < b_before  # via the correlations
    assert np.trace(slam.landmarks()[1][1][:2, :2]) <= np.trace(a_cov_start) + 1e-12


def test_filter_is_consistent():
    """Simulate with the filter's own noise model: mean NEES and NIS are near their DOF."""
    true = {'odom_dist_std': 0.05, 'odom_turn_std': 0.1, 'odom_drift_std': 0.03,
            'range_std': 0.02, 'range_std_per_m': 0.02, 'bearing_std': 0.02,
            'orientation_std': 0.1}
    noise = NoiseParams(**true)
    size = 4 * FT
    robot_nees, landmark_nees, nis = [], [], []
    for seed in range(20):
        result = simulate(square_waypoints(size, 1), arena_landmarks(),
                          centred_start('square', size), noise=noise,
                          params=SLAMParams(latency=0.0),
                          cfg=SimConfig(seed=seed, v_scale=1.0, w_scale=1.0, veer=0.0, **true))
        slam, (_, tx, ty, tth) = result.slam, result.truth[-1]
        e = np.array([tx - slam.mu[0], ty - slam.mu[1], wrap_angle(tth - slam.mu[2])])
        robot_nees.append(e @ np.linalg.solve(slam.sigma[:3, :3], e))
        for tag, (m, cov) in slam.landmarks().items():
            e = m[:2] - np.array(result.landmarks[tag][:2])
            landmark_nees.append(e @ np.linalg.solve(cov[:2, :2], e))
        nis += [obs.d2 for _, _, _, obs, _, _ in result.events
                if obs.event in ('updated', 'rejected')]
    assert 1.2 < np.mean(robot_nees) < 4.5     # 3 DOF
    assert 0.8 < np.mean(landmark_nees) < 3.0  # 2 DOF
    assert 2.6 < np.mean(nis) < 3.4            # 3 DOF


def test_outliers_are_gated_and_spurious_tags_never_added():
    slam = EKFSLAM()
    lm = (2.0, 0.5, math.pi)
    h, _ = measurement_model((0.0, 0.0, 0.0), lm)
    for _ in range(3):
        slam.observe(5, h)
    assert slam.n_landmarks == 1
    mu, sigma = slam.mu.copy(), slam.sigma.copy()

    obs = slam.observe(5, (h[0] + 0.8, h[1] + 0.4, h[2]))  # 80 cm and 23 deg off
    assert obs.event == 'rejected'
    assert np.array_equal(slam.mu, mu) and np.array_equal(slam.sigma, sigma)

    # One sighting of an unknown id: buffered, not added
    assert slam.observe(42, (1.5, 0.2, math.pi + 0.2)).event == 'candidate'
    assert slam.n_landmarks == 1
    # A phantom id seen repeatedly at random places never agrees with itself
    rng = random.Random(0)
    for _ in range(50):
        z = (rng.uniform(0.5, 2.5), rng.uniform(-0.7, 0.7), 0.0)
        z = (z[0], z[1], wrap_angle(z[1] + math.pi))
        assert slam.observe(77, z).event == 'candidate'
    assert slam.n_landmarks == 1
    # Grazing views (> 60 deg off the tag normal) are ignored
    assert slam.observe(6, (1.0, 0.0, math.pi + 1.2)).event == 'skipped'

    # In the closed loop, garbage sightings neither break the map nor add landmarks
    result = run_square(laps=1, outlier_rate=0.05, seed=3)
    assert set(result.slam.index) == set(arena_landmarks())
    assert np.mean(list(result.landmark_errors().values())) < 0.05


def test_late_image_with_several_tags_is_not_double_counted():
    """Two tags in one late frame: the second must see the first one's correction."""
    noise = NoiseParams(odom_dist_std=0.1, odom_turn_std=0.1, odom_drift_std=0.05,
                        range_std=0.001, range_std_per_m=0.0, bearing_std=0.001,
                        orientation_std=0.01)
    params = SLAMParams(initial_std=1e-6, latency=0.0)
    a, b = (2.0, 0.3, math.pi), (2.0, -0.3, math.pi)
    slam = EKFSLAM(noise, params)
    for tag, lm in ((1, a), (2, b)):
        for _ in range(3):
            slam.observe(tag, measurement_model((0.0, 0.0, 0.0), lm)[0], stamp=0.0)
    true = np.array([0.5, 0.0, 0.0])
    for i in range(1, 51):      # drives 0.5 m, but the estimate says 0.55 m (10% over)
        slam.predict(0.11, 0.0, 0.1, stamp=0.1 * i)
    stamp = 4.81                # image from 0.2 s ago; the robot moved 2.2 cm since
    true_at_stamp = true - [0.02, 0.0, 0.0]
    for tag, lm in ((1, a), (2, b)):
        z = measurement_model(true_at_stamp, lm)[0]
        assert slam.observe(tag, z, stamp=stamp).event == 'updated'
    assert slam.pose[0] == pytest.approx(true[0], abs=0.005)
    assert slam.pose_at(stamp)[0] == pytest.approx(true_at_stamp[0], abs=0.005)


def test_waypoint_files_and_laps():
    square = load_waypoints(os.path.join(CONFIG_DIR, 'waypoints_hw3_square.txt'))
    octagon = load_waypoints(os.path.join(CONFIG_DIR, 'waypoints_hw3_octagon.txt'))
    assert np.array(square) == pytest.approx(np.array(square_waypoints(4 * FT, 3)), abs=1e-3)
    assert np.array(octagon) == pytest.approx(np.array(octagon_waypoints(2 * FT, 3)), abs=1e-3)
    for (x0, y0, _), (x1, y1, t1) in zip(octagon, octagon[1:]):
        assert math.hypot(x1 - x0, y1 - y0) == pytest.approx(2 * FT, abs=1e-3)
    laps = [i for i, wp in enumerate(square) if is_lap_end(i, wp)]
    assert laps == [3, 7, 11]
    assert [i for i, wp in enumerate(octagon) if is_lap_end(i, wp)] == [7, 15, 23]
    assert [i for i, wp in enumerate(octagon) if is_lap_end(i, wp, 8)] == [7, 15, 23]
