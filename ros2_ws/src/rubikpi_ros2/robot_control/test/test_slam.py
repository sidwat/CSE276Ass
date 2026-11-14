import math
import os

import numpy as np
import pytest
from robot_control.evaluate_slam import best_fit, map_errors, Run, stats
from robot_control.localization import load_landmarks, NoiseParams
from robot_control.simulate_slam import PATHS, run_simulation, to_start_frame
from robot_control.slam import (
    EKFSLAM, incidence_angle, inverse_model, measurement_model, SLAMParams)
from robot_control.slam_logger import SLAMRunLogger
from robot_control.waypoint_controller import Gains, load_waypoints, wrap_angle

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')
GT = load_landmarks(os.path.join(CONFIG_DIR, 'landmarks_hw3_gt.txt'))
GAINS = Gains(pos_tolerance=0.03, heading_tolerance=0.05, max_v=0.15, max_w=1.0, min_w=0.4,
              settle_time=1.0, k_v=1.26, k_w=1.26)
NOISE = NoiseParams(odom_dist_std=0.1, odom_turn_std=0.2, odom_drift_std=0.05)


def numeric_jacobian(f, x, eps=1e-6):
    x = np.asarray(x, float)
    cols = []
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = eps
        diff = np.asarray(f(x + d)) - np.asarray(f(x - d))
        diff = np.array([wrap_angle(v) for v in diff])  # small differences: only fixes +-2 pi
        cols.append(diff / (2 * eps))
    return np.array(cols).T


@pytest.mark.parametrize('dof', [2, 3])
def test_jacobians_match_finite_differences(dof):
    robot = np.array([0.3, -0.2, 0.7])
    landmark = np.array([2.0, 1.1, -2.5])[:dof]
    _, H_r, H_m = measurement_model(robot, landmark, dof)
    assert H_r == pytest.approx(
        numeric_jacobian(lambda r: measurement_model(r, landmark, dof)[0], robot), abs=1e-6)
    assert H_m == pytest.approx(
        numeric_jacobian(lambda m: measurement_model(robot, m, dof)[0], landmark), abs=1e-6)

    z = np.array([1.7, 0.4, 2.8])[:dof]
    _, J_x, J_z = inverse_model(robot, z, dof)
    assert J_x == pytest.approx(
        numeric_jacobian(lambda r: inverse_model(r, z, dof)[0], robot), abs=1e-6)
    assert J_z == pytest.approx(
        numeric_jacobian(lambda q: inverse_model(robot, q, dof)[0], z), abs=1e-6)


def test_inverse_model_inverts_measurement_model():
    robot = (0.5, 0.2, 1.0)
    landmark = (2.0, 2.5, -1.2)
    z, _, _ = measurement_model(robot, landmark)
    m, _, _ = inverse_model(robot, z)
    assert m == pytest.approx(landmark)


def test_state_grows_only_after_confirmed_sightings():
    slam = EKFSLAM(params=SLAMParams(min_sightings=3))
    z = measurement_model((0, 0, 0), (2.0, 0.0, math.pi))[0]
    assert slam.observe(7, z, stamp=0.0).kind == 'candidate'
    assert slam.observe(7, z, stamp=0.1).kind == 'candidate'
    assert slam.n_landmarks == 0 and len(slam.mu) == 3
    assert slam.observe(7, z, stamp=0.2).kind == 'new'
    assert slam.n_landmarks == 1 and len(slam.mu) == 6
    assert slam.observe(7, z, stamp=0.3).kind == 'update'
    assert slam.landmarks()[7][0] == pytest.approx((2.0, 0.0, math.pi), abs=1e-6)


def test_single_spurious_sighting_never_creates_a_landmark():
    slam = EKFSLAM(params=SLAMParams(min_sightings=3, candidate_window=1.0))
    z = measurement_model((0, 0, 0), (2.0, 0.0, math.pi))[0]
    for t in (0.0, 5.0, 10.0):  # sightings too far apart in time
        assert slam.observe(3, z, stamp=t).kind == 'candidate'
    # sightings that disagree in position
    for i, r in enumerate((1.0, 2.0, 1.5)):
        assert slam.observe(4, (r, 0.0, math.pi), stamp=20.0 + 0.1 * i).kind == 'candidate'
    assert slam.n_landmarks == 0


def test_grazing_view_is_rejected():
    z = measurement_model((0, 0, 0), (2.0, 0.0, math.pi + math.radians(75)))[0]
    assert abs(math.degrees(incidence_angle(z))) == pytest.approx(75)
    slam = EKFSLAM()
    assert slam.observe(1, z, stamp=0.0).kind == 'incidence'


def test_outlier_is_gated_and_does_not_move_the_map():
    slam = EKFSLAM()
    z = measurement_model((0, 0, 0), (2.0, 0.0, math.pi))[0]
    for i in range(10):
        slam.observe(1, z, stamp=0.1 * i)
    before = slam.mu.copy()
    obs = slam.observe(1, (1.0, 0.5, math.pi), stamp=1.0)
    assert obs.kind == 'rejected' and obs.mahalanobis2 > slam.gate
    assert slam.mu == pytest.approx(before)


def test_reobservation_after_driving_shrinks_uncertainty():
    """Loop closure: seeing an old landmark again corrects the robot and correlated landmarks."""
    slam = EKFSLAM(NoiseParams(odom_dist_std=0.2, odom_turn_std=0.3))
    lm_a, lm_b = (2.0, 0.0, math.pi), (1.0, 2.0, -math.pi / 2)
    true = [0.0, 0.0, 0.0]
    for i in range(3):
        slam.observe('a', measurement_model(true, lm_a)[0], stamp=0.1 * i)
    for k in range(100):  # turn left 90 deg while a stays out of view
        slam.predict(0.0, math.pi / 2 / 5.0, 0.05, stamp=1.0 + 0.05 * k)
    true[2] = math.pi / 2
    for i in range(3):
        slam.observe('b', measurement_model(true, lm_b)[0], stamp=7.0 + 0.1 * i)
    for k in range(100):  # turn back
        slam.predict(0.0, -math.pi / 2 / 5.0, 0.05, stamp=8.0 + 0.05 * k)
    true[2] = 0.0
    b = slam.index['b']
    var_b = np.trace(slam.sigma[b:b + 2, b:b + 2])
    trace, std_theta = np.trace(slam.sigma), slam.std[2]
    assert slam.observe('a', measurement_model(true, lm_a)[0], stamp=14.0).kind == 'update'
    assert np.trace(slam.sigma) < trace
    assert slam.std[2] < 0.3 * std_theta
    assert np.trace(slam.sigma[b:b + 2, b:b + 2]) < var_b  # b improved without being seen


def test_covariance_stays_symmetric_positive_definite():
    result = simulate('square', seed=3)
    S = result['slam'].sigma
    assert S == pytest.approx(S.T)
    assert np.all(np.linalg.eigvalsh(S) > -1e-12)


def simulate(path, seed=0, dof=3, logger=None):
    wp_name, per_lap, start = PATHS[path]
    waypoints = load_waypoints(os.path.join(CONFIG_DIR, wp_name))
    return run_simulation(waypoints, GT, start, NOISE, SLAMParams(landmark_dof=dof), GAINS,
                          seed=seed, waypoints_per_lap=per_lap, logger=logger)


def start_frame_gt(start):
    return {k: to_start_frame(start, v) for k, v in GT.items()}


@pytest.mark.parametrize('path', ['square', 'octagon'])
def test_simulated_laps_map_all_tags_accurately(path):
    result = simulate(path)
    slam = result['slam']
    assert result['done']
    assert set(slam.index) == set(GT)  # discovered exactly the tags that exist
    truth = start_frame_gt(PATHS[path][2])
    errors = [math.hypot(m[0] - truth[k][0], m[1] - truth[k][1])
              for k, (m, _, _) in slam.landmarks().items()]
    assert np.mean(errors) < 0.10
    # The final pose is better than dead reckoning's
    true_end = to_start_frame(PATHS[path][2], result['truth'][-1][1])
    slam_err = math.hypot(slam.pose[0] - true_end[0], slam.pose[1] - true_end[1])
    odom_err = math.hypot(result['odom'][0] - true_end[0], result['odom'][1] - true_end[1])
    assert slam_err < odom_err
    assert slam_err < 0.1


def test_two_dof_landmarks_also_work():
    result = simulate('square', dof=2)
    slam = result['slam']
    assert set(slam.index) == set(GT)
    truth = start_frame_gt(PATHS['square'][2])
    errors = [math.hypot(m[0] - truth[k][0], m[1] - truth[k][1])
              for k, (m, _, _) in slam.landmarks().items()]
    assert np.mean(errors) < 0.15


def test_map_uncertainty_shrinks_over_laps():
    laps = simulate('square')['laps']
    assert len(laps) == 3
    areas = [np.mean([np.linalg.det(c) for _, c, _ in lm.values()]) for _, lm, _ in laps]
    assert areas[-1] < areas[0]


def test_filter_is_consistent_when_its_noise_matches_the_simulation():
    """Mean NIS near its dof (3) when R matches the simulated camera noise."""
    noise = NoiseParams(odom_dist_std=0.1, odom_turn_std=0.1, odom_drift_std=0.05,
                        range_std=0.01, range_std_per_m=0.02, bearing_std=0.01,
                        orientation_std=0.08)  # = simulate_slam's camera noise
    wp_name, per_lap, start = PATHS['square']
    waypoints = load_waypoints(os.path.join(CONFIG_DIR, wp_name))
    nis = []
    for seed in range(4):
        nis += run_simulation(waypoints, GT, start, noise, SLAMParams(latency_std=0.0), GAINS,
                              seed=seed, waypoints_per_lap=per_lap)['slam'].nis
    assert 2.0 < np.mean(nis) < 4.0


def test_best_fit_recovers_rigid_transform():
    a = np.array([[0, 0], [1, 0], [1, 2], [-1, 1]], float)
    theta, t = 0.3, np.array([0.5, -0.2])
    c, s = math.cos(theta), math.sin(theta)
    b = a @ np.array([[c, -s], [s, c]]).T + t
    th, tt, scale = best_fit(a, b, scale=True)
    assert th == pytest.approx(theta) and tt == pytest.approx(t) and scale == pytest.approx(1)


def test_logged_run_evaluates_end_to_end(tmp_path):
    path = 'square'
    start = PATHS[path][2]
    logger = SLAMRunLogger(str(tmp_path), {'path': path, 'waypoints_per_lap': 4,
                                           'params': {'landmark_dof': 3}},
                           os.path.join(CONFIG_DIR, PATHS[path][0]), start_pose=start,
                           run_name='run_test')
    simulate(path, logger=logger)
    run = Run(logger.run_dir)
    assert run.start == pytest.approx(start, abs=1e-4)
    assert run.lap_labels == ['lap0', 'lap1', 'lap2']
    assert set(run.final_map) == set(GT)
    for name in ('trajectory', 'measurements', 'landmark_events', 'waypoints'):
        assert os.path.getsize(os.path.join(logger.run_dir, name + '.csv')) > 100
    assert len(run.events) == len(GT)
    assert os.path.exists(os.path.join(logger.run_dir, 'state_final.npz'))
    raw = stats(map_errors(run.final_map, GT, run.start)[0])
    aligned = stats(map_errors(run.final_map, GT, run.start, align=True)[0])
    assert raw['n'] == len(GT) and raw['mean'] < 0.10
    assert aligned['mean'] <= raw['mean'] + 1e-9
