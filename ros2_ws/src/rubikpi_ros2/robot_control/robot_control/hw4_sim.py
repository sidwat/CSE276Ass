"""
Simulate an HW4 run offline (no ROS): the same controller, EKF and log files as the robot.

The simulated robot is miscalibrated (drives a few % too far, turns too little, veers), starts a
little off its nominal pose, and the camera is noisy, has a limited field of view and range, and
cannot see through the box. velocity_control's command path (saturation, dead-band, pause at
stop waypoints, heading confirmation, SEARCH inflation) is reproduced so the logs look like a
real run's and `hw4 eval` reads both the same way.
"""

import csv
import math
import os
import random

from robot_control.hw4_world import visible
from robot_control.localization import NoiseParams, PoseEKF
from robot_control.waypoint_controller import (
    body_to_wheel_cmds, Gains, WaypointController, wheel_cmds_to_body, wrap_angle)


def load_params_yaml(path):
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return next(iter(data.values())).get('ros__parameters', {})


def pick(cls, params):
    return cls(**{k: type(v)(params[k]) for k, v in vars(cls()).items() if k in params})


def simulate(world, waypoints, params=None, seed=0, dt=0.05, start_delay=2.0,
             camera_period=0.1, outlier_prob=0.01, v_scale=1.04, w_scale=0.93, veer=0.02,
             start_error=(0.01, 0.01, 0.03), max_time=300.0):
    """
    Drive the waypoints (world frame) from world.start; return logs and summary numbers.

    params is the ros__parameters dict of hw4.yaml (controller gains, EKF noise, calibration).
    """
    p = params or {}
    rng = random.Random(seed)
    gains = pick(Gains, p) if p else Gains()
    noise = pick(NoiseParams, p) if p else NoiseParams()
    speed_per_unit = float(p.get('speed_per_unit', 1.26))
    wheel_base = float(p.get('wheel_base', 0.68))
    max_cmd = float(p.get('max_wheel_cmd', 0.5))
    min_cmd = float(p.get('min_wheel_cmd', 0.15))
    pause = float(p.get('pause_at_waypoint', 0.5))
    log_period = float(p.get('log_period', 0.1))
    sxy = float(p.get('start_std_xy', 0.02))
    sth = float(p.get('start_std_theta', 0.05))
    cam = world.camera

    x0, y0, t0 = world.start
    true = [x0 + rng.gauss(0, start_error[0]), y0 + rng.gauss(0, start_error[1]),
            wrap_angle(t0 + rng.gauss(0, start_error[2]))]
    ekf = PoseEKF(x0, y0, t0, initial_std=(sxy, sxy, sth), noise=noise)
    odom = [x0, y0, t0]
    controller = WaypointController(waypoints, gains)
    v = w = 0.0
    hold_until = start_delay
    last_fix = -math.inf
    next_camera = 0.0
    last_log = -math.inf
    last_phase = None
    tags_seen = 0
    min_true_clearance = math.inf
    traj, truth, reached_rows, meas_rows = [], [], [], []
    t = 0.0
    step = 0
    while t < max_time:
        t = step * dt
        step += 1
        # true motion: not quite what was commanded
        tv = v * v_scale * (1 + rng.gauss(0, 0.02))
        tw = w * w_scale + veer * v + rng.gauss(0, 0.02) * abs(w)
        h = true[2] + tw * dt / 2.0
        true[0] += tv * math.cos(h) * dt
        true[1] += tv * math.sin(h) * dt
        true[2] = wrap_angle(true[2] + tw * dt)
        h = odom[2] + w * dt / 2.0
        odom = [odom[0] + v * math.cos(h) * dt, odom[1] + v * math.sin(h) * dt,
                wrap_angle(odom[2] + w * dt)]
        ekf.predict(v, w, dt, stamp=t)
        min_true_clearance = min(min_true_clearance, float(world.clearance(true[0], true[1])))

        if t >= next_camera:
            next_camera = t + camera_period
            for tag, lm in sorted(world.landmarks.items()):
                z = visible(world, true, lm, cam['half_fov'], cam['max_range'],
                            cam['max_facing'])
                if z is None:
                    continue
                r, b, o = z
                if rng.random() < outlier_prob:
                    z = (r * rng.uniform(0.6, 1.4), b + rng.uniform(-0.3, 0.3),
                         o + rng.uniform(-1.0, 1.0))
                else:
                    z = (r + rng.gauss(0, 0.01 + 0.02 * r), b + rng.gauss(0, 0.01),
                         o + rng.gauss(0, 0.08))
                pose = ekf.pose
                accepted, d2 = ekf.correct(z, lm, stamp=t)
                meas_rows.append([t, tag, *z, int(accepted), d2, *pose])
                if accepted:
                    tags_seen += 1
                    last_fix = t

        if t - last_log >= log_period:
            last_log = t
            ex, ey, eth = ekf.pose
            sx, sy, st = ekf.std
            traj.append([t, ex, ey, eth, sx, sy, st, controller.index, tags_seen,
                         *odom, ex, ey, eth, v, w])
            truth.append([t, *true])
            tags_seen = 0

        if t < hold_until:
            v = w = 0.0
            continue
        if controller.done:
            break
        confirmed = t - last_fix <= max(gains.settle_time, 0.5)
        ex, ey, eth = ekf.pose
        cv, cw, reached = controller.update(ex, ey, eth, dt, heading_confirmed=confirmed)
        if controller.phase == WaypointController.SEARCH and last_phase != controller.phase:
            ekf.inflate((0.0, 0.0, 1.0))  # as velocity_control does without SLAM
        last_phase = controller.phase
        if reached is not None:
            wx, wy, wt = waypoints[reached]
            reached_rows.append([reached, t, wx, wy, wt, ex, ey, eth])
            if not controller.is_pass_through(reached):
                hold_until = t + pause
        left, right = body_to_wheel_cmds(cv, cw, speed_per_unit, wheel_base, max_cmd)
        largest = max(abs(left), abs(right))
        if 1e-6 < largest < min_cmd:
            left *= min_cmd / largest
            right *= min_cmd / largest
        v, w = wheel_cmds_to_body(left, right, speed_per_unit, wheel_base)

    ex, ey, eth = ekf.pose
    return {
        'trajectory': traj, 'truth': truth, 'reached': reached_rows, 'measurements': meas_rows,
        'done': controller.done, 'time': t, 'true_final': tuple(true),
        'est_final': (ex, ey, eth), 'min_true_clearance': min_true_clearance,
        'collided': min_true_clearance < world.robot_radius,
    }


def write_sim_logs(run_dir, result):
    """Write the run in velocity_control's CSV formats, plus truth.csv."""
    os.makedirs(run_dir, exist_ok=True)

    def fmt(row):
        return [f'{v:.4f}' if isinstance(v, float) else v for v in row]

    with open(os.path.join(run_dir, 'trajectory.csv'), 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['t', 'x', 'y', 'theta', 'std_x', 'std_y', 'std_theta', 'waypoint',
                      'tags_seen', 'odom_x', 'odom_y', 'odom_theta', 'pred_x', 'pred_y',
                      'pred_theta', 'v', 'w'])
        out.writerows(fmt(r) for r in result['trajectory'])
    with open(os.path.join(run_dir, 'waypoints.csv'), 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['index', 't', 'goal_x', 'goal_y', 'goal_theta', 'x', 'y', 'theta'])
        out.writerows(fmt(r) for r in result['reached'])
    with open(os.path.join(run_dir, 'measurements.csv'), 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['t', 'tag', 'range', 'bearing', 'orientation', 'accepted',
                      'mahalanobis2', 'x', 'y', 'theta'])
        out.writerows(fmt(r) for r in result['measurements'])
    with open(os.path.join(run_dir, 'truth.csv'), 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['t', 'x', 'y', 'theta'])
        out.writerows(fmt(r) for r in result['truth'])
