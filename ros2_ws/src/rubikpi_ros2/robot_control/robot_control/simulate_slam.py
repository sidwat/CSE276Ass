"""
Simulate an HW3 SLAM run offline (no ROS): same controller, filter and log files as the robot.

Use it to rehearse the whole pipeline before the real runs, e.g.

  ros2 run robot_control simulate_slam --path square      # -> <workspace>/hw3_logs/sim_square_...
  ros2 run robot_control hw3 eval latest-sim       # or in one go: hw3 sim square

The simulated robot is miscalibrated (drives 5% too far, turns 7% too little, veers) and the
camera is noisy with occasional outliers, so the plots look like a real run's.
"""

import argparse
import math
import os
import random
import shutil

from robot_control import hw3_paths
from robot_control.localization import load_landmarks, NoiseParams
from robot_control.slam import EKFSLAM, SLAMParams
from robot_control.slam_logger import SLAMRunLogger
from robot_control.waypoint_controller import (
    body_to_wheel_cmds, Gains, load_waypoints, WaypointController, wheel_cmds_to_body,
    wrap_angle)

CONFIG_DIR = hw3_paths.config_dir()

# Start pose of each path in the world frame of landmarks_hw3_gt.txt (see the plan, section 3):
# the square's south-west corner and the octagon's south-side left end, both facing east.
PATHS = {
    'square': ('waypoints_hw3_square.txt', 4, hw3_paths.NOMINAL_START['square']),
    'octagon': ('waypoints_hw3_octagon.txt', 8, hw3_paths.NOMINAL_START['octagon']),
}


def load_params_yaml(path):
    """ros__parameters of the first node in a ROS params file, as a flat dict."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    node = next(iter(data.values()))
    return node.get('ros__parameters', {})


def pick(cls, params):
    return cls(**{k: type(v)(params[k]) for k, v in vars(cls()).items() if k in params})


def to_start_frame(start, pose):
    x0, y0, t0 = start
    dx, dy = pose[0] - x0, pose[1] - y0
    c, s = math.cos(t0), math.sin(t0)
    return (c * dx + s * dy, -s * dx + c * dy, wrap_angle(pose[2] - t0))


def sense(true_pose, landmark, half_fov, max_range, max_facing):
    """Exact (range, bearing, orientation) of a landmark if the camera can see it, else None."""
    x, y, th = true_pose
    lx, ly, lyaw = landmark
    r = math.hypot(lx - x, ly - y)
    bearing = wrap_angle(math.atan2(ly - y, lx - x) - th)
    orientation = wrap_angle(lyaw - th)
    facing = abs(wrap_angle(orientation - (bearing + math.pi)))
    if r > max_range or abs(bearing) > half_fov or facing > max_facing:
        return None
    return r, bearing, orientation


def run_simulation(waypoints, landmarks, start, noise=None, slam_params=None, gains=None,
                   seed=0, dt=0.05, waypoints_per_lap=0, start_delay=3.0, settle_window=1.0,
                   search_inflate=0.3, logger=None, outlier_prob=0.01, camera_period=0.1,
                   v_scale=1.05, w_scale=0.93, veer=0.03, max_wheel_cmd=0.5,
                   speed_per_unit=1.26, wheel_base=0.68, max_steps=200000):
    """
    Drive the waypoints (relative to start) among landmarks (world frame) and run EKF-SLAM.

    Returns a dict with the filter, the true trajectory and the snapshots at each lap end.
    """
    rng = random.Random(seed)
    slam = EKFSLAM(noise or NoiseParams(), slam_params or SLAMParams())
    controller = WaypointController(waypoints, gains or Gains())
    true = list(start)
    odom = [0.0, 0.0, 0.0]
    v = w = 0.0
    t = 0.0
    last_update = -math.inf
    next_camera = 0.0
    truth, laps, trace_events = [], [], []
    last_log = -math.inf
    last_phase = None
    sightings = updates = 0
    for step in range(max_steps):
        t = step * dt
        # Motion: the real robot does not do exactly what was commanded
        tv = v * v_scale * (1 + rng.gauss(0, 0.02))
        tw = w * w_scale + veer * v + rng.gauss(0, 0.02) * abs(w)
        heading = true[2] + tw * dt / 2.0
        true[0] += tv * math.cos(heading) * dt
        true[1] += tv * math.sin(heading) * dt
        true[2] = wrap_angle(true[2] + tw * dt)
        heading = odom[2] + w * dt / 2.0
        odom = [odom[0] + v * math.cos(heading) * dt, odom[1] + v * math.sin(heading) * dt,
                wrap_angle(odom[2] + w * dt)]
        slam.predict(v, w, dt, stamp=t)

        if t >= next_camera:
            next_camera = t + camera_period
            for tag, lm in sorted(landmarks.items()):
                z = sense(true, lm, 0.9, 3.0, math.radians(70))
                if z is None:
                    continue
                r, b, o = z
                if rng.random() < outlier_prob:
                    z = (r * rng.uniform(0.6, 1.4), b + rng.uniform(-0.3, 0.3),
                         o + rng.uniform(-1.0, 1.0))
                else:
                    z = (r + rng.gauss(0, 0.01 + 0.02 * r), b + rng.gauss(0, 0.01),
                         o + rng.gauss(0, 0.08))
                before = float(sum(slam.sigma.diagonal()))
                obs = slam.observe(tag, z, stamp=t)
                sightings += 1
                if logger:
                    logger.measurement(t, t, tag, z, obs, slam.n_landmarks)
                if obs.kind == 'new' and logger:
                    logger.landmark_added(t, tag, slam)
                if obs.kind == 'update':
                    updates += 1
                    last_update = t
                    trace_events.append((t, tag, before, float(sum(slam.sigma.diagonal()))))

        truth.append((t, tuple(true)))
        lap = controller.index // waypoints_per_lap if waypoints_per_lap else 0
        if logger and t - last_log >= 0.5:
            last_log = t
            logger.trajectory(t, slam, controller.index, lap, controller.phase, sightings,
                              updates, v, w, odom, slam.pose)
            logger.truth_pose(t, to_start_frame(start, true))
            sightings = updates = 0

        if t < start_delay:
            v = w = 0.0
            continue
        if controller.done:
            break
        confirmed = t - last_update <= settle_window
        cv, cw, reached = controller.update(*slam.pose, dt, heading_confirmed=confirmed)
        if controller.phase == WaypointController.SEARCH and last_phase != controller.phase:
            slam.inflate((0.0, 0.0, search_inflate))  # as velocity_control does
        last_phase = controller.phase
        if reached is not None:
            lap_now = reached // waypoints_per_lap if waypoints_per_lap else 0
            if logger:
                logger.waypoint(reached, lap_now, t, waypoints[reached], slam)
                logger.snapshot(f'wp{reached}', lap_now, t, slam)
            if waypoints_per_lap and (reached + 1) % waypoints_per_lap == 0:
                laps.append((lap_now, slam.landmarks(), to_start_frame(start, true)))
                if logger:
                    logger.snapshot(f'lap{lap_now}', lap_now, t, slam, save_state=True)
        left, right = body_to_wheel_cmds(cv, cw, speed_per_unit, wheel_base, max_wheel_cmd)
        v, w = wheel_cmds_to_body(left, right, speed_per_unit, wheel_base)

    if logger:
        logger.finish(slam, t, controller.done, {'simulated': True, 'seed': seed})
    return {'slam': slam, 'truth': truth, 'laps': laps, 'done': controller.done,
            'trace_events': trace_events, 'odom': odom, 'time': t}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--path', choices=sorted(PATHS), default='square')
    parser.add_argument('--waypoints', help='waypoint file (default: the --path one)')
    parser.add_argument('--waypoints-per-lap', type=int)
    parser.add_argument('--gt', default=os.path.join(CONFIG_DIR, 'landmarks_hw3_gt.txt'),
                        help='true landmark positions in the world frame')
    parser.add_argument('--start', type=float, nargs=3, metavar=('X0', 'Y0', 'THETA0'),
                        help='true world-frame start pose (default: the --path one)')
    parser.add_argument('--params', default=os.path.join(CONFIG_DIR, 'hw3.yaml'))
    parser.add_argument('--landmark-dof', type=int, choices=(2, 3))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-dir', help='default: <workspace>/hw3_logs')
    args = parser.parse_args()

    wp_name, per_lap, start = PATHS[args.path]
    wp_file = args.waypoints or os.path.join(CONFIG_DIR, wp_name)
    per_lap = args.waypoints_per_lap or per_lap
    start = tuple(args.start) if args.start else start
    params = load_params_yaml(args.params) if os.path.exists(args.params) else {}
    if args.landmark_dof:
        params['landmark_dof'] = args.landmark_dof
    noise, slam_params, gains = (pick(NoiseParams, params), pick(SLAMParams, params),
                                 pick(Gains, params))
    info = {'simulated': True, 'path': args.path, 'gt_file': os.path.abspath(args.gt),
            'waypoints_per_lap': per_lap,
            'params': {**vars(noise), **vars(slam_params), **vars(gains)}}
    dof = int(params.get('landmark_dof', 3))
    logger = SLAMRunLogger(args.log_dir or hw3_paths.logs_dir(), info, wp_file, start_pose=start,
                           run_name=hw3_paths.sim_run_name(args.path, dof, args.seed))
    # keep the true map with the run: the shared GT file may be edited later
    shutil.copy(args.gt, logger.path('landmarks_gt_sim.txt'))
    result = run_simulation(load_waypoints(wp_file), load_landmarks(args.gt), start, noise,
                            slam_params, gains, seed=args.seed, waypoints_per_lap=per_lap,
                            logger=logger,
                            search_inflate=float(params.get('search_inflate', 0.3)))
    slam = result['slam']
    print(f'Simulated {result["time"]:.0f} s, finished={result["done"]}, '
          f'{slam.n_landmarks} landmarks {sorted(slam.index)}, counts {slam.counts}')
    print(f'Wrote {logger.run_dir}')


if __name__ == '__main__':
    main()
