"""
Simulated HW3 runs: the waypoint controller drives EKF-SLAM in a simulated arena.

Used by test/test_slam.py, and by the simulate_slam entry point, which writes a run folder in
the same format as velocity_control (use_slam) so evaluate_slam can be tried without the robot:

    ros2 run robot_control simulate_slam --path square --laps 3
    ros2 run robot_control evaluate_slam <run_dir> --gt <run_dir>/landmarks_gt.txt \
        --start 0.914 0.914 0
"""

import argparse
import csv
from dataclasses import dataclass, field
import math
import os
import random
import time

import numpy as np
from robot_control.localization import NoiseParams
from robot_control.slam import (
    EKFSLAM, is_lap_end, MAP_COLUMNS, map_rows, MEASUREMENT_COLUMNS, measurement_row, SLAMParams,
    SNAPSHOT_COLUMNS)
from robot_control.waypoint_controller import Gains, WaypointController, wrap_angle

ARENA = 3.048  # 10 ft
FT = 0.3048


def arena_landmarks(side=ARENA, ids=(11, 4, 17, 2, 23, 8, 14, 29)):
    """Two tags per wall at 1/3 and 2/3, world frame (origin SW corner, x east, y north)."""
    a, b = side / 3.0, 2.0 * side / 3.0
    poses = [(a, 0.0, math.pi / 2), (b, 0.0, math.pi / 2),        # south wall, facing north
             (side, a, math.pi), (side, b, math.pi),              # east wall, facing west
             (b, side, -math.pi / 2), (a, side, -math.pi / 2),    # north wall, facing south
             (0.0, b, 0.0), (0.0, a, 0.0)]                        # west wall, facing east
    return dict(zip(ids, poses))


def square_waypoints(n=4 * FT, laps=3):
    """Square of side n from the start (its south-west corner), counter-clockwise."""
    lap = [(n, 0.0, math.pi / 2), (n, n, math.pi), (0.0, n, -math.pi / 2), (0.0, 0.0, 0.0)]
    return lap * laps


def octagon_waypoints(m=2 * FT, laps=3):
    """Regular octagon of side m from the start (left end of its south side), counter-clockwise."""
    lap, x, y = [], 0.0, 0.0
    for i in range(8):
        x += m * math.cos(i * math.pi / 4)
        y += m * math.sin(i * math.pi / 4)
        lap.append((x, y, wrap_angle((i + 1) * math.pi / 4)))
    lap[-1] = (0.0, 0.0, 0.0)
    return lap * laps


def centred_start(path, size, side=ARENA):
    """World start pose (facing east) that centres the square / octagon in the arena."""
    if path == 'square':
        return ((side - size) / 2.0, (side - size) / 2.0, 0.0)
    across = size * (1.0 + math.sqrt(2.0))
    return ((side - size) / 2.0, (side - across) / 2.0, 0.0)


def to_start_frame(landmarks, start):
    x0, y0, t0 = start
    c, s = math.cos(t0), math.sin(t0)
    out = {}
    for tag_id, (x, y, psi) in landmarks.items():
        dx, dy = x - x0, y - y0
        out[tag_id] = (c * dx + s * dy, -s * dx + c * dy, wrap_angle(psi - t0))
    return out


def observe(pose, landmark):
    """Exact (range, bearing, orientation) of a landmark seen from pose."""
    x, y, theta = pose
    lx, ly, lyaw = landmark
    return (math.hypot(lx - x, ly - y),
            wrap_angle(math.atan2(ly - y, lx - x) - theta),
            wrap_angle(lyaw - theta))


def visible(pose, landmark, half_fov, max_range):
    r, bearing, orientation = observe(pose, landmark)
    facing = abs(wrap_angle(orientation - (bearing + math.pi))) < math.radians(75)
    return r < max_range and abs(bearing) < half_fov and facing


@dataclass
class SimConfig:
    dt: float = 0.05
    camera_period: int = 2        # control steps per camera frame (10 Hz)
    half_fov: float = 0.8         # rad (the real camera sees ~0.94)
    max_range: float = 2.8        # m
    # Systematic odometry error: the robot drives 5% farther, turns 6% less than commanded and
    # veers left, as in HW2
    v_scale: float = 1.05
    w_scale: float = 0.94
    veer: float = 0.02            # rad/s of turn per m/s of forward speed
    # Random odometry error, as random walks (same form as the filter's model)
    odom_dist_std: float = 0.03   # m after 1 m
    odom_turn_std: float = 0.05   # rad after 1 rad
    odom_drift_std: float = 0.02  # rad after 1 m
    # Camera noise (true, the filter's R is set separately)
    range_std: float = 0.01
    range_std_per_m: float = 0.01
    bearing_std: float = 0.01
    orientation_std: float = 0.05
    outlier_rate: float = 0.0     # fraction of sightings replaced by garbage
    max_time: float = 2000.0      # s
    seed: int = 1


@dataclass
class SimResult:
    slam: EKFSLAM
    landmarks: dict                       # true landmarks in the start frame
    truth: list = field(default_factory=list)       # (t, x, y, theta)
    estimate: list = field(default_factory=list)    # (t, x, y, theta, std_x, std_y, std_theta)
    odom: list = field(default_factory=list)        # dead reckoning (t, x, y, theta)
    n_history: list = field(default_factory=list)   # (t, N) whenever N changes
    events: list = field(default_factory=list)      # (t, tag, Observation, true pose)
    dr_sightings: dict = field(default_factory=dict)  # tag -> [implied (x, y) from dead reckoning]
    reached: list = field(default_factory=list)     # (index, t, goal, estimate)
    snapshots: list = field(default_factory=list)   # (lap, t, robot pose, map rows)
    seen_ids: set = field(default_factory=set)      # ids that were ever in view
    nees: list = field(default_factory=list)        # robot pose NEES at each camera frame

    def landmark_errors(self):
        """{tag: |estimated - true| position} in the start frame."""
        return {tag: math.hypot(m[0] - self.landmarks[tag][0], m[1] - self.landmarks[tag][1])
                for tag, (m, _) in self.slam.landmarks().items()}

    def dead_reckoning_errors(self):
        """Return the same errors for a map built from dead reckoning (mean of all sightings)."""
        out = {}
        for tag, points in self.dr_sightings.items():
            mx = sum(p[0] for p in points) / len(points)
            my = sum(p[1] for p in points) / len(points)
            out[tag] = math.hypot(mx - self.landmarks[tag][0], my - self.landmarks[tag][1])
        return out


def default_gains():
    """Return the controller settings of config/hw3.yaml."""
    return Gains(pos_tolerance=0.03, heading_tolerance=0.05, replan_distance=0.1,
                 settle_time=1.0, k_v=1.26, k_w=1.26, max_v=0.15, min_v=0.05,
                 max_w=1.6, min_w=1.1, realign_angle=0.5)


def simulate(waypoints, landmarks_world, start, noise=None, params=None, cfg=None, gains=None,
             waypoints_per_lap=0):
    """Drive the waypoints with SLAM in the loop. Returns a SimResult (everything start frame)."""
    cfg = cfg or SimConfig()
    rng = random.Random(cfg.seed)
    slam = EKFSLAM(noise or NoiseParams(), params or SLAMParams())
    landmarks = to_start_frame(landmarks_world, start)
    controller = WaypointController(waypoints, gains or default_gains())
    result = SimResult(slam=slam, landmarks=landmarks)

    true = [0.0, 0.0, 0.0]
    odom = [0.0, 0.0, 0.0]
    v = w = 0.0
    last_fix = -math.inf
    hold_until = 0.0
    lap = 0
    steps = int(cfg.max_time / cfg.dt)
    for step in range(steps):
        t = step * cfg.dt
        # Truth: systematic + random odometry error
        dist, turn = v * cfg.dt, w * cfg.dt
        t_dist = cfg.v_scale * dist + rng.gauss(0.0, cfg.odom_dist_std * math.sqrt(abs(dist)))
        t_turn = (cfg.w_scale * turn + cfg.veer * dist
                  + rng.gauss(0.0, math.sqrt(cfg.odom_turn_std ** 2 * abs(turn)
                                             + cfg.odom_drift_std ** 2 * abs(dist))))
        heading = true[2] + t_turn / 2.0
        true[0] += t_dist * math.cos(heading)
        true[1] += t_dist * math.sin(heading)
        true[2] = wrap_angle(true[2] + t_turn)
        heading = odom[2] + turn / 2.0
        odom[0] += dist * math.cos(heading)
        odom[1] += dist * math.sin(heading)
        odom[2] = wrap_angle(odom[2] + turn)
        slam.predict(v, w, cfg.dt, stamp=t)

        if step % cfg.camera_period == 0:
            n_before = slam.n_landmarks
            for tag, lm in landmarks.items():
                if not visible(true, lm, cfg.half_fov, cfg.max_range):
                    continue
                result.seen_ids.add(tag)
                r, b, o = observe(true, lm)
                if rng.random() < cfg.outlier_rate:
                    z = (rng.uniform(0.3, 3.0), rng.uniform(-0.8, 0.8),
                         rng.uniform(-math.pi, math.pi))
                else:
                    z = (r + rng.gauss(0.0, cfg.range_std + cfg.range_std_per_m * r),
                         b + rng.gauss(0.0, cfg.bearing_std),
                         wrap_angle(o + rng.gauss(0.0, cfg.orientation_std)))
                pose_at = tuple(slam.pose_at(t))
                obs = slam.observe(tag, z, stamp=t, w=w)
                result.events.append((t, tag, z, obs, pose_at, tuple(true)))
                if obs.event == 'updated':
                    last_fix = t
                hb = odom[2] + z[1]
                result.dr_sightings.setdefault(tag, []).append(
                    (odom[0] + z[0] * math.cos(hb), odom[1] + z[0] * math.sin(hb)))
            if slam.n_landmarks != n_before:
                result.n_history.append((t, slam.n_landmarks))
            e = np.array([true[0] - slam.mu[0], true[1] - slam.mu[1],
                          wrap_angle(true[2] - slam.mu[2])])
            result.nees.append(float(e @ np.linalg.solve(slam.sigma[:3, :3], e)))

        result.truth.append((t, *true))
        result.estimate.append((t, *slam.pose, *slam.std))
        result.odom.append((t, *odom))

        if t < hold_until:
            v = w = 0.0
            continue
        confirmed = t - last_fix <= max(controller.gains.settle_time, 0.5)
        v_cmd, w_cmd, reached = controller.update(*slam.pose, cfg.dt, heading_confirmed=confirmed)
        if reached is not None:
            goal = controller.waypoints[reached]
            result.reached.append((reached, t, goal, slam.pose))
            hold_until = t + 0.5
            if is_lap_end(reached, goal, waypoints_per_lap):
                lap += 1
                result.snapshots.append((lap, t, slam.pose, map_rows(slam)))
        if controller.done:
            break
        v, w = v_cmd, w_cmd
    else:
        raise RuntimeError(f'simulation did not finish (stuck at waypoint {controller.index})')
    return result


def write_run(result, run_dir, landmarks_world, start, log_period=0.2):
    """Write the run in velocity_control's log format, plus truth.csv and landmarks_gt.txt."""
    os.makedirs(run_dir, exist_ok=True)

    def writer(name, header):
        f = open(os.path.join(run_dir, name), 'w', newline='')
        w = csv.writer(f)
        w.writerow(header)
        return f, w

    f, w = writer('trajectory.csv', ['t', 'x', 'y', 'theta', 'std_x', 'std_y', 'std_theta',
                                     'waypoint', 'tags_seen', 'odom_x', 'odom_y', 'odom_theta',
                                     'n_landmarks'])
    changes = list(result.n_history)
    n, last = 0, -math.inf
    for est, od in zip(result.estimate, result.odom):
        while changes and changes[0][0] <= est[0]:
            n = changes.pop(0)[1]
        if est[0] - last < log_period:
            continue
        last = est[0]
        w.writerow([f'{est[0]:.2f}'] + [f'{v:.4f}' for v in est[1:]] + [0, 0]
                   + [f'{v:.4f}' for v in od[1:]] + [n])
    f.close()

    f, w = writer('truth.csv', ['t', 'x', 'y', 'theta'])
    for row in result.truth[::max(1, int(log_period / 0.05))]:
        w.writerow([f'{v:.4f}' for v in row])
    f.close()

    f, w = writer('measurements.csv', MEASUREMENT_COLUMNS)
    n = 0
    for t, tag, z, obs, pose_at, _ in result.events:
        if obs.event == 'added':
            n += 1
        w.writerow(measurement_row(t, tag, z, obs, pose_at, n))
    f.close()

    f, w = writer('waypoints.csv', ['index', 't', 'goal_x', 'goal_y', 'goal_theta',
                                    'x', 'y', 'theta'])
    for index, t, goal, est in result.reached:
        w.writerow([index, f'{t:.2f}', *goal] + [f'{v:.4f}' for v in est])
    f.close()

    f, w = writer('map_snapshots.csv', SNAPSHOT_COLUMNS)
    for lap, t, pose, rows in result.snapshots:
        for row in rows:
            w.writerow([lap, f'{t:.2f}'] + [f'{v:.4f}' for v in pose] + row)
    f.close()

    f, w = writer('map.csv', MAP_COLUMNS)
    for row in map_rows(result.slam):
        w.writerow(row)
    f.close()

    with open(os.path.join(run_dir, 'landmarks_gt.txt'), 'w') as f:
        f.write('# Simulated ground truth, world frame: id, x [m], y [m], yaw [rad]\n')
        for tag, (x, y, psi) in landmarks_world.items():
            f.write(f'{tag}, {x:.4f}, {y:.4f}, {psi:.5f}\n')
    with open(os.path.join(run_dir, 'start_pose.txt'), 'w') as f:
        f.write(f'{start[0]:.4f} {start[1]:.4f} {start[2]:.5f}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--path', choices=('square', 'octagon'), default='square')
    parser.add_argument('--size', type=float, help='n or m in feet (default 4 / 2)')
    parser.add_argument('--laps', type=int, default=3)
    parser.add_argument('--dof', type=int, choices=(2, 3), default=3, help='landmark_dof')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--outlier-rate', type=float, default=0.0)
    parser.add_argument('--log-dir', default='~/hw3_logs/sim')
    args = parser.parse_args()

    size = (args.size or (4.0 if args.path == 'square' else 2.0)) * FT
    waypoints = (square_waypoints(size, args.laps) if args.path == 'square'
                 else octagon_waypoints(size, args.laps))
    start = centred_start(args.path, size)
    landmarks_world = arena_landmarks()
    noise = NoiseParams(odom_dist_std=0.1, odom_turn_std=0.3, odom_drift_std=0.05)
    result = simulate(waypoints, landmarks_world, start, noise=noise,
                      params=SLAMParams(landmark_dof=args.dof),
                      cfg=SimConfig(seed=args.seed, outlier_rate=args.outlier_rate))
    run_dir = os.path.join(os.path.expanduser(args.log_dir),
                           time.strftime(f'run_%Y%m%d_%H%M%S_{args.path}'))
    write_run(result, run_dir, landmarks_world, start)

    errors = result.landmark_errors()
    print(f'Simulated {args.path} ({args.laps} laps, {result.truth[-1][0]:.0f} s): '
          f'{result.slam.n_landmarks} landmarks, '
          f'mean error {sum(errors.values()) / len(errors):.3f} m')
    print(f'Wrote {run_dir}')
    print(f'Evaluate with:\n  ros2 run robot_control evaluate_slam {run_dir} '
          f'--gt {run_dir}/landmarks_gt.txt --start {start[0]:.4f} {start[1]:.4f} {start[2]:.4f}')


if __name__ == '__main__':
    main()
