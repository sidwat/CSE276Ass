"""
HW4 extra credit: SLAM-mapped obstacle + replanning. Simulate, evaluate, inspect (no ROS needed).

  ros2 run robot_control hw4_slam sim fast|safe [--seed N] [--prior none|walls]
        rehearse a run offline: EKF-SLAM + obstacle estimate + replanning + the real controller,
        against a miscalibrated robot and a noisy camera that cannot see through the box
  ros2 run robot_control hw4_slam eval [RUN ...]
        score a run: SLAM map vs the measured tags, obstacle estimate vs the measured box,
        clearance to the TRUE box, time, final error; writes RUN/eval/ (map.png, timeline.png)
  ros2 run robot_control hw4_slam prior [fast|safe]
        the plan the robot makes before mapping anything (prior region) vs on the true box
  ros2 run robot_control hw4_slam list

Runs live in <workspace>/hw4_logs as slam_<mode>_<date>_<time>[_tag] (robot) and
sim_slam_<mode>_seed<N>_<date>_<time> (simulation). RUN is a folder, a unique part of its name,
'latest' (newest robot run) or 'latest-sim'. Default world: config/hw4_world_slam.yaml.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import shutil
import sys

import numpy as np

from robot_control import hw4_tool
from robot_control.localization import NoiseParams
from robot_control.path_tools import path_length, Timing
from robot_control.slam import EKFSLAM, SLAMParams
from robot_control.slam_logger import SLAMRunLogger
from robot_control.slam_obstacle import (
    estimate_obstacle, prior_estimate, seed_known_walls, SlamConfig, SlamPlanner)
from robot_control.waypoint_controller import (
    body_to_wheel_cmds, Gains, WaypointController, wheel_cmds_to_body, wrap_angle)
from robot_control.world_model import load_world

COLORS = hw4_tool.COLORS
FACE_COLORS = {'east': '#2a78d6', 'west': '#eb6834', 'north': '#1baf7a', 'south': '#eda100'}
FACE_STYLES = {'east': '-', 'west': '--', 'north': '-.', 'south': ':'}
TAG_COLOR = '#8a1c4a'


def default_world_file():
    return os.path.realpath(os.path.join(hw4_tool.config_dir(), 'hw4_world_slam.yaml'))


def default_params_file():
    return os.path.realpath(os.path.join(hw4_tool.config_dir(), 'hw4_slam.yaml'))


def slam_runs():
    runs = [d for d in glob.glob(os.path.join(hw4_tool.logs_dir(), '*slam_*'))
            if os.path.isfile(os.path.join(d, 'trajectory.csv'))]
    return sorted(runs, key=os.path.getmtime)


def resolve_run(spec):
    if spec and os.path.isdir(os.path.expanduser(spec)):
        return os.path.abspath(os.path.expanduser(spec))
    runs = slam_runs()
    sims = [d for d in runs if os.path.basename(d).startswith('sim_')]
    robot = [d for d in runs if d not in sims]
    if spec in (None, '', 'latest'):
        if not (robot or sims):
            raise FileNotFoundError(f'no SLAM runs in {hw4_tool.logs_dir()}')
        return (robot or sims)[-1]
    if spec == 'latest-sim':
        if not sims:
            raise FileNotFoundError('no simulated SLAM runs')
        return sims[-1]
    matches = [d for d in runs if spec in os.path.basename(d)]
    if len(matches) != 1:
        raise FileNotFoundError(f'{spec!r} matches {len(matches)} SLAM runs')
    return matches[0]


# --- simulation ----------------------------------------------------------------------------------

def line_of_sight(world, p, q, end_gap=0.03):
    """Check that no box blocks the segment p -> q (q may lie on a box face)."""
    L = math.dist(p, q)
    n = max(2, int(L / 0.02))
    for k in range(1, n):
        t = k / n
        if L * (1.0 - t) < end_gap:
            break
        x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
        if any(float(b.distance(x, y)) <= 0.0 for b in world.boxes):
            return False
    return True


def sense(world, pose, landmark, half_fov, max_range, max_incidence):
    """Exact (range, bearing, orientation) of a tag the camera can see, else None."""
    x, y, th = pose
    lx, ly, lyaw = landmark
    r = math.hypot(lx - x, ly - y)
    bearing = wrap_angle(math.atan2(ly - y, lx - x) - th)
    orientation = wrap_angle(lyaw - th)
    incidence = abs(wrap_angle(orientation - (bearing + math.pi)))
    if r > max_range or abs(bearing) > half_fov or incidence > max_incidence:
        return None
    if not line_of_sight(world, (x, y), (lx, ly)):
        return None
    return r, bearing, orientation


def simulate(world, mode, params, seed=0, logger=None, dt=0.05, camera_period=0.1,
             half_fov=0.9, v_scale=1.05, w_scale=0.93, veer=0.03,
             place_std=(0.01, 0.01, 0.02), outlier_prob=0.01, max_time=240.0):
    """
    Run the SLAM + replanning loop of velocity_control against a simulated robot.

    world.boxes and the box tags of world.landmarks are the simulated truth; the robot's
    SlamPlanner never sees them. Returns a summary dict; logs through logger (SLAMRunLogger).
    """
    rng = random.Random(seed)
    gains = hw4_tool.pick(Gains, params)
    noise = hw4_tool.pick(NoiseParams, params)
    sp = hw4_tool.pick(SLAMParams, params)
    spu = float(params.get('speed_per_unit', 1.26))
    base = float(params.get('wheel_base', 0.68))
    max_cmd = float(params.get('max_wheel_cmd', 0.5))
    min_cmd = float(params.get('min_wheel_cmd', 0.0))
    max_range = float(params.get('max_tag_range', 2.5))
    start_delay = float(params.get('start_delay', 5.0))
    pause = float(params.get('pause_at_waypoint', 0.5))
    search_inflate = float(params.get('search_inflate', 0.3))
    log_period = float(params.get('log_period', 0.2))

    cfg = SlamConfig.from_world(world)
    slam = EKFSLAM(noise, sp)
    slam.set_pose(world.start)
    seed_known_walls(slam, world, cfg)
    planner = SlamPlanner(world, mode, cfg, Timing.from_world(world))
    controller = None
    true = [world.start[i] + rng.gauss(0, place_std[i]) for i in range(3)]
    odom = list(world.start)
    v = w = 0.0
    hold_until = start_delay
    last_fix = -math.inf
    last_phase = None
    next_camera = 0.0
    last_log = -math.inf
    sightings = updates = 0
    estimates = []
    min_true_gap = math.inf
    t = 0.0
    while t < max_time:
        tv = v * v_scale * (1 + rng.gauss(0, 0.02))
        tw = w * w_scale + veer * v + rng.gauss(0, 0.02) * abs(w)
        h = true[2] + tw * dt / 2.0
        true = [true[0] + tv * math.cos(h) * dt, true[1] + tv * math.sin(h) * dt,
                wrap_angle(true[2] + tw * dt)]
        h = odom[2] + w * dt / 2.0
        odom = [odom[0] + v * math.cos(h) * dt, odom[1] + v * math.sin(h) * dt,
                wrap_angle(odom[2] + w * dt)]
        slam.predict(v, w, dt, stamp=t)
        min_true_gap = min(min_true_gap, world.clearance_at(true[0], true[1])
                           - world.robot_radius)

        if t >= next_camera:
            next_camera = t + camera_period
            for tag in sorted(world.landmarks, key=str):
                z = sense(world, true, world.landmarks[tag], half_fov, max_range,
                          math.radians(70))
                if z is None:
                    continue
                r, b, o = z
                if rng.random() < outlier_prob:
                    z = (r * rng.uniform(0.6, 1.4), b + rng.uniform(-0.3, 0.3),
                         o + rng.uniform(-1.0, 1.0))
                else:
                    z = (r + rng.gauss(0, 0.01 + 0.02 * r), b + rng.gauss(0, 0.01),
                         o + rng.gauss(0, 0.08))
                obs = slam.observe(tag, z, stamp=t)
                sightings += 1
                if logger:
                    logger.measurement(t, t, tag, z, obs, slam.n_landmarks)
                if obs.kind == 'update':
                    updates += 1
                    last_fix = t
                elif obs.kind == 'new' and logger:
                    logger.landmark_added(t, tag, slam)

        if t - last_log >= log_period:
            last_log = t
            est = planner.latest or prior_estimate(cfg)
            estimates.append([t, *est.bounds(), est.source, ' '.join(sorted(est.faces)),
                              len(est.tags), slam.n_landmarks])
            if logger:
                logger.trajectory(t, slam, controller.index if controller else 0, 0,
                                  controller.phase if controller else 'hold', sightings,
                                  updates, v, w, odom, slam.pose)
                logger.truth_pose(t, true)
            sightings = updates = 0

        cv = cw = 0.0
        if t >= hold_until:
            remaining = controller.waypoints[controller.index:] if controller else None
            wps = planner.update(t, slam.pose, slam.landmarks(), remaining)
            if wps is not None:
                controller = WaypointController(wps, gains)
                if logger:
                    logger.snapshot(f'plan{len([h for h in planner.history if h["ok"]])}',
                                    0, t, slam)
            if controller is not None and controller.done:
                break
            if controller is not None:
                x, y, th = slam.pose
                confirmed = t - last_fix <= max(gains.settle_time, 0.5)
                cv, cw, reached = controller.update(x, y, th, dt, heading_confirmed=confirmed)
                if (controller.phase == WaypointController.SEARCH
                        and last_phase != controller.phase):
                    slam.inflate((0.0, 0.0, search_inflate))
                last_phase = controller.phase
                if reached is not None:
                    if logger:
                        logger.waypoint(reached, 0, t, controller.waypoints[reached], slam)
                    if not controller.is_pass_through(reached):
                        hold_until = t + pause
        left, right = body_to_wheel_cmds(cv, cw, spu, base, max_cmd)
        largest = max(abs(left), abs(right))
        if 1e-6 < largest < min_cmd:
            left *= min_cmd / largest
            right *= min_cmd / largest
        v, w = wheel_cmds_to_body(left, right, spu, base)
        t += dt

    done = controller is not None and controller.done
    est = planner.latest
    summary = {'simulated': True, 'seed': seed, 'slam_plan_mode': mode,
               'plans': len([h for h in planner.history if h['ok']]),
               'min_true_body_gap': min_true_gap, 'true_final_pose': true,
               'final_obstacle_estimate': est.to_dict() if est else None}
    if logger:
        logger.finish(slam, t, done, summary)
        with open(logger.path('replans.json'), 'w') as f:
            json.dump(planner.history, f, indent=1, default=str)
        with open(logger.path('obstacle_estimates.csv'), 'w', newline='') as f:
            out = csv.writer(f)
            out.writerow(['t', 'x_min', 'x_max', 'y_min', 'y_max', 'source', 'faces',
                          'n_obstacle_tags', 'n_landmarks'])
            for row in estimates:
                out.writerow([f'{row[0]:.2f}'] + [f'{v:.4f}' for v in row[1:5]] + row[5:])
    return {'done': done, 'time': t, 'slam': slam, 'planner': planner, 'true': true,
            'min_true_gap': min_true_gap}


def cmd_sim(args):
    world = load_world(args.world)
    if args.prior:
        world.slam = dict(world.slam or {}, prior_landmarks=args.prior)
    params = hw4_tool.load_params(args.params)
    name = f'sim_slam_{args.mode}_seed{args.seed}_' + hw4_tool.run_name('', '').lstrip('_')
    logger = SLAMRunLogger(hw4_tool.logs_dir(), {
        'simulated': True, 'path': f'slam_{args.mode}', 'world': os.path.abspath(args.world),
        'params_file': os.path.abspath(args.params), 'prior_landmarks':
            SlamConfig.from_world(world).prior_landmarks}, run_name=name,
        start_pose=world.start)
    shutil.copy(args.world, logger.path('world.yaml'))
    if args.prior:  # keep the run reproducible from its own folder
        import yaml
        data = yaml.safe_load(open(args.world))
        data.setdefault('slam', {})['prior_landmarks'] = args.prior
        with open(logger.path('world.yaml'), 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
    res = simulate(world, args.mode, params, args.seed, logger)
    print(f'simulated {res["time"]:.1f} s, reached goal: {res["done"]}, plans: '
          f'{len([h for h in res["planner"].history if h["ok"]])}, landmarks '
          f'{res["slam"].n_landmarks}, true min body gap {100 * res["min_true_gap"]:.1f} cm')
    evaluate(logger.run_dir, quiet=args.quiet, plot=not args.no_plot)
    return 0 if res['done'] and res['min_true_gap'] > 0 else 1


# --- evaluation ----------------------------------------------------------------------------------

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _tag(value):
    try:
        return int(value)
    except ValueError:
        return value


def read_map(path):
    """map_final.csv -> EKFSLAM.landmarks() format."""
    out = {}
    for r in read_csv(path):
        cov = np.array([[float(r['cov_xx']), float(r['cov_xy'])],
                        [float(r['cov_xy']), float(r['cov_yy'])]])
        psi = float(r['psi']) if r['psi'] not in ('', 'nan') else float('nan')
        out[_tag(r['tag'])] = ((float(r['x']), float(r['y']), psi), cov,
                               float(r['var_psi']) if r['var_psi'] not in ('', 'nan') else 0.0)
    return out


def read_map_state(run_dir):
    """Read the final map at full precision (map_final.csv rounds the covariances)."""
    path = os.path.join(run_dir, 'state_final.npz')
    if not os.path.exists(path):
        return read_map(os.path.join(run_dir, 'map_final.csv'))
    z = np.load(path)
    mu, sigma, dof = z['mu'], z['sigma'], int(z['landmark_dof'])
    out = {}
    for tag, j in zip(z['ids'], z['offsets']):
        tag = _tag(str(tag))
        psi = float(mu[j + 2]) if dof == 3 else float('nan')
        var_psi = float(sigma[j + 2, j + 2]) if dof == 3 else 0.0
        out[tag] = ((float(mu[j]), float(mu[j + 1]), psi), sigma[j:j + 2, j:j + 2].copy(),
                    var_psi)
    return out


def box_bounds(box):
    return (box.cx - box.hx, box.cx + box.hx, box.cy - box.hy, box.cy + box.hy)


def evaluate(run_dir, quiet=False, plot=True):
    """Score one SLAM run folder; writes eval/summary.json, results.md and figures."""
    world = load_world(os.path.join(run_dir, 'world.yaml'))
    cfg = SlamConfig.from_world(world)
    traj = read_csv(os.path.join(run_dir, 'trajectory.csv'))
    if not traj:
        raise FileNotFoundError(f'{run_dir}: no trajectory.csv')
    truth = read_csv(os.path.join(run_dir, 'truth.csv'))
    wps = read_csv(os.path.join(run_dir, 'waypoints.csv'))
    est_rows = read_csv(os.path.join(run_dir, 'obstacle_estimates.csv'))
    replans = json.load(open(os.path.join(run_dir, 'replans.json'))) \
        if os.path.exists(os.path.join(run_dir, 'replans.json')) else []
    info = json.load(open(os.path.join(run_dir, 'run_info.json'))) \
        if os.path.exists(os.path.join(run_dir, 'run_info.json')) else {}
    s_info = info.get('summary', {})
    lms = read_map_state(run_dir)

    T = np.array([float(r['t']) for r in traj])
    P = np.array([[float(r['x']), float(r['y']), float(r['theta'])] for r in traj])
    S = np.array([[float(r['std_x']), float(r['std_y'])] for r in traj])
    moving = [i for i, r in enumerate(traj)
              if abs(float(r['v'])) > 1e-4 or abs(float(r['w'])) > 1e-4]
    t_start = float(T[max(moving[0] - 1, 0)]) if moving else float(T[0])
    completed = bool(s_info.get('completed_all_waypoints', False))
    t_end = float(wps[-1]['t']) if (completed and wps) else float(T[-1])
    k0 = int(np.searchsorted(T, t_start))
    k1 = int(np.searchsorted(T, t_end, side='right'))
    seg = P[k0:max(k1, k0 + 1)]
    r = world.robot_radius
    true_box = world.boxes[0] if world.boxes else None
    d_est, _ = world.clearance(P[:, 0], P[:, 1])  # walls + TRUE box
    box_est_traj = true_box.distance(P[:, 0], P[:, 1]) if true_box else d_est
    goal = world.goal
    final = P[min(k1, len(P)) - 1]
    est = estimate_obstacle(lms, world, cfg) if lms else prior_estimate(cfg)
    ok_plans = [h for h in replans if h.get('ok')]
    summary = {
        'run': os.path.basename(run_dir),
        'mode': s_info.get('slam_plan_mode') or (ok_plans[0].get('mode') if ok_plans else ''),
        'simulated': bool(info.get('simulated', False)),
        'prior_landmarks': cfg.prior_landmarks,
        'completed': completed,
        'plans': len(ok_plans),
        'plan_reasons': [h['reason'] for h in ok_plans],
        'failed_replans': len(replans) - len(ok_plans),
        'time_s': t_end - t_start,
        'driven_length_m': path_length([tuple(p[:2]) for p in seg]),
        'first_plan_length_m': ok_plans[0]['length'] if ok_plans else None,
        'last_plan_length_m': ok_plans[-1]['length'] if ok_plans else None,
        'min_body_gap_est_m': float(d_est[k0:k1].min() - r) if k1 > k0 else None,
        'min_box_gap_est_m': float(box_est_traj[k0:k1].min() - r) if k1 > k0 else None,
        'mean_body_gap_est_m': float(d_est[k0:k1].mean() - r) if k1 > k0 else None,
        'final_error_est_m': float(math.hypot(final[0] - goal[0], final[1] - goal[1])),
        'max_std_m': float(np.hypot(S[:, 0], S[:, 1]).max()),
        'n_landmarks': len(lms),
    }
    # SLAM map vs measured tags
    errs = {'wall': [], 'box': []}
    tag_rows = []
    for tag, ((x, y, psi), cov, _) in sorted(lms.items(), key=lambda kv: str(kv[0])):
        gt = world.landmarks.get(tag)
        kind = 'wall' if min(x, world.size - x, y, world.size - y) < cfg.wall_band else 'box'
        if gt is None:
            tag_rows.append((tag, kind, x, y, None, None))
            continue
        e = math.hypot(x - gt[0], y - gt[1])
        e_psi = abs(wrap_angle(psi - gt[2])) if math.isfinite(psi) else float('nan')
        errs[kind].append(e)
        tag_rows.append((tag, kind, x, y, e, e_psi))
    for kind in ('wall', 'box'):
        summary[f'{kind}_tags_mapped'] = len([t for t in tag_rows if t[1] == kind])
        summary[f'{kind}_tag_error_mean_m'] = float(np.mean(errs[kind])) if errs[kind] else None
        summary[f'{kind}_tag_error_max_m'] = float(np.max(errs[kind])) if errs[kind] else None
    summary['tag_errors_m'] = {str(t[0]): t[4] for t in tag_rows}
    # obstacle estimate vs the measured box
    summary['obstacle_estimate'] = list(est.bounds())
    summary['faces_seen'] = sorted(est.faces)
    if true_box:
        tb = box_bounds(true_box)
        # positive = the estimated face lies outside the true face (conservative)
        outside = {'west': tb[0] - est.x_min, 'east': est.x_max - tb[1],
                   'south': tb[2] - est.y_min, 'north': est.y_max - tb[3]}
        ix = max(0.0, min(est.x_max, tb[1]) - max(est.x_min, tb[0]))
        iy = max(0.0, min(est.y_max, tb[3]) - max(est.y_min, tb[2]))
        inter = ix * iy
        area_e = (est.x_max - est.x_min) * (est.y_max - est.y_min)
        area_t = (tb[1] - tb[0]) * (tb[3] - tb[2])
        summary.update({
            'true_obstacle': list(tb),
            'face_offset_outside_m': outside,
            'estimate_contains_true_box': all(v >= -1e-9 for v in outside.values()),
            'obstacle_iou': inter / (area_e + area_t - inter),
            'obstacle_center_error_m': math.hypot(est.box.cx - true_box.cx,
                                                  est.box.cy - true_box.cy),
        })
    if truth:
        Q = np.array([[float(r['x']), float(r['y']), float(r['theta'])] for r in truth])
        d_true, _ = world.clearance(Q[:, 0], Q[:, 1])
        summary.update({
            'min_body_gap_true_m': float(d_true.min() - r),
            'collision_true': bool(d_true.min() < r),
            'final_error_true_m': float(math.hypot(Q[-1, 0] - goal[0], Q[-1, 1] - goal[1])),
        })
    else:
        Q = None

    out = os.path.join(run_dir, 'eval')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=str)
    lines = [f'# HW4 SLAM run {summary["run"]}', '', '| metric | value |', '|---|---|']
    for k, v in summary.items():
        if isinstance(v, float):
            lines.append(f'| {k} | {v:.3f} |')
        elif isinstance(v, dict):
            lines.append(f'| {k} | ' + ', '.join(
                f'{kk}: {vv:.3f}' if isinstance(vv, float) else f'{kk}: {vv}'
                for kk, vv in v.items()) + ' |')
        else:
            lines.append(f'| {k} | {v} |')
    with open(os.path.join(out, 'results.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    if plot:
        plot_map(world, cfg, lms, est, replans, P, Q, summary, os.path.join(out, 'map.png'))
        plot_timeline(world, est_rows, replans, traj, truth, t_start, summary,
                      os.path.join(out, 'timeline.png'))
    if not quiet:
        print('\n'.join(lines[4:]).replace('| ', '').replace(' |', ''))
        print(f'\nwrote {out}/')
    return summary


# --- figures -------------------------------------------------------------------------------------

def _rect(ax, bounds, **kw):
    from matplotlib.patches import Rectangle
    x0, x1, y0, y1 = bounds
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, **kw))


def _ellipse(ax, x, y, cov, color, k=2.0):
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(np.asarray(cov))
    vals = np.maximum(vals, 1e-10)
    ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse((x, y), 2 * k * math.sqrt(vals[1]), 2 * k * math.sqrt(vals[0]),
                         angle=ang, fill=False, ec=color, lw=0.8))


def plot_map(world, cfg, lms, est, replans, P, Q, summary, png):
    plt = hw4_tool._pyplot()
    if plt is None:
        return
    mode = summary.get('mode') or 'fast'
    col = COLORS.get(mode, 'k')
    L = world.size
    fig, ax = plt.subplots(figsize=(7, 7.4))
    _rect(ax, (0, L, 0, L), fill=False, lw=2, ec='k')
    for b in world.boxes:
        _rect(ax, box_bounds(b), color='0.25', label='true box (measured)')
    _rect(ax, prior_estimate(cfg).bounds(), fill=False, ls=':', lw=1.2, ec='0.35',
          label='prior region (before mapping)')
    _rect(ax, est.bounds(), fill=False, ls='--', lw=1.6, ec=col,
          label=f'SLAM estimate (faces: {", ".join(sorted(est.faces)) or "none"})')
    ok = [h for h in replans if h.get('ok')]
    for i, h in enumerate(ok):
        xs, ys = zip(*h['path'])
        last = i == len(ok) - 1
        ax.plot(xs, ys, '-o' if last else '-', color=col, lw=2.0 if last else 1.0,
                ms=3, alpha=1.0 if last else 0.35,
                label=(f'plan {i + 1} (final)' if last else
                       ('earlier plans' if i == 0 else None)))
    ax.plot(P[:, 0], P[:, 1], color='#2f3fa0', lw=1.4, label='SLAM pose estimate')
    if Q is not None:
        ax.plot(Q[:, 0], Q[:, 1], color='k', lw=0.9, label='true path (simulation)')
    first = True
    for tag, (x, y, yaw) in world.landmarks.items():
        ax.plot(x, y, 'x', color='0.45', ms=5, label='measured tags' if first else None)
        first = False
    first = True
    for tag, ((x, y, psi), cov, _) in lms.items():
        ax.plot(x, y, 'o', color=TAG_COLOR, ms=3.5, label='mapped tags (2-sigma)' if first
                else None)
        _ellipse(ax, x, y, cov, TAG_COLOR)
        ax.text(x + 0.05, y + 0.05, str(tag), fontsize=6.5, color=TAG_COLOR)
        first = False
    sx, sy, sth = world.start
    ax.annotate('', (sx + 0.2 * math.cos(sth), sy + 0.2 * math.sin(sth)), (sx, sy),
                arrowprops={'arrowstyle': '->', 'color': 'k'})
    ax.plot(*world.goal[:2], 'k*', ms=12)
    ax.set_xlim(-0.3, L + 0.3)
    ax.set_ylim(-0.3, L + 0.3)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.set_xlabel('x east [m]')
    ax.set_ylabel('y north [m]')
    ax.set_title(f'{summary["run"]}: SLAM map, obstacle estimate and plans', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=7,
              frameon=False)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def plot_timeline(world, est_rows, replans, traj, truth, t_start, summary, png):
    plt = hw4_tool._pyplot()
    if plt is None:
        return
    mode = summary.get('mode') or 'fast'
    col = COLORS.get(mode, 'k')
    r = world.robot_radius
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)
    a1, a2, a3 = axes
    tb = box_bounds(world.boxes[0]) if world.boxes else None
    if est_rows and tb:
        t = np.array([float(e['t']) for e in est_rows]) - t_start
        side = {'west': lambda e: tb[0] - float(e['x_min']),
                'east': lambda e: float(e['x_max']) - tb[1],
                'south': lambda e: tb[2] - float(e['y_min']),
                'north': lambda e: float(e['y_max']) - tb[3]}
        for face, fn in side.items():
            a1.plot(t, [100 * fn(e) for e in est_rows], color=FACE_COLORS[face],
                    ls=FACE_STYLES[face], lw=1.6, label=face)
        a1.axhline(0, color='k', lw=1)
        a1.text(t[0], 1, 'true face', fontsize=7, color='0.3', va='bottom')
    a1.set_ylabel('estimated face outside\ntrue face [cm]')
    a1.legend(loc='upper right', ncol=4, fontsize=7, frameon=False)
    a1.set_title(f'{summary["run"]}\nobstacle estimate, clearance and map over time '
                 '(t = 0: first motion; dotted: (re)plans)', fontsize=9)
    T = np.array([float(x['t']) for x in traj]) - t_start
    P = np.array([[float(x['x']), float(x['y'])] for x in traj])
    d, _ = world.clearance(P[:, 0], P[:, 1])
    a2.plot(T, 100 * (d - r), color=col, lw=1.8, label='SLAM estimate')
    if truth:
        Tq = np.array([float(x['t']) for x in truth]) - t_start
        Q = np.array([[float(x['x']), float(x['y'])] for x in truth])
        dq, _ = world.clearance(Q[:, 0], Q[:, 1])
        a2.plot(Tq, 100 * (dq - r), color='k', lw=1.0, label='true (simulation)')
    a2.axhline(0, color='#c0392b', lw=1)
    a2.set_ylabel('body gap to the TRUE\nobstacles [cm]')
    a2.legend(loc='upper right', fontsize=7, frameon=False)
    a3.step(T, [int(x['n_landmarks']) for x in traj], where='post', color='0.2', lw=1.4)
    a3.set_ylabel('mapped landmarks')
    a3.set_xlabel('time [s]')
    for h in replans:
        if h.get('ok'):
            for a in axes:
                a.axvline(h['t'] - t_start, color='0.5', ls=':', lw=0.9)
    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


# --- prior vs true -------------------------------------------------------------------------------

def cmd_prior(args):
    world = load_world(args.world)
    cfg = SlamConfig.from_world(world)
    out = args.out or os.path.join(hw4_tool.logs_dir(), 'plans_slam')
    os.makedirs(out, exist_ok=True)
    modes = ['fast', 'safe'] if args.mode == 'both' else [args.mode]
    plt = hw4_tool._pyplot()
    for mode in modes:
        p = SlamPlanner(world, mode, cfg)
        p.update(0.0, world.start, {})
        prior_plan = p.plan
        true_plan = hw4_tool.make_plan(world, mode)
        print(f'{mode}: prior-only plan {path_length(prior_plan.path):.3f} m, '
              f'{len(prior_plan.path) - 1} waypoints; on the true box '
              f'{path_length(true_plan.path):.3f} m, {len(true_plan.path) - 1} waypoints')
        if plt is None:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        hw4_tool.draw_world(ax, world)
        _rect(ax, prior_estimate(cfg).bounds(), fill=False, ls=':', lw=1.2, ec='0.35',
              label='prior region')
        xs, ys = zip(*prior_plan.path)
        ax.plot(xs, ys, '--o', color=COLORS[mode], ms=3, lw=1.5, label='plan on the prior')
        xs, ys = zip(*true_plan.path)
        ax.plot(xs, ys, '-o', color=COLORS[mode], ms=3, lw=2, label='plan on the true box')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'{mode}: before mapping vs with the box known', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'prior_vs_true_{mode}.png'), dpi=150)
        plt.close(fig)
    print(f'wrote {out}/')
    return 0


def cmd_eval(args):
    for spec in args.runs or ['latest']:
        evaluate(resolve_run(spec), quiet=args.quiet, plot=not args.no_plot)
    return 0


def cmd_list(args):
    for d in slam_runs():
        s = os.path.join(d, 'eval', 'summary.json')
        tail = ''
        if os.path.exists(s):
            j = json.load(open(s))
            tail = (f'  done={j.get("completed")} time={j.get("time_s", 0):.1f}s '
                    f'plans={j.get("plans")} faces={j.get("faces_seen")}')
        print(os.path.basename(d) + tail)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='hw4_slam', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('sim', help='simulate a SLAM run and evaluate it')
    p.add_argument('mode', choices=['fast', 'safe'])
    p.add_argument('--world', default=default_world_file())
    p.add_argument('--params', default=default_params_file())
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--prior', choices=['none', 'walls'], help='override slam.prior_landmarks')
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--no-plot', action='store_true')
    p.set_defaults(func=cmd_sim)
    p = sub.add_parser('eval', help='evaluate SLAM run folders')
    p.add_argument('runs', nargs='*')
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--no-plot', action='store_true')
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser('prior', help='plan on the prior region vs on the true box')
    p.add_argument('mode', nargs='?', default='both', choices=['fast', 'safe', 'both'])
    p.add_argument('--world', default=default_world_file())
    p.add_argument('--out')
    p.set_defaults(func=cmd_prior)
    p = sub.add_parser('list', help='list SLAM runs')
    p.set_defaults(func=cmd_list)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
