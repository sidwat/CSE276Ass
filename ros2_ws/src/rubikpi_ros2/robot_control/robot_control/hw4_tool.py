"""
HW4 path planning: plan, simulate and evaluate the safe / fast runs (no ROS needed).

  ros2 run robot_control hw4 plan [fast|safe|weighted|all] [--world FILE] [--out DIR]
        plan from config/hw4_world.yaml; writes waypoints_hw4_<mode>.txt, plan_<mode>.json and
        plan_<mode>.png (+ plan_all.png comparing them) to <workspace>/hw4_logs/plans/
  ros2 run robot_control hw4 sim fast|safe [--seed N] [--world FILE] [--params FILE]
        plan, then drive the plan with the real controller + EKF against a miscalibrated
        robot and a noisy camera; writes a run folder like the robot's and evaluates it
  ros2 run robot_control hw4 eval [RUN]      trajectory vs plan, clearance profile, time, error
  ros2 run robot_control hw4 list            runs in <workspace>/hw4_logs, newest last

RUN is a run folder, a unique part of its name, or 'latest' (default).
Plots need matplotlib; in the devcontainer run with
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH (numpy/matplotlib version clash).
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
import time

import numpy as np

from robot_control import hw3_paths
from robot_control.gvd_planner import plan_safe, plan_weighted
from robot_control.localization import NoiseParams, PoseEKF
from robot_control.path_tools import (
    metrics, path_length, PlanningError, save_plan_json, Timing, to_waypoints, write_waypoints)
from robot_control.visibility_planner import plan_fast
from robot_control.waypoint_controller import (
    body_to_wheel_cmds, Gains, load_waypoints, WaypointController, wheel_cmds_to_body,
    wrap_angle)
from robot_control.world_model import load_world, WorldError

PLANNERS = {'fast': plan_fast, 'safe': plan_safe, 'weighted': plan_weighted}
COLORS = {'fast': '#c25e14', 'safe': '#17768a', 'weighted': '#7655c4'}


# --- paths ---------------------------------------------------------------------------------------

def config_dir():
    return hw3_paths.config_dir()


def default_world_file():
    """Return the world file to edit (the source file behind the symlink install, if any)."""
    return os.path.realpath(os.path.join(config_dir(), 'hw4_world.yaml'))


def default_params_file():
    return os.path.realpath(os.path.join(config_dir(), 'hw4.yaml'))


def logs_dir():
    d = os.environ.get('HW4_LOG_DIR') or os.path.join(hw3_paths.workspace_root(), 'hw4_logs')
    d = os.path.expanduser(d)
    os.makedirs(d, exist_ok=True)
    marker = os.path.join(d, 'COLCON_IGNORE')  # keep colcon from crawling the bags
    if not os.path.exists(marker):
        open(marker, 'w').close()
    return d


def run_name(mode, tag='', sim=False):
    name = ('sim_' if sim else '') + f'{mode}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    return name + (f'_{tag}' if tag else '')


def all_runs():
    runs = [d for d in glob.glob(os.path.join(logs_dir(), '*'))
            if os.path.isfile(os.path.join(d, 'trajectory.csv'))]
    return sorted(runs, key=os.path.getmtime)


def resolve_run(spec):
    if spec and os.path.isdir(os.path.expanduser(spec)):
        return os.path.abspath(os.path.expanduser(spec))
    runs = all_runs()
    if not runs:
        raise FileNotFoundError(f'no runs in {logs_dir()}')
    if spec in (None, '', 'latest'):
        return runs[-1]
    matches = [d for d in runs if spec in os.path.basename(d)]
    if len(matches) != 1:
        raise FileNotFoundError(f'{spec!r} matches {len(matches)} runs in {logs_dir()}')
    return matches[0]


# --- planning ------------------------------------------------------------------------------------

def pass_through(world):
    return bool(world.planner.get('pass_through', True))


def make_plan(world, mode):
    if mode not in PLANNERS:
        raise ValueError(f'mode must be one of {sorted(PLANNERS)}')
    return PLANNERS[mode](world, Timing.from_world(world))


def write_plan_files(world, plan, out_dir, plot=True):
    """waypoints_hw4_<mode>.txt, plan_<mode>.json and (if matplotlib works) plan_<mode>.png."""
    os.makedirs(out_dir, exist_ok=True)
    timing = Timing.from_world(world)
    pt = pass_through(world)
    m = metrics(world, plan.path, timing, pt)
    wp_file = os.path.join(out_dir, f'waypoints_hw4_{plan.mode}.txt')
    write_waypoints(wp_file, to_waypoints(plan.path, world.goal[2], pt), [
        f'HW4 {plan.mode} plan from {world.source}',
        'world frame: x east, y north from the SW arena corner; theta nan = pass-through',
        f'start {world.start}  goal {world.goal}',
        f'length {m["length"]:.3f} m, min centre clearance {m["min_clearance"]:.3f} m '
        f'(body gap {m["min_body_gap"]:.3f} m), est. time {m["time"]:.1f} s'])
    save_plan_json(os.path.join(out_dir, f'plan_{plan.mode}.json'), plan, world, timing, pt)
    png = None
    if plot:
        png = os.path.join(out_dir, f'plan_{plan.mode}.png')
        if not plot_plans(world, [plan], png, title=f'HW4 {plan.mode} plan'):
            png = None
    return wp_file, m, png


def format_metrics_table(rows):
    head = (f'{"plan":9s} {"length m":>9s} {"min clr":>8s} {"mean clr":>9s} {"body gap":>9s} '
            f'{"turn rad":>9s} {"wps":>4s} {"t stop s":>9s} {"t pass s":>9s}')
    lines = [head, '-' * len(head)]
    for mode, m in rows:
        lines.append(f'{mode:9s} {m["length"]:9.3f} {m["min_clearance"]:8.3f} '
                     f'{m["mean_clearance"]:9.3f} {m["min_body_gap"]:9.3f} {m["turn"]:9.2f} '
                     f'{m["waypoints"]:4d} {m["time_stop"]:9.1f} {m["time_pass"]:9.1f}')
    return '\n'.join(lines)


def cmd_plan(args):
    world = load_world(args.world)
    modes = list(PLANNERS) if args.mode == 'all' else [args.mode]
    out = os.path.expanduser(args.out) if args.out else os.path.join(logs_dir(), 'plans')
    plans, rows = [], []
    for mode in modes:
        plan = make_plan(world, mode)
        wp_file, m, png = write_plan_files(world, plan, out, plot=not args.no_plot)
        plans.append(plan)
        rows.append((mode, m))
        print(f'[{mode}] {wp_file}' + (f'  ({png})' if png else ''))
    print()
    print(format_metrics_table(rows))
    if len(plans) > 1 and not args.no_plot:
        png = os.path.join(out, 'plan_all.png')
        if plot_plans(world, plans, png, title='HW4 plans'):
            print(f'\ncomparison plot: {png}')
    return 0


# --- simulation ----------------------------------------------------------------------------------

def load_params(path):
    if not path or not os.path.exists(path):
        return {}
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return dict(next(iter(data.values())).get('ros__parameters', {}))


def pick(cls, params):
    return cls(**{k: type(v)(params[k]) for k, v in vars(cls()).items() if k in params})


def sense(pose, landmark, half_fov, max_range, max_incidence):
    """Return the exact (range, bearing, orientation) of a visible landmark, else None."""
    x, y, th = pose
    lx, ly, lyaw = landmark
    r = math.hypot(lx - x, ly - y)
    bearing = wrap_angle(math.atan2(ly - y, lx - x) - th)
    orientation = wrap_angle(lyaw - th)
    incidence = abs(wrap_angle(orientation - (bearing + math.pi)))
    if r > max_range or abs(bearing) > half_fov or incidence > max_incidence:
        return None
    return r, bearing, orientation


def simulate(world, waypoints, params, seed=0, dt=0.05, camera_period=0.1, half_fov=0.9,
             v_scale=1.05, w_scale=0.93, veer=0.03, place_std=(0.01, 0.01, 0.02),
             outlier_prob=0.01, max_time=300.0):
    """
    Drive the waypoints with the node's controller logic and EKF against a simulated robot.

    The simulated robot drives 5% too far, turns 7% too little and veers; it is placed with a
    small error; the camera sees landmarks inside +-half_fov and max_tag_range with noise and
    occasional outliers. Returns a dict of time series.
    """
    rng = random.Random(seed)
    gains = pick(Gains, params)
    noise = pick(NoiseParams, params)
    spu = float(params.get('speed_per_unit', 1.26))
    base = float(params.get('wheel_base', 0.68))
    max_cmd = float(params.get('max_wheel_cmd', 0.5))
    min_cmd = float(params.get('min_wheel_cmd', 0.0))
    max_range = float(params.get('max_tag_range', 2.5))
    start_delay = float(params.get('start_delay', 3.0))
    pause = float(params.get('pause_at_waypoint', 0.5))
    search_inflate = 1.0

    ekf = PoseEKF(*world.start, noise=noise)
    controller = WaypointController(waypoints, gains)
    true = [world.start[i] + rng.gauss(0, place_std[i]) for i in range(3)]
    odom = list(world.start)
    v = w = 0.0
    hold_until = start_delay
    last_fix = -math.inf
    last_phase = None
    next_camera = 0.0
    rows, truth, reached_log = [], [], []
    accepted = rejected = 0
    t = 0.0
    while t < max_time:
        # true motion (miscalibrated) and the filter's prediction from the command
        tv = v * v_scale * (1 + rng.gauss(0, 0.02))
        tw = w * w_scale + veer * v + rng.gauss(0, 0.02) * abs(w)
        h = true[2] + tw * dt / 2.0
        true = [true[0] + tv * math.cos(h) * dt, true[1] + tv * math.sin(h) * dt,
                wrap_angle(true[2] + tw * dt)]
        h = odom[2] + w * dt / 2.0
        odom = [odom[0] + v * math.cos(h) * dt, odom[1] + v * math.sin(h) * dt,
                wrap_angle(odom[2] + w * dt)]
        ekf.predict(v, w, dt, stamp=t)

        if t >= next_camera:
            next_camera = t + camera_period
            for tag, lm in sorted(world.landmarks.items()):
                z = sense(true, lm, half_fov, max_range, math.radians(70))
                if z is None:
                    continue
                r, b, o = z
                if rng.random() < outlier_prob:
                    z = (r * rng.uniform(0.6, 1.4), b + rng.uniform(-0.3, 0.3),
                         o + rng.uniform(-1.0, 1.0))
                else:
                    z = (r + rng.gauss(0, 0.01 + 0.02 * r), b + rng.gauss(0, 0.01),
                         o + rng.gauss(0, 0.08))
                ok, _ = ekf.correct(z, lm, stamp=t)
                if ok:
                    accepted += 1
                    last_fix = t
                else:
                    rejected += 1

        x, y, th = ekf.pose
        std = ekf.std
        rows.append([t, x, y, th, *std, controller.index, *odom])
        truth.append([t, *true])

        cv = cw = 0.0
        if t >= hold_until and not controller.done:
            confirmed = t - last_fix <= max(gains.settle_time, 0.5)
            cv, cw, reached = controller.update(x, y, th, dt, heading_confirmed=confirmed)
            if controller.phase == WaypointController.SEARCH and last_phase != controller.phase:
                ekf.inflate((0.0, 0.0, search_inflate))
            last_phase = controller.phase
            if reached is not None:
                reached_log.append([reached, t, *waypoints[reached], x, y, th])
                if not controller.is_pass_through(reached):
                    hold_until = t + pause
        elif controller.done:
            break
        left, right = body_to_wheel_cmds(cv, cw, spu, base, max_cmd)
        largest = max(abs(left), abs(right))
        if 1e-6 < largest < min_cmd:
            left *= min_cmd / largest
            right *= min_cmd / largest
        v, w = wheel_cmds_to_body(left, right, spu, base)
        t += dt
    return {'rows': rows, 'truth': truth, 'reached': reached_log, 'done': controller.done,
            'time': t, 'accepted': accepted, 'rejected': rejected}


def write_sim_run(run_dir, world, waypoints_file, plan_dir, result):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'trajectory.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['t', 'x', 'y', 'theta', 'std_x', 'std_y', 'std_theta', 'waypoint',
                     'odom_x', 'odom_y', 'odom_theta'])
        for r in result['rows'][::10]:  # 0.5 s, like the node's log_period
            wr.writerow([f'{v:.4f}' if isinstance(v, float) else v for v in r])
        wr.writerow([f'{v:.4f}' if isinstance(v, float) else v for v in result['rows'][-1]])
    with open(os.path.join(run_dir, 'truth.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['t', 'x', 'y', 'theta'])
        for r in result['truth']:
            wr.writerow([f'{v:.4f}' for v in r])
    with open(os.path.join(run_dir, 'waypoints.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['index', 't', 'goal_x', 'goal_y', 'goal_theta', 'x', 'y', 'theta'])
        for r in result['reached']:
            wr.writerow([r[0]] + [f'{v:.4f}' for v in r[1:]])
    shutil.copy(world.source, os.path.join(run_dir, 'world.yaml'))
    shutil.copy(waypoints_file, os.path.join(run_dir, 'waypoints.txt'))
    for f in glob.glob(os.path.join(plan_dir, 'plan_*')):
        if os.path.abspath(os.path.dirname(f)) != os.path.abspath(run_dir):
            shutil.copy(f, run_dir)
    with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
        json.dump({'simulated': True, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'completed': result['done'], 'initial_pose': list(world.start),
                   'tags_accepted': result['accepted'], 'tags_rejected': result['rejected']},
                  f, indent=1)


def cmd_sim(args):
    world = load_world(args.world)
    params = load_params(args.params)
    run_dir = os.path.join(logs_dir(), run_name(args.mode, f'seed{args.seed}', sim=True))
    plan = make_plan(world, args.mode)
    wp_file, m, _ = write_plan_files(world, plan, run_dir, plot=not args.no_plot)
    result = simulate(world, load_waypoints(wp_file), params, seed=args.seed)
    write_sim_run(run_dir, world, wp_file, run_dir, result)
    drive = result['time'] - float(params.get('start_delay', 3.0))
    print(f'Simulated {drive:.1f} s of driving after the {params.get("start_delay", 3.0)} s '
          f'start delay (plan estimate {m["time"]:.1f} s), finished={result["done"]}, '
          f'tag updates {result["accepted"]} accepted / {result["rejected"]} rejected\n'
          f'run folder: {run_dir}')
    return evaluate(run_dir, world_file=None, quiet=False, plot=not args.no_plot)


# --- evaluation ----------------------------------------------------------------------------------

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def evaluate(run_dir, world_file=None, quiet=False, plot=True):
    """Compare a run with its plan; write eval/summary.json, results.md and eval.png."""
    world_file = world_file or os.path.join(run_dir, 'world.yaml')
    world = load_world(world_file)
    traj = read_csv(os.path.join(run_dir, 'trajectory.csv'))
    if not traj:
        raise FileNotFoundError(f'{run_dir}: no trajectory.csv')
    reached = read_csv(os.path.join(run_dir, 'waypoints.csv'))
    truth = read_csv(os.path.join(run_dir, 'truth.csv'))
    plans = sorted(glob.glob(os.path.join(run_dir, 'plan_*.json')))
    plan = json.load(open(plans[0])) if plans else None
    info = {}
    if os.path.exists(os.path.join(run_dir, 'run_info.json')):
        info = json.load(open(os.path.join(run_dir, 'run_info.json')))

    T = np.array([float(r['t']) for r in traj])
    P = np.array([[float(r['x']), float(r['y']), float(r['theta'])] for r in traj])
    S = np.array([[float(r['std_x']), float(r['std_y'])] for r in traj])
    moved = np.nonzero(np.hypot(P[:, 0] - P[0, 0], P[:, 1] - P[0, 1]) > 0.01)[0]
    t_start = float(T[max(moved[0] - 1, 0)]) if len(moved) else float(T[0])
    t_end = float(reached[-1]['t']) if reached else float(T[-1])
    est_path = [tuple(p[:2]) for p in P]
    d_est, _ = world.clearance(P[:, 0], P[:, 1])
    goal = world.goal
    final = P[-1]
    summary = {
        'run': os.path.basename(run_dir),
        'mode': plan['mode'] if plan else '',
        'simulated': bool(info.get('simulated', False)),
        'completed': bool(info.get('completed', bool(reached))),
        'waypoints_reached': len(reached),
        'waypoints_planned': len(plan['waypoints']) if plan else None,
        'time_s': t_end - t_start,
        'planned_time_s': plan['metrics']['time'] if plan else None,
        'driven_length_m': path_length(est_path),
        'planned_length_m': plan['metrics']['length'] if plan else None,
        'min_body_gap_est_m': float(d_est.min() - world.robot_radius),
        'mean_body_gap_est_m': float(d_est.mean() - world.robot_radius),
        'planned_min_body_gap_m': plan['metrics']['min_body_gap'] if plan else None,
        'final_error_est_m': float(math.hypot(final[0] - goal[0], final[1] - goal[1])),
        'final_heading_error_est_rad': (float(abs(wrap_angle(final[2] - goal[2])))
                                        if math.isfinite(goal[2]) else None),
        'final_std_m': float(np.hypot(*S[-1])),
        'max_std_m': float(np.hypot(S[:, 0], S[:, 1]).max()),
    }
    Q = None
    if truth:
        Q = np.array([[float(r['x']), float(r['y']), float(r['theta'])] for r in truth])
        d_true, _ = world.clearance(Q[:, 0], Q[:, 1])
        summary.update({
            'min_body_gap_true_m': float(d_true.min() - world.robot_radius),
            'collision_true': bool(d_true.min() < world.robot_radius),
            'final_error_true_m': float(math.hypot(Q[-1, 0] - goal[0], Q[-1, 1] - goal[1])),
        })
    measurements = read_csv(os.path.join(run_dir, 'measurements.csv'))
    if measurements:
        summary['tags_accepted'] = sum(int(r['accepted']) for r in measurements)
        summary['tags_rejected'] = len(measurements) - summary['tags_accepted']
    elif 'tags_accepted' in info:
        summary['tags_accepted'] = info['tags_accepted']
        summary['tags_rejected'] = info['tags_rejected']

    out = os.path.join(run_dir, 'eval')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    lines = [f'# HW4 run {summary["run"]}', '', '| metric | value |', '|---|---|']
    for k, v in summary.items():
        lines.append(f'| {k} | {v:.3f} |' if isinstance(v, float) else f'| {k} | {v} |')
    with open(os.path.join(out, 'results.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    if plot:
        plot_run(world, plan, T, P, S, d_est, Q, os.path.join(out, 'eval.png'),
                 title=f'{summary["run"]}')
    if not quiet:
        print('\n'.join(lines[4:]).replace('| ', '').replace(' |', ''))
        print(f'\nwrote {out}/')
    return 0


def cmd_eval(args):
    for spec in args.runs or ['latest']:
        evaluate(resolve_run(spec), args.world, args.quiet, not args.no_plot)
    return 0


def cmd_list(args):
    for d in all_runs():
        s = os.path.join(d, 'eval', 'summary.json')
        if os.path.exists(s):
            m = json.load(open(s))
            print(f'{os.path.basename(d):45s} done={m["completed"]!s:5s} '
                  f't={m["time_s"]:5.1f}s gap={100 * m["min_body_gap_est_m"]:5.1f}cm '
                  f'goal err={100 * m["final_error_est_m"]:4.1f}cm')
        else:
            print(f'{os.path.basename(d):45s} (not evaluated: hw4 eval {os.path.basename(d)})')
    return 0


# --- plotting ------------------------------------------------------------------------------------

def _pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # noqa: BLE001  (numpy/matplotlib ABI clash raises odd errors)
        print(f'(no plot: matplotlib unavailable: {e}. In the devcontainer use '
              f'PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH)', file=sys.stderr)
        return None


def draw_world(ax, world, rho=None, field=None):
    from matplotlib.patches import Polygon, Rectangle
    L = world.size
    if field is not None:
        xs, D = field
        ax.imshow(np.maximum(D, 0), origin='lower', extent=(0, L, 0, L), cmap='Blues',
                  alpha=0.55, interpolation='nearest')
    ax.add_patch(Rectangle((0, 0), L, L, fill=False, lw=2, ec='k'))
    if rho:
        ax.add_patch(Rectangle((rho, rho), L - 2 * rho, L - 2 * rho, fill=False, ls='--',
                               lw=0.8, ec='0.4'))
    for b in world.boxes:
        ax.add_patch(Rectangle((b.cx - b.hx, b.cy - b.hy), 2 * b.hx, 2 * b.hy, color='0.25'))
        if rho:
            ax.add_patch(Polygon(world.c_obstacle(b, rho), closed=True, fill=False, ls='--',
                                 lw=0.8, ec='0.3'))
    for tag, (x, y, yaw) in world.landmarks.items():
        dx, dy = -math.sin(yaw) * 0.0865, math.cos(yaw) * 0.0865
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color='#8a1c4a', lw=3)
        ax.annotate('', (x + 0.12 * math.cos(yaw), y + 0.12 * math.sin(yaw)), (x, y),
                    arrowprops={'arrowstyle': '->', 'color': '#8a1c4a', 'lw': 0.8})
        ax.text(x + 0.22 * math.cos(yaw), y + 0.22 * math.sin(yaw), str(tag), color='#8a1c4a',
                fontsize=7, ha='center', va='center')
    sx, sy, sth = world.start
    from matplotlib.patches import Circle
    ax.add_patch(Circle((sx, sy), world.robot_radius, fill=False, ec='k', lw=0.8))
    ax.annotate('', (sx + 0.2 * math.cos(sth), sy + 0.2 * math.sin(sth)), (sx, sy),
                arrowprops={'arrowstyle': '->', 'color': 'k'})
    ax.plot(*world.goal[:2], 'k*', ms=12)
    ax.text(sx + 0.15, sy - 0.12, 'start', fontsize=8)
    ax.text(world.goal[0] + 0.08, world.goal[1] + 0.1, 'goal', fontsize=8)
    ax.set_xlim(-0.3, L + 0.3)
    ax.set_ylim(-0.3, L + 0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('x east [m]')
    ax.set_ylabel('y north [m]')
    ax.grid(alpha=0.25)


def draw_plan(ax, plan, label=True):
    col = COLORS.get(plan.mode, 'k')
    ex = plan.extras
    if plan.mode == 'fast' and 'edges' in ex:
        for i, j in ex['edges']:
            a, b = ex['nodes'][i], ex['nodes'][j]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=0.4, alpha=0.35)
    if plan.mode == 'safe' and 'gvd' in ex:
        g = ex['grid']
        r, c = np.nonzero(ex['gvd'])
        ax.scatter(g.xs[c], g.xs[r], s=0.6, color='0.15', alpha=0.5, lw=0)
    xs, ys = zip(*[p[:2] for p in plan.path])
    ax.plot(xs, ys, '-o', color=col, lw=2, ms=4, label=plan.mode if label else None)


def plot_plans(world, plans, png, title=''):
    plt = _pyplot()
    if plt is None:
        return False
    field = None
    for p in plans:
        if 'grid' in p.extras:
            field = (p.extras['grid'].xs, p.extras['grid'].D)
    if field is None:
        from robot_control.gvd_planner import ClearanceGrid
        g = ClearanceGrid(world, 0.03)
        field = (g.xs, g.D)
    fast = [p for p in plans if p.mode == 'fast']
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_world(ax, world, fast[0].rho if fast else None, field)
    for p in plans:
        draw_plan(ax, p)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return True


def plot_run(world, plan, T, P, S, d_est, Q, png, title=''):
    plt = _pyplot()
    if plt is None:
        return False
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.2),
                                  gridspec_kw={'width_ratios': [1.1, 1]})
    draw_world(ax, world, plan['rho'] if plan and plan['mode'] == 'fast' else None)
    if plan:
        xs, ys = zip(*plan['path'])
        ax.plot(xs, ys, '--o', color=COLORS.get(plan['mode'], 'k'), lw=1.5, ms=4,
                label=f'plan ({plan["mode"]})')
    ax.plot(P[:, 0], P[:, 1], '-', color='#2f45b0', lw=1.5, label='EKF estimate')
    for k in range(0, len(P), max(1, len(P) // 15)):
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((P[k, 0], P[k, 1]), 4 * S[k, 0], 4 * S[k, 1], fill=False,
                             ec='#2f45b0', lw=0.6, alpha=0.6))
    if Q is not None:
        ax.plot(Q[:, 0], Q[:, 1], '-', color='#2c7a3f', lw=1.2, label='true (sim)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title)
    ax2.plot(T, d_est - world.robot_radius, color='#2f45b0', label='estimated body gap')
    if Q is not None:
        tq = np.linspace(T[0], T[-1], len(Q))
        d_true, _ = world.clearance(Q[:, 0], Q[:, 1])
        ax2.plot(tq, d_true - world.robot_radius, color='#2c7a3f', label='true body gap (sim)')
    ax2.axhline(0.0, color='r', lw=1, label='contact')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('gap between robot body and nearest obstacle [m]')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return True


# --- main ----------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog='hw4', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan', help='plan and write waypoint files + plots')
    p.add_argument('mode', nargs='?', default='all', choices=list(PLANNERS) + ['all'])
    p.add_argument('--world', default=default_world_file())
    p.add_argument('--out', help='output folder (default: <workspace>/hw4_logs/plans)')
    p.add_argument('--no-plot', action='store_true')
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser('sim', help='plan, simulate the run, evaluate it')
    p.add_argument('mode', choices=list(PLANNERS))
    p.add_argument('--world', default=default_world_file())
    p.add_argument('--params', default=default_params_file())
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-plot', action='store_true')
    p.set_defaults(func=cmd_sim)
    p = sub.add_parser('eval', help='evaluate run folders')
    p.add_argument('runs', nargs='*')
    p.add_argument('--world', help='world file (default: the copy in the run folder)')
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--no-plot', action='store_true')
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser('list', help='list runs')
    p.set_defaults(func=cmd_list)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (WorldError, PlanningError, FileNotFoundError) as e:
        print(f'hw4: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
