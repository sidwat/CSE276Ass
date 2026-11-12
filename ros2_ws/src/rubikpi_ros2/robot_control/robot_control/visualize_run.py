"""
Plot the map (landmarks, waypoints) with the motion-model, landmark-only and EKF-corrected paths.

  visualize_run sim [--seed N] [--unit 0.5]   offline: real EKF + controller on a synthetic robot
  visualize_run run [RUN_DIR] [--unit 0.5]    report on a velocity_control log (default: newest)

Paths: dead reckoning only (what the motion model believes), the pose each tag sighting implies by
itself (landmarks only), and the EKF estimate that fuses both and drives the robot. The sim also
knows the true pose. --unit scales the map, e.g. --unit 0.5 draws everything in units of 0.5 m.

A run report (written automatically when velocity_control exits) is, in the run folder:
  map.png       map with all paths, total correction and heading correction over time
  states.png    x, y, theta per tick: corrected state (+-2 sigma), motion-model prediction,
                dead reckoning and landmark-only poses
  commands.png  v, w (requested and sent), L/R wheel commands and the correction per tick
  summary.txt   timing, distance, sightings per tag and the estimate at each waypoint
  final_*.png/svg, poses.html   every estimated pose as an arrow (see pose_plots)
"""

import argparse
import csv
import glob
import math
import os

import numpy as np

from robot_control.localization import (
    config_dir, load_landmarks, NoiseParams, pose_from_measurement, PoseEKF)
from robot_control.waypoint_controller import (
    body_to_wheel_cmds, Gains, load_waypoints, WaypointController, wheel_cmds_to_body, wrap_angle)

CONFIG = config_dir()


def load_params(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)['velocity_control_node']['ros__parameters']


def simulate(landmarks, waypoints, params, seed=1, use_vision=True, speed_scale=0.9,
             turn_scale=1.07, drift=0.03, dropout=0.1, fov=math.radians(100),
             tag_view=math.radians(70),
             max_time=300.0):
    """Closed-loop run against a synthetic robot whose real motion differs from the model."""
    rng = np.random.default_rng(seed)
    p = params
    noise = NoiseParams(**{k: p[k] for k in vars(NoiseParams()) if k in p})
    gains = Gains(**{k: float(p[k]) for k in vars(Gains()) if k in p})
    cam_x, cam_y, cam_yaw = (float(p.get(k, 0.0)) for k in ('camera_x', 'camera_y', 'camera_yaw'))
    max_range = float(p.get('max_tag_range', 3.0))
    spu, base = float(p['speed_per_unit']), float(p['wheel_base'])
    max_cmd, pause = float(p.get('max_wheel_cmd', 0.5)), float(p.get('pause_at_waypoint', 1.0))
    ekf = PoseEKF(noise=noise)
    controller = WaypointController(waypoints, gains)
    truth, odom = np.zeros(3), np.zeros(3)
    v = w = 0.0
    hold_until = 0.0
    dt, camera_dt = 0.05, 0.1
    out = {'t': [], 'ekf': [], 'std': [], 'odom': [], 'truth': [], 'vision': [], 'reached': []}
    for step in range(int(max_time / dt)):
        t = step * dt
        # motion: the model integrates the commanded velocity, the real robot does something else
        heading = odom[2] + w * dt / 2
        odom += [v * math.cos(heading) * dt, v * math.sin(heading) * dt, w * dt]
        odom[2] = wrap_angle(odom[2])
        tv, tw = v * speed_scale, w * turn_scale + (drift * math.copysign(1, v) if v else 0.0)
        heading = truth[2] + tw * dt / 2
        truth += [tv * math.cos(heading) * dt, tv * math.sin(heading) * dt, tw * dt]
        truth[2] = wrap_angle(truth[2])
        ekf.predict(v, w, dt, t)
        # camera
        if use_vision and step % round(camera_dt / dt) == 0:
            c, s = math.cos(truth[2]), math.sin(truth[2])
            cx, cy = truth[0] + c * cam_x - s * cam_y, truth[1] + s * cam_x + c * cam_y
            for tag_id, (lx, ly, lyaw) in landmarks.items():
                to_tag = math.atan2(ly - cy, lx - cx)
                if (math.hypot(lx - cx, ly - cy) > max_range or rng.random() < dropout
                        or abs(wrap_angle(to_tag - truth[2] - cam_yaw)) > fov / 2
                        or abs(wrap_angle(math.atan2(cy - ly, cx - lx) - lyaw)) > tag_view):
                    continue
                dx, dy = lx - truth[0], ly - truth[1]
                r = math.hypot(dx, dy)
                z = np.array([r, wrap_angle(math.atan2(dy, dx) - truth[2]),
                              wrap_angle(lyaw - truth[2])])
                z += rng.normal(0, [noise.range_std + noise.range_std_per_m * r,
                                    noise.bearing_std, noise.orientation_std])
                ok, d2 = ekf.correct(z, (lx, ly, lyaw), t)
                out['vision'].append((t, tag_id, *pose_from_measurement(z, (lx, ly, lyaw)), ok))
        # control on the EKF estimate
        if t < hold_until or controller.done:
            v = w = 0.0
        else:
            cv, cw, reached = controller.update(*ekf.pose, dt)
            if reached is not None:
                hold_until = t + pause
                out['reached'].append((reached, t))
            left, right = body_to_wheel_cmds(cv, cw, spu, base, max_cmd)
            v, w = wheel_cmds_to_body(left, right, spu, base)
        for key, value in (('t', t), ('ekf', ekf.pose), ('std', ekf.std), ('odom', odom.copy()),
                           ('truth', truth.copy())):
            out[key].append(value)
        if controller.done and t > hold_until + 1.0:
            break
    return {k: (np.array(v) if k != 'vision' and k != 'reached' else v) for k, v in out.items()}


def read_csv(path):
    """Rows of a CSV file as dicts: numbers as float (blank as nan), anything else as text."""
    def value(text):
        try:
            return float(text) if text != '' else math.nan
        except ValueError:
            return text
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(f)]


def read_run(run_dir, landmarks):
    ticks = read_csv(os.path.join(run_dir, 'ticks.csv'))
    if not ticks:
        raise SystemExit(f'no ticks.csv rows in {run_dir}: was velocity_control run with log_dir?')
    # The pose each sighting implies by itself, from the logged (range, bearing, orientation)
    vision, latency = [], []
    for r in read_csv(os.path.join(run_dir, 'measurements.csv')):
        tag = int(r['tag'])
        if r.get('status') == 'out_of_range':
            continue  # ignored by the EKF (beyond max_tag_range)
        if not math.isnan(r.get('latency', math.nan)):
            latency.append(r['latency'])
        if tag in landmarks:
            pose = pose_from_measurement((r['range'], r['bearing'], r['orientation']),
                                         landmarks[tag])
            vision.append((r['t'], tag, *pose, bool(r['accepted'])))
    reached = read_csv(os.path.join(run_dir, 'waypoints.csv'))

    def col(*keys):
        values = np.array([[r[k] for k in keys] for r in ticks], dtype=float)
        return values[:, 0] if len(keys) == 1 else values
    return {'t': col('t'), 'dt': col('dt'),
            'ekf': col('corr_x', 'corr_y', 'corr_theta'),
            'std': col('std_x', 'std_y', 'std_theta'),
            'model': col('model_x', 'model_y', 'model_theta'),
            'odom': col('odom_x', 'odom_y', 'odom_theta'),
            'v_cmd': col('v_cmd'), 'w_cmd': col('w_cmd'), 'v': col('v'), 'w': col('w'),
            'left': col('left'), 'right': col('right'),
            'mode': [r['mode'] for r in ticks],
            'vision': vision,
            'reached': [(int(r['index']), r['t']) for r in reached],
            'arrivals': reached, 'latency': np.array(latency)}


def wrap_array(a):
    return np.arctan2(np.sin(a), np.cos(a))


def draw(data, landmarks, waypoints, unit, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    k = 1.0 / unit
    label = 'm' if unit == 1.0 else f'units of {unit:g} m'
    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(grid[:, 0])
    # map: reference trajectory, waypoints, landmarks
    wp = np.array(waypoints) * [k, k, 1]
    ax.plot(wp[:, 0], wp[:, 1], color='tab:blue', linestyle=':', linewidth=1, zorder=1)
    for i, (gx, gy, gt) in enumerate(wp):
        ax.annotate('', xy=(gx + 0.25 * k * math.cos(gt), gy + 0.25 * k * math.sin(gt)),
                    xytext=(gx, gy), arrowprops={'arrowstyle': '->', 'color': 'tab:blue'})
        ax.annotate(f'W{i}', (gx, gy), textcoords='offset points', xytext=(-14, -14),
                    color='tab:blue', fontweight='bold')
    ax.scatter(wp[:, 0], wp[:, 1], s=140, facecolors='none', edgecolors='tab:blue', zorder=3,
               label='waypoints')
    for tag_id, (lx, ly, lyaw) in landmarks.items():
        h = 0.09 * k
        ax.plot([lx * k - h * math.sin(lyaw), lx * k + h * math.sin(lyaw)],
                [ly * k + h * math.cos(lyaw), ly * k - h * math.cos(lyaw)],
                color='black', linewidth=5, zorder=3)
        tip = (lx * k + 0.25 * k * math.cos(lyaw), ly * k + 0.25 * k * math.sin(lyaw))
        ax.annotate('', xy=tip,
                    xytext=(lx * k, ly * k), arrowprops={'arrowstyle': '->', 'color': '0.4'})
        ax.annotate(f'tag {tag_id}', (lx * k, ly * k), textcoords='offset points', xytext=(7, 7))
    ax.plot([], [], color='black', linewidth=5, label='landmarks (arrow = face normal)')
    # paths
    if 'truth' in data:
        ax.plot(data['truth'][:, 0] * k, data['truth'][:, 1] * k, color='tab:green', linewidth=2,
                label='true path (sim only)', zorder=2)
    ax.plot(data['odom'][:, 0] * k, data['odom'][:, 1] * k, color='tab:orange', linestyle='--',
            linewidth=1.5, label='motion model only (dead reckoning)', zorder=2)
    good = [v for v in data['vision'] if v[5]]
    bad = [v for v in data['vision'] if not v[5]]
    if good:
        ax.scatter([v[2] * k for v in good], [v[3] * k for v in good], s=10, color='tab:purple',
                   alpha=0.5, label='landmark-only pose (accepted)', zorder=2)
    if bad:
        ax.scatter([v[2] * k for v in bad], [v[3] * k for v in bad], s=18, marker='x',
                   color='0.5', label='landmark-only pose (rejected)', zorder=2)
    t = data['t']
    ax.plot(data['ekf'][:, 0] * k, data['ekf'][:, 1] * k, color='tab:red', linewidth=1.5,
            label='EKF corrected', zorder=4)
    dots = np.r_[True, np.diff(np.floor((t - t[0]) / 0.5)) > 0]
    ax.scatter(data['ekf'][dots, 0] * k, data['ekf'][dots, 1] * k, s=14, color='tab:red',
               zorder=5, label='EKF, one dot per 0.5 s')
    ax.set_xlabel(f'x [{label}]')
    ax.set_ylabel(f'y [{label}]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_title(title)

    # how far the corrections move the estimate over time
    ax2 = fig.add_subplot(grid[0, 1])
    sigma = np.hypot(data['std'][:, 0], data['std'][:, 1]) * k
    ax2.plot(t, np.hypot(*(data['ekf'] - data['odom'])[:, :2].T) * k, color='tab:red',
             label='|EKF - motion model|: total correction applied')
    if 'truth' in data:
        ax2.plot(t, np.hypot(*(data['odom'] - data['truth'])[:, :2].T) * k, color='tab:orange',
                 linestyle='--', label='motion model error vs truth')
        ax2.plot(t, np.hypot(*(data['ekf'] - data['truth'])[:, :2].T) * k, color='tab:green',
                 label='EKF error vs truth')
        ax2.scatter([v[0] for v in data['vision']],
                    [math.hypot(v[2] - np.interp(v[0], t, data['truth'][:, 0]),
                                v[3] - np.interp(v[0], t, data['truth'][:, 1])) * k
                     for v in data['vision']],
                    s=6, color='tab:purple', alpha=0.4, label='landmark-only error vs truth')
    ax2.plot(t, sigma, color='0.5', linewidth=1, label='EKF position std')
    for index, when in data['reached']:
        ax2.axvline(when, color='tab:blue', alpha=0.3)
        ax2.annotate(f'W{index}', (when, ax2.get_ylim()[1]), color='tab:blue', fontsize=8,
                     va='top')
    ax2.set_ylabel(f'position [{label}]')
    ax2.set_xlabel('time [s]')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)
    ax2.set_title('Position correction / error')

    ax3 = fig.add_subplot(grid[1, 1])
    ax3.plot(t, np.degrees(wrap_array(data['ekf'][:, 2] - data['odom'][:, 2])), color='tab:red',
             label='EKF - motion model: heading correction')
    if 'truth' in data:
        ax3.plot(t, np.degrees(wrap_array(data['odom'][:, 2] - data['truth'][:, 2])),
                 color='tab:orange', linestyle='--', label='motion model heading error')
        ax3.plot(t, np.degrees(wrap_array(data['ekf'][:, 2] - data['truth'][:, 2])),
                 color='tab:green', label='EKF heading error')
    ax3.set_ylabel('heading [deg]')
    ax3.set_xlabel('time [s]')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)
    ax3.set_title('Heading correction / error')
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f'Wrote {output}')


def mark_waypoints(ax, reached):
    for index, when in reached:
        ax.axvline(when, color='tab:blue', alpha=0.3, linewidth=1)
        ax.annotate(f'W{index}', (when, 1.0), xycoords=('data', 'axes fraction'),
                    color='tab:blue', fontsize=8, va='top', ha='left')


def draw_states(data, output, title):
    """x, y, theta per tick: corrected state, motion-model prediction, dead reckoning, vision."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = data['t']
    corrected_theta = np.unwrap(data['ekf'][:, 2])

    def near(theta, at):
        """Shift an angle by whole turns to sit next to the unwrapped corrected heading."""
        ref = np.interp(at, t, corrected_theta)
        return theta + 2 * math.pi * np.round((ref - theta) / (2 * math.pi))

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    names = ['x [m]', 'y [m]', 'theta [deg]']
    good = [v for v in data['vision'] if v[5]]
    bad = [v for v in data['vision'] if not v[5]]
    for i, ax in enumerate(axes):
        if i < 2:
            corr, model, odom = data['ekf'][:, i], data['model'][:, i], data['odom'][:, i]
            band = 2 * data['std'][:, i]
            vis_good = np.array([v[2 + i] for v in good])
            vis_bad = np.array([v[2 + i] for v in bad])
        else:
            corr = np.degrees(corrected_theta)
            model = np.degrees(near(data['model'][:, 2], t))
            odom = np.degrees(near(data['odom'][:, 2], t))
            band = np.degrees(2 * data['std'][:, 2])
            vis_good = np.degrees([near(v[4], v[0]) for v in good])
            vis_bad = np.degrees([near(v[4], v[0]) for v in bad])
        ax.fill_between(t, corr - band, corr + band, color='tab:red', alpha=0.12,
                        label='corrected +- 2 sigma')
        ax.plot(t, odom, color='tab:orange', linestyle='--', linewidth=1.2,
                label='dead reckoning only (never corrected)')
        ax.plot(t, model, color='tab:blue', linewidth=0.8,
                label='motion model from the corrected state (prior)')
        ax.plot(t, corr, color='tab:red', linewidth=1.4, label='corrected state (EKF)')
        if good:
            ax.scatter([v[0] for v in good], vis_good, s=12, color='tab:purple', zorder=5,
                       label='landmark-only pose (accepted)')
        if bad:
            ax.scatter([v[0] for v in bad], vis_bad, s=18, marker='x', color='0.5', zorder=5,
                       label='landmark-only pose (rejected)')
        mark_waypoints(ax, data['reached'])
        ax.set_ylabel(names[i])
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc='best', ncol=2)
    axes[0].set_title(f'{title}: state per control tick')
    axes[-1].set_xlabel('time [s]')
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f'Wrote {output}')


def draw_commands(data, output, title):
    """Velocity and wheel commands per tick, and how much each tick's sightings corrected."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = data['t']
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].step(t, data['v_cmd'], where='post', color='0.6', linestyle='--',
                 label='v requested by the controller')
    axes[0].step(t, data['v'], where='post', color='tab:green',
                 label='v sent (after saturation)')
    axes[0].set_ylabel('v [m/s]')
    axes[1].step(t, data['w_cmd'], where='post', color='0.6', linestyle='--',
                 label='w requested by the controller')
    axes[1].step(t, data['w'], where='post', color='tab:purple',
                 label='w sent (after saturation)')
    axes[1].set_ylabel('w [rad/s]')
    axes[2].step(t, data['left'], where='post', color='tab:blue', label='L (after trim)')
    axes[2].step(t, data['right'], where='post', color='tab:red', label='R (after trim)')
    axes[2].axhline(0, color='k', linewidth=0.5)
    axes[2].set_ylabel('wheel command')
    # correction applied by the sightings between tick k and k+1: corrected[k+1] - prior[k]
    ax = axes[3]
    shift = np.hypot(*(data['ekf'][1:, :2] - data['model'][:-1, :2]).T) * 100
    turn = np.degrees(wrap_array(data['ekf'][1:, 2] - data['model'][:-1, 2]))
    ax.plot(t[1:], shift, color='tab:red', linewidth=1, label='position correction [cm]')
    ax.set_ylabel('position correction [cm]')
    twin = ax.twinx()
    twin.plot(t[1:], turn, color='tab:blue', linewidth=1, label='heading correction [deg]')
    twin.set_ylabel('heading correction [deg]')
    for v in data['vision']:
        ax.axvline(v[0], ymax=0.06, color='tab:purple' if v[5] else '0.5', linewidth=1)
    lines = ax.get_legend_handles_labels()
    twin_lines = twin.get_legend_handles_labels()
    ax.legend(lines[0] + twin_lines[0], lines[1] + twin_lines[1], fontsize=8, loc='upper right')
    ax.set_title('Correction per tick (ticks at the bottom: sightings, purple = accepted)',
                 fontsize=9)
    for a in axes:
        mark_waypoints(a, data['reached'])
        a.grid(True, alpha=0.3)
    for a in axes[:3]:
        a.legend(fontsize=8, loc='upper right')
    axes[0].set_title(f'{title}: commands per control tick')
    axes[-1].set_xlabel('time [s]')
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f'Wrote {output}')


def summarize_run(data, waypoints):
    t, ekf = data['t'], data['ekf']
    moving = np.abs(data['v']) + np.abs(data['w']) > 0
    lines = [f'Duration {t[-1] - t[0]:.1f} s, {len(t)} ticks, tick period mean '
             f'{np.mean(data["dt"]) * 1000:.0f} ms, max {np.max(data["dt"]) * 1000:.0f} ms',
             f'Driving for {np.sum(data["dt"][moving]):.1f} s',
             f'Corrected path length {np.hypot(*np.diff(ekf[:, :2], axis=0).T).sum():.2f} m, '
             f'rotation {np.degrees(np.abs(wrap_array(np.diff(ekf[:, 2]))).sum()):.0f} deg',
             f'Final pose: corrected ({ekf[-1, 0]:.3f}, {ekf[-1, 1]:.3f}, '
             f'{math.degrees(ekf[-1, 2]):.1f} deg), dead reckoning ({data["odom"][-1, 0]:.3f}, '
             f'{data["odom"][-1, 1]:.3f}, {math.degrees(data["odom"][-1, 2]):.1f} deg)']
    duration = max(t[-1] - t[0], 1e-9)
    tags = sorted({v[1] for v in data['vision']})
    total = len(data['vision'])
    accepted = sum(1 for v in data['vision'] if v[5])
    lines.append(f'Rates: prediction (control tick) {len(t) / duration:.1f} Hz, '
                 f'EKF corrections {accepted / duration:.2f} Hz, '
                 f'landmark-only poses {total / duration:.2f} Hz')
    if len(data['latency']):
        lines.append(f'Camera-to-EKF latency: mean {np.mean(data["latency"]) * 1000:.0f} ms, '
                     f'max {np.max(data["latency"]) * 1000:.0f} ms')
    lines.append(f'Sightings: {total}, {accepted} accepted, {total - accepted} rejected')
    for tag in tags:
        mine = [v for v in data['vision'] if v[1] == tag]
        lines.append(f'  tag {tag}: {len(mine)} sightings ({len(mine) / duration:.2f}/s), '
                     f'{sum(1 for v in mine if v[5])} accepted')
    lines.append('Waypoints (estimate at arrival; measure the true pose at the floor marks):')
    for r in data['arrivals']:
        goal = waypoints[int(r['index'])] if int(r['index']) < len(waypoints) else (
            r['goal_x'], r['goal_y'], r['goal_theta'])
        index = int(r['index'])
        lines.append(f'  W{index} t={r["t"]:6.1f} s  goal ({goal[0]:.2f}, {goal[1]:.2f}, '
                     f'{math.degrees(goal[2]):.0f} deg)  estimate ({r["x"]:.3f}, {r["y"]:.3f}, '
                     f'{math.degrees(r["theta"]):.1f} deg)')
    return '\n'.join(lines)


def report(run_dir, unit=1.0, waypoints_path=None, landmarks_path=None):
    """Write map.png, states.png, commands.png and summary.txt for a velocity_control run."""
    def pick(name, fallback):
        saved = os.path.join(run_dir, name)
        return saved if os.path.exists(saved) else fallback
    landmarks = load_landmarks(pick('landmarks.txt',
                                    landmarks_path or str(CONFIG / 'landmarks.txt')))
    waypoints = load_waypoints(pick('waypoints.txt',
                                    waypoints_path or str(CONFIG / 'waypoints_hw2.txt')))
    data = read_run(run_dir, landmarks)
    title = os.path.basename(os.path.normpath(run_dir))
    draw(data, landmarks, waypoints, unit, os.path.join(run_dir, 'map.png'), title)
    draw_states(data, os.path.join(run_dir, 'states.png'), title)
    draw_commands(data, os.path.join(run_dir, 'commands.png'), title)
    summary = summarize_run(data, waypoints)
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')
    print(summary)
    from robot_control.pose_plots import make_pose_plots
    make_pose_plots(run_dir)
    print(f'Report written to {run_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['sim', 'run'])
    parser.add_argument('run_dir', nargs='?', help='run mode: a run_* folder (default: newest)')
    parser.add_argument('--landmarks', default=str(CONFIG / 'landmarks.txt'))
    parser.add_argument('--waypoints',
                        help="sim: default waypoints_hw2.txt; run: default the run's own copy")
    parser.add_argument('--params', default=str(CONFIG / 'hw2.yaml'))
    parser.add_argument('--unit', type=float, default=1.0, help='metres per plotted unit')
    parser.add_argument('--seed', type=int, default=1, help='sim: random seed')
    parser.add_argument('--no-vision', action='store_true', help='sim: dead reckoning only')
    parser.add_argument('--output', help='image file')
    args = parser.parse_args()

    if args.mode == 'run':
        run_dir = args.run_dir or max(glob.glob(os.path.expanduser('~/hw2_logs/run_*')))
        report(run_dir, args.unit, args.waypoints, args.landmarks)
        return
    landmarks = load_landmarks(args.landmarks)
    waypoints = load_waypoints(args.waypoints or str(CONFIG / 'waypoints_hw2.txt'))
    data = simulate(landmarks, waypoints, load_params(args.params), args.seed, not args.no_vision)
    last = data['truth'][-1]
    print(f'Finished at {data["t"][-1]:.0f} s, {len(data["reached"])}/{len(waypoints)} '
          f'waypoints; true final pose ({last[0]:.2f}, {last[1]:.2f}, {last[2]:.2f})')
    for index, when in data['reached']:
        i = np.searchsorted(data['t'], when)
        goal, truth = waypoints[index], data['truth'][min(i, len(data['t']) - 1)]
        error = math.hypot(truth[0] - goal[0], truth[1] - goal[1])
        print(f'  W{index} t={when:5.1f}s  true error {error:.3f} m, '
              f'{math.degrees(wrap_angle(truth[2] - goal[2])):+.1f} deg')
    draw(data, landmarks, waypoints, args.unit, args.output or 'hw2_sim.png',
         f'Simulation (seed {args.seed})')


if __name__ == '__main__':
    main()
