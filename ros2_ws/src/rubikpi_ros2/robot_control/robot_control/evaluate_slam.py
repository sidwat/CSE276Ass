"""
Evaluate an EKF-SLAM run (log_dir/run_*/) against the measured ground-truth map (HW3).

The filter's map is in the start frame (the robot's unknown start pose). With --start x0 y0
theta0, the measured start pose in the world frame, it is transformed into the world frame:
    p_w = p0 + R0 p_s,  psi_w = psi_s + theta0,  Sigma_w = R0 Sigma_s R0^T
and compared with the ground truth by tag id: per-landmark error, mean error and RMSE, raw and
after a least-squares rigid alignment (which separates map-shape error from start-pose
measurement error), per lap from map_snapshots.csv, and the mean Mahalanobis^2 (NIS) of the
accepted sightings as a consistency check.

    ros2 run robot_control evaluate_slam ~/hw3_logs/run_... --start 0.914 0.914 0
"""

import argparse
import csv
import glob
import math
import os

import numpy as np
from robot_control.localization import load_landmarks
from robot_control.waypoint_controller import wrap_angle

CHI2_2D_95 = 5.991  # 95% of a 2-D Gaussian lies inside the ellipse at this Mahalanobis^2


def read_csv(path):
    """Rows as dicts; numbers become floats, anything else stays a string."""
    def value(v):
        try:
            return float(v)
        except ValueError:
            return v
    with open(path) as f:
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(f)]


def default_gt_file():
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('robot_control'), 'config',
                            'landmarks_hw3_gt.txt')
        return path if os.path.exists(path) else None
    except Exception:
        return None


def map_from_rows(rows):
    """{tag: (mean (x, y, psi), 2x2 position covariance)} from map.csv-style rows."""
    out = {}
    for r in rows:
        cov = np.array([[r['var_x'], r['cov_xy']], [r['cov_xy'], r['var_y']]])
        out[int(r['tag'])] = (np.array([r['x'], r['y'], r['psi']]), cov)
    return out


def rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform_map(landmarks, start):
    """Start frame -> world frame for {tag: (mean, cov)}."""
    x0, y0, t0 = start
    R0 = rot(t0)
    out = {}
    for tag, (m, cov) in landmarks.items():
        p = np.array([x0, y0]) + R0 @ m[:2]
        psi = wrap_angle(m[2] + t0) if not math.isnan(m[2]) else math.nan
        out[tag] = (np.array([p[0], p[1], psi]), R0 @ cov @ R0.T)
    return out


def transform_poses(poses, start):
    x0, y0, t0 = start
    c, s = math.cos(t0), math.sin(t0)
    return [(x0 + c * x - s * y, y0 + s * x + c * y, wrap_angle(th + t0)) for x, y, th in poses]


def best_fit(a, b):
    """Rotation theta* and translation t* minimising sum |R a_i + t - b_i|^2 (2-D, no scale)."""
    a, b = np.asarray(a), np.asarray(b)
    ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
    theta = math.atan2(np.sum(ac[:, 0] * bc[:, 1] - ac[:, 1] * bc[:, 0]),
                       np.sum(ac[:, 0] * bc[:, 0] + ac[:, 1] * bc[:, 1]))
    t = b.mean(axis=0) - rot(theta) @ a.mean(axis=0)
    return theta, t


def ellipse_axes(cov):
    """95% ellipse: semi-axes (major, minor) and the major axis direction."""
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 0.0)
    a, b = math.sqrt(CHI2_2D_95 * evals[1]), math.sqrt(CHI2_2D_95 * evals[0])
    return a, b, math.atan2(evecs[1, 1], evecs[0, 1])


def compare(est, gt):
    """Errors of the estimated map against the ground truth, raw and after best-fit alignment."""
    common = sorted(set(est) & set(gt))
    out = {'common': common, 'missing': sorted(set(gt) - set(est)),
           'extra': sorted(set(est) - set(gt)), 'errors': {}, 'aligned': {}, 'psi': {}}
    if not common:
        return out
    a = [est[t][0][:2] for t in common]
    b = [np.array(gt[t][:2]) for t in common]
    for tag, ai, bi in zip(common, a, b):
        out['errors'][tag] = float(np.linalg.norm(ai - bi))
        if not math.isnan(est[tag][0][2]):
            out['psi'][tag] = abs(wrap_angle(est[tag][0][2] - gt[tag][2]))
    if len(common) >= 2:
        theta, t = best_fit(a, b)
        out['fit'] = (theta, t)
        for tag, ai, bi in zip(common, a, b):
            out['aligned'][tag] = float(np.linalg.norm(rot(theta) @ ai + t - bi))
    return out


def stats(errors):
    v = list(errors.values())
    if not v:
        return math.nan, math.nan, math.nan
    return (sum(v) / len(v), math.sqrt(sum(e * e for e in v) / len(v)), max(v))


def print_comparison(est, gt, title):
    c = compare(est, gt)
    print(f'\n{title}')
    print('  tag   estimate (x, y)     truth (x, y)       error [m]  aligned [m]  '
          '95% ellipse a x b [cm]')
    for tag in c['common']:
        m, cov = est[tag]
        a, b, _ = ellipse_axes(cov)
        aligned = c['aligned'].get(tag, math.nan)
        print(f'  {tag:<4}  ({m[0]:6.3f}, {m[1]:6.3f})   ({gt[tag][0]:6.3f}, {gt[tag][1]:6.3f})'
              f'   {c["errors"][tag]:8.3f}   {aligned:9.3f}    {100 * a:5.1f} x {100 * b:4.1f}')
    mean, rmse, worst = stats(c['errors'])
    print(f'  Raw:     mean error {mean:.3f} m, RMSE {rmse:.3f} m, max {worst:.3f} m '
          f'over {len(c["common"])} landmarks')
    if c['aligned']:
        mean, rmse, worst = stats(c['aligned'])
        theta, t = c['fit']
        print(f'  Aligned: mean error {mean:.3f} m, RMSE {rmse:.3f} m, max {worst:.3f} m '
              f'(best fit moves the map by ({t[0]:+.3f}, {t[1]:+.3f}) m, '
              f'{math.degrees(theta):+.1f} deg)')
    if c['psi']:
        print(f'  Tag orientation error: mean {math.degrees(np.mean(list(c["psi"].values()))):.1f}'
              f' deg')
    if c['missing']:
        print(f'  In the ground truth but never mapped: {c["missing"]}')
    if c['extra']:
        print(f'  Mapped but not in the ground truth: {c["extra"]}')
    return c


def print_consistency(measurements):
    events = {}
    for m in measurements:
        events[m['event']] = events.get(m['event'], 0) + 1
    print('\nSightings: ' + ', '.join(f'{n} {e}' for e, n in sorted(events.items())))
    d2 = [m['mahalanobis2'] for m in measurements if m['event'] == 'updated']
    if d2:
        gated = events.get('rejected', 0) / (len(d2) + events.get('rejected', 0))
        print(f'Mean Mahalanobis^2 (NIS) of accepted updates {np.mean(d2):.2f}, median '
              f'{np.median(d2):.2f}; {100 * gated:.1f}% gated out')
        print('With landmark_dof 3 the mean should be close to 3 (2 for dof 2). Much higher: the '
              'noise values are too small (overconfident); much lower: too large.')


def draw_landmarks(ax, est, gt, ellipse_scale, color='tab:blue', label=True):
    from matplotlib.patches import Ellipse
    for tag, (m, cov) in est.items():
        a, b, angle = ellipse_axes(cov)
        ax.add_patch(Ellipse((m[0], m[1]), 2 * a * ellipse_scale, 2 * b * ellipse_scale,
                             angle=math.degrees(angle), facecolor=color, alpha=0.25,
                             edgecolor=color, linewidth=1.2, zorder=4))
        ax.plot(m[0], m[1], 'x', color=color, markersize=7, zorder=5)
        if label:
            ax.annotate(f'{tag}', (m[0], m[1]), textcoords='offset points', xytext=(5, -12),
                        color=color, fontsize=8)
        if tag in gt:
            ax.plot([m[0], gt[tag][0]], [m[1], gt[tag][1]], color='0.4', linewidth=0.8, zorder=3)


def draw_truth(ax, gt):
    for i, (tag, (lx, ly, lyaw)) in enumerate(gt.items()):
        half = 0.0865  # 17.3 cm tag
        ax.plot([lx - half * math.sin(lyaw), lx + half * math.sin(lyaw)],
                [ly + half * math.cos(lyaw), ly - half * math.cos(lyaw)],
                color='black', linewidth=4, zorder=3, label='ground truth tag' if i == 0 else None)
        ax.annotate(f'tag {tag}', (lx, ly), textcoords='offset points', xytext=(6, 6), fontsize=8)


def finish(ax, title, arena):
    if arena:
        ax.plot([0, arena, arena, 0, 0], [0, 0, arena, arena, 0], color='0.6', linewidth=1,
                linestyle='--', label='arena')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize='small')
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', nargs='?',
                        help='a run_* folder (default: the newest one in ~/hw3_logs)')
    parser.add_argument('--gt', help='ground-truth landmarks, world frame (default: the '
                        'installed config/landmarks_hw3_gt.txt)')
    parser.add_argument('--start', nargs=3, type=float, metavar=('X0', 'Y0', 'THETA0'),
                        help='measured start pose in the world frame [m, m, rad] (default: '
                        'start_pose.txt in the run folder, else the start frame is used)')
    parser.add_argument('--arena', type=float, default=3.048, help='arena side [m], 0: none')
    parser.add_argument('--ellipse-scale', type=float, default=1.0,
                        help='draw the 95%% ellipses this many times larger')
    args = parser.parse_args()

    run_dir = args.run_dir or max(glob.glob(os.path.expanduser('~/hw3_logs/run_*')))
    trajectory = read_csv(os.path.join(run_dir, 'trajectory.csv'))
    map_path = os.path.join(run_dir, 'map.csv')
    if not os.path.exists(map_path):
        raise SystemExit(f'{map_path} not found: was this run made with use_slam?')
    est = map_from_rows(read_csv(map_path))
    snapshots_path = os.path.join(run_dir, 'map_snapshots.csv')
    snapshots = read_csv(snapshots_path) if os.path.exists(snapshots_path) else []
    measurements_path = os.path.join(run_dir, 'measurements.csv')
    measurements = read_csv(measurements_path) if os.path.exists(measurements_path) else []

    start = args.start
    start_file = os.path.join(run_dir, 'start_pose.txt')
    if start is None and os.path.exists(start_file):
        start = [float(v) for v in open(start_file).read().split()]
    world = start is not None
    start = start or (0.0, 0.0, 0.0)
    gt_path = args.gt or default_gt_file()
    gt = load_landmarks(gt_path) if gt_path and world else {}

    print(f'Run {run_dir}: {trajectory[-1]["t"]:.1f} s, {len(est)} landmarks mapped '
          f'(ids {sorted(est)})')
    if world:
        print(f'Start pose in the world frame: ({start[0]:.3f}, {start[1]:.3f}, '
              f'{math.degrees(start[2]):.1f} deg)' + (f'; ground truth {gt_path}' if gt else ''))
    else:
        print('No --start given: everything stays in the start frame, no ground-truth comparison')
    est_w = transform_map(est, start)

    if gt:
        print_comparison(est_w, gt, 'Final map vs ground truth (world frame)')

    laps = sorted({int(r['lap']) for r in snapshots})
    lap_maps = {lap: transform_map(map_from_rows([r for r in snapshots if int(r['lap']) == lap]),
                                   start) for lap in laps}
    lap_rows = []
    if laps:
        print('\nPer lap (map at the end of each lap)')
        print('  lap  N   mean err [m]  aligned [m]  mean 95% ellipse area [cm^2]  '
              'mean sigma_xy [cm]')
        for lap in laps:
            m = lap_maps[lap]
            areas = [math.pi * a * b for a, b, _ in (ellipse_axes(c) for _, c in m.values())]
            sig = [math.sqrt(np.trace(c) / 2) for _, c in m.values()]
            c = compare(m, gt) if gt else {'errors': {}, 'aligned': {}}
            raw, aligned = stats(c['errors'])[0], stats(c['aligned'])[0]
            lap_rows.append((lap, len(m), raw, aligned, np.mean(areas) * 1e4))
            print(f'  {lap:<4} {len(m):<3} {raw:11.3f}  {aligned:11.3f}  '
                  f'{np.mean(areas) * 1e4:26.2f}  {100 * np.mean(sig):18.2f}')

    if measurements and 'event' in measurements[0]:
        print_consistency(measurements)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frame = 'world' if world else 'start'
    poses = transform_poses([(p['x'], p['y'], p['theta']) for p in trajectory], start)
    odom = (transform_poses([(p['odom_x'], p['odom_y'], p['odom_theta']) for p in trajectory],
                            start) if 'odom_x' in trajectory[0] else [])
    truth_path = os.path.join(run_dir, 'truth.csv')  # simulated runs only
    truth = (transform_poses([(p['x'], p['y'], p['theta']) for p in read_csv(truth_path)], start)
             if os.path.exists(truth_path) else [])
    scale_note = f' (ellipses x{args.ellipse_scale:g})' if args.ellipse_scale != 1 else ''

    # 1. Trajectory and map with 95% covariance ellipses
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_truth(ax, gt)
    if odom:
        ax.plot([p[0] for p in odom], [p[1] for p in odom], color='tab:green', linestyle=':',
                linewidth=1, label='dead reckoning (motion model only)')
    if truth:
        ax.plot([p[0] for p in truth], [p[1] for p in truth], color='0.3', linewidth=1,
                label='true path (simulation)')
    ax.plot([p[0] for p in poses], [p[1] for p in poses], color='red', linewidth=1.2,
            label='EKF-SLAM trajectory')
    ax.plot(poses[0][0], poses[0][1], 'o', color='red', markersize=8, label='start')
    draw_landmarks(ax, est_w, gt, args.ellipse_scale)
    ax.plot([], [], 'x', color='tab:blue', label='estimated landmark, 95% ellipse')
    finish(ax, f'EKF-SLAM map and trajectory ({frame} frame){scale_note}',
           args.arena if world else 0)
    out = os.path.join(run_dir, 'slam_map.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nWrote {out}')

    # 2. State size and uncertainty over time, per-lap error
    n_plots = 3 if lap_rows else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4))
    t = [p['t'] for p in trajectory]
    if 'n_landmarks' in trajectory[0]:
        axes[0].step(t, [p['n_landmarks'] for p in trajectory], where='post')
    axes[0].set(xlabel='t [s]', ylabel='landmarks in the state, N',
                title='State grows as tags are discovered')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, [100 * math.hypot(p['std_x'], p['std_y']) for p in trajectory],
                 label='position std [cm]')
    axes[1].plot(t, [math.degrees(p['std_theta']) for p in trajectory], label='heading std [deg]')
    axes[1].set(xlabel='t [s]', title='Robot pose uncertainty')
    axes[1].legend(fontsize='small')
    axes[1].grid(True, alpha=0.3)
    if lap_rows:
        ax = axes[2]
        lap_ids = [r[0] for r in lap_rows]
        if gt:
            ax.plot(lap_ids, [100 * r[2] for r in lap_rows], 'o-', label='mean error')
            ax.plot(lap_ids, [100 * r[3] for r in lap_rows], 's--', label='aligned error')
            ax.set_ylim(bottom=0)
            ax.legend(loc='lower left', fontsize='small')
        ax.set(xlabel='lap', ylabel='landmark error [cm]', title='Map after each lap',
               xticks=lap_ids)
        ax.grid(True, alpha=0.3)
        area_ax = ax.twinx()
        area_ax.plot(lap_ids, [r[4] for r in lap_rows], '^:', color='tab:green')
        area_ax.set_ylabel('mean 95% ellipse area [cm^2]', color='tab:green')
        area_ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = os.path.join(run_dir, 'slam_stats.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'Wrote {out}')

    # 3. The map at the end of each lap
    if laps:
        fig, axes = plt.subplots(1, len(laps), figsize=(5 * len(laps), 5), squeeze=False)
        for ax, lap in zip(axes[0], laps):
            draw_truth(ax, gt)
            draw_landmarks(ax, lap_maps[lap], gt, args.ellipse_scale)
            finish(ax, f'After lap {lap}{scale_note}', args.arena if world else 0)
        fig.tight_layout()
        out = os.path.join(run_dir, 'slam_laps.png')
        fig.savefig(out, dpi=120, bbox_inches='tight')
        print(f'Wrote {out}')


if __name__ == '__main__':
    main()
