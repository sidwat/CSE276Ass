"""
Evaluate HW3 EKF-SLAM runs (log_dir/run_*/ from velocity_control or simulate_slam) for the report.

  ros2 run robot_control evaluate_slam square_3dof_2026-11-12_15-00-00 --gt landmarks_hw3_gt.txt
  ros2 run robot_control evaluate_slam RUN [RUN ...] --gt GT --start 0.914 0.914 0 --out cmp/
  ros2 run robot_control evaluate_slam RUN --sensor-noise      # robot standing still: fit R

The start pose comes from --start, else from RUN/start_pose.txt (fill it in after the run).
Writes into RUN/eval/: map_world.png (trajectory + landmarks with 95% ellipses vs ground truth),
map_aligned.png, map_start_frame.png, timeline.png (N(t), uncertainties, sightings), laps.png
(error and ellipse size per lap), convergence.png (each landmark's error over the run), nis.png
(filter consistency), results.md (tables to paste into the report) and summary.json.
With several runs, also a comparison table (--out, default: next to the first run).
"""

import argparse
import csv
import json
import math
import os

import numpy as np

from robot_control.localization import load_landmarks
from robot_control.waypoint_controller import wrap_angle

CHI2_95_2D = 5.991  # 95% ellipse for a 2D Gaussian
ELLIPSE_SCALE = 1.0  # --ellipse-scale: magnify drawn ellipses (labelled in the legend)


# --- Loading ------------------------------------------------------------------------------------

def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        conv = {}
        for k, v in row.items():
            try:
                conv[k] = float(v)
            except (TypeError, ValueError):
                conv[k] = v
        out.append(conv)
    return out


def read_start_pose(run_dir):
    path = os.path.join(run_dir, 'start_pose.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line:
                parts = line.replace(',', ' ').split()
                if len(parts) == 3:
                    return tuple(float(p) for p in parts)
    return None


def map_from_rows(rows):
    """{tag: (x, y, psi, 2x2 cov, var_psi, sightings)} from map CSV rows."""
    return {int(r['tag']): (r['x'], r['y'], r['psi'],
                            np.array([[r['cov_xx'], r['cov_xy']], [r['cov_xy'], r['cov_yy']]]),
                            r['var_psi'], r['sightings'])
            for r in rows}


class Run:
    def __init__(self, run_dir):
        self.dir = run_dir
        self.name = os.path.basename(os.path.normpath(run_dir))
        p = lambda n: os.path.join(run_dir, n)  # noqa: E731
        with open(p('run_info.json')) as f:
            self.info = json.load(f)
        self.trajectory = read_rows(p('trajectory.csv'))
        self.measurements = read_rows(p('measurements.csv'))
        self.events = read_rows(p('landmark_events.csv'))
        self.waypoints = read_rows(p('waypoints.csv'))
        self.truth = read_rows(p('truth.csv'))
        snap_rows = read_rows(p('map_snapshots.csv'))
        self.snapshots = {}  # label -> (lap, t, map), in order of appearance
        for r in snap_rows:
            label = r['label']
            if label not in self.snapshots:
                self.snapshots[label] = (int(r['lap']), r['t'], [])
            self.snapshots[label][2].append(r)
        self.snapshots = {k: (lap, t, map_from_rows(rows))
                          for k, (lap, t, rows) in self.snapshots.items()}
        final_rows = read_rows(p('map_final.csv'))
        self.final_map = map_from_rows(final_rows) if final_rows else (
            self.snapshots[list(self.snapshots)[-1]][2] if self.snapshots else {})
        self.start = read_start_pose(run_dir)

    @property
    def lap_labels(self):
        return [k for k in self.snapshots if k.startswith('lap')]


# --- Geometry -----------------------------------------------------------------------------------

def rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def to_world(start, x, y, psi=float('nan'), cov=None):
    """Start-frame point (and covariance) into the world frame given the world start pose."""
    x0, y0, t0 = start
    R = rot(t0)
    p = np.array([x0, y0]) + R @ np.array([x, y])
    psi_w = wrap_angle(psi + t0) if math.isfinite(psi) else psi
    cov_w = R @ cov @ R.T if cov is not None else None
    return p, psi_w, cov_w


def best_fit(a, b, scale=False):
    """Least-squares rigid (or similarity) transform taking points a onto b: (theta, t, s)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    A, B = a - ma, b - mb
    theta = math.atan2(np.sum(A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]),
                       np.sum(A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1]))
    s = 1.0
    if scale and np.sum(A * A) > 0:
        s = float(np.sum((A @ rot(theta).T) * B) / np.sum(A * A))
    t = mb - s * rot(theta) @ ma
    return theta, t, s


def ellipse_area(cov):
    return math.pi * CHI2_95_2D * math.sqrt(max(np.linalg.det(cov), 0.0))


def map_errors(slam_map, gt, start=None, align=False):
    """
    Per-landmark world-frame errors for tags in both maps.

    With start, the map is moved by the measured start pose; with align, by the best rigid fit
    to the ground truth (separates map-shape error from start-pose measurement error).
    Returns (rows, transform) where rows are dicts per tag.
    """
    common = sorted(set(slam_map) & set(gt))
    if not common:
        return [], None
    est = []
    for tag in common:
        x, y, psi, cov, var_psi, n = slam_map[tag]
        if start is not None:
            p, psi_w, cov_w = to_world(start, x, y, psi, cov)
        else:
            p, psi_w, cov_w = np.array([x, y]), psi, cov
        est.append((p, psi_w, cov_w, var_psi, n))
    transform = None
    inliers = set(common)
    if align and len(common) >= 2:
        src, dst = [e[0] for e in est], [np.array(gt[k][:2]) for k in common]
        theta, t, _ = best_fit(src, dst)
        if len(common) >= 4:
            # Robust: refit without gross outliers (e.g. a landmark initialised from a bad pose),
            # so one wrong tag does not rotate the whole map; outliers are still reported.
            res = [np.linalg.norm(rot(theta) @ p + t - q) for p, q in zip(src, dst)]
            cut = max(2.5 * float(np.median(res)), 0.15)
            keep = [i for i, r in enumerate(res) if r <= cut]
            if 3 <= len(keep) < len(common):
                theta, t, _ = best_fit([src[i] for i in keep], [dst[i] for i in keep])
                inliers = {common[i] for i in keep}
        transform = (theta, t)
        R = rot(theta)
        est = [(R @ p + t, wrap_angle(psi + theta) if math.isfinite(psi) else psi,
                R @ c @ R.T, vp, n) for p, psi, c, vp, n in est]
    rows = []
    for tag, (p, psi, cov, var_psi, n) in zip(common, est):
        gx, gy, gyaw = gt[tag]
        err = math.hypot(p[0] - gx, p[1] - gy)
        rows.append({
            'tag': tag, 'x': p[0], 'y': p[1], 'psi': psi, 'gt_x': gx, 'gt_y': gy, 'gt_yaw': gyaw,
            'err': err, 'err_x': p[0] - gx, 'err_y': p[1] - gy,
            'err_psi_deg': math.degrees(wrap_angle(psi - gyaw)) if math.isfinite(psi) else None,
            'std_x': math.sqrt(cov[0, 0]), 'std_y': math.sqrt(cov[1, 1]),
            'area95': ellipse_area(cov), 'cov': cov, 'sightings': n,
            'fit_inlier': tag in inliers,
            # Mahalanobis distance of the true position under the filter's covariance
            'nees': float(np.array([p[0] - gx, p[1] - gy])
                          @ np.linalg.solve(cov + 1e-12 * np.eye(2),
                                            np.array([p[0] - gx, p[1] - gy]))),
        })
    return rows, transform


def stats(rows):
    if not rows:
        return {'n': 0, 'mean': None, 'rmse': None, 'max': None}
    e = np.array([r['err'] for r in rows])
    return {'n': len(rows), 'mean': float(e.mean()), 'rmse': float(np.sqrt(np.mean(e ** 2))),
            'max': float(e.max()), 'median': float(np.median(e)),
            'mean_area95': float(np.mean([r['area95'] for r in rows])),
            'mean_nees': float(np.mean([r['nees'] for r in rows]))}


def chi2_pdf(x, k):
    return x ** (k / 2 - 1) * math.exp(-x / 2) / (2 ** (k / 2) * math.gamma(k / 2))


def path_length(traj):
    return sum(math.hypot(b['x'] - a['x'], b['y'] - a['y']) for a, b in zip(traj, traj[1:]))


# --- Plots --------------------------------------------------------------------------------------

def plt_module():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def draw_ellipse(ax, center, cov, **kw):
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    angle = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    k = ELLIPSE_SCALE
    ax.add_patch(Ellipse(center, 2 * k * math.sqrt(CHI2_95_2D * vals[1]),
                         2 * k * math.sqrt(CHI2_95_2D * vals[0]), angle=angle, **kw))


def ellipse_label():
    return '95% covariance' + (f' (drawn x{ELLIPSE_SCALE:g})' if ELLIPSE_SCALE != 1 else '')


def draw_gt(ax, gt, arena):
    if arena:
        ax.plot([0, arena, arena, 0, 0], [0, 0, arena, arena, 0], color='0.4', linewidth=1.5,
                label=f'arena {arena:.3f} m')
    first = True
    for tag, (x, y, yaw) in sorted(gt.items()):
        half = 0.0865
        ax.plot([x - half * math.sin(yaw), x + half * math.sin(yaw)],
                [y + half * math.cos(yaw), y - half * math.cos(yaw)], color='black',
                linewidth=4, solid_capstyle='butt', label='ground-truth tag' if first else None)
        ax.annotate(f'{tag}', (x, y), textcoords='offset points',
                    xytext=(8 * math.cos(yaw), 8 * math.sin(yaw)), fontsize=8, color='black',
                    ha='center', va='center')
        first = False


def lap_colors(plt, n):
    cmap = plt.get_cmap('viridis')
    return [cmap(i / max(n - 1, 1)) for i in range(max(n, 1))]


def draw_trajectory(ax, plt, run, transform, label='SLAM trajectory'):
    traj = run.trajectory
    if not traj:
        return
    laps = sorted({int(p['lap']) for p in traj})
    colors = lap_colors(plt, len(laps))
    for i, lap in enumerate(laps):
        pts = np.array([transform(p['x'], p['y']) for p in traj if int(p['lap']) == lap])
        ax.plot(pts[:, 0], pts[:, 1], color=colors[i], linewidth=1.8,
                label=f'{label}, lap {lap}' if len(laps) > 1 else label)
    odom = np.array([transform(p['odom_x'], p['odom_y']) for p in traj])
    ax.plot(odom[:, 0], odom[:, 1], color='tab:gray', linestyle='--', linewidth=1,
            label='dead reckoning only')
    if run.truth:
        tru = np.array([transform(p['x'], p['y']) for p in run.truth])
        ax.plot(tru[:, 0], tru[:, 1], color='tab:green', linewidth=1, alpha=0.7,
                label='true trajectory (simulation)')
    s = transform(0.0, 0.0)
    ax.plot(*s, marker='*', markersize=14, color='gold', markeredgecolor='black',
            label='start', linestyle='none')


def draw_slam_map(ax, rows_or_map, transform=None):
    first = True
    if isinstance(rows_or_map, list):  # error rows (world frame)
        for r in rows_or_map:
            draw_ellipse(ax, (r['x'], r['y']), r['cov'], facecolor='tab:red', alpha=0.25,
                         edgecolor='tab:red', label=ellipse_label() if first else None)
            ax.plot(r['x'], r['y'], 'x', color='tab:red', markersize=8,
                    label='SLAM landmark' if first else None)
            ax.plot([r['x'], r['gt_x']], [r['y'], r['gt_y']], color='tab:red', linewidth=0.8)
            ax.annotate(f"{r['tag']}: {100 * r['err']:.1f} cm", (r['x'], r['y']),
                        textcoords='offset points', xytext=(5, -12), fontsize=7,
                        color='tab:red')
            first = False
        return
    for tag, (x, y, psi, cov, _, _) in sorted(rows_or_map.items()):
        p, psi_w, cov_w = (np.array([x, y]), psi, cov) if transform is None else transform(
            x, y, psi, cov)
        draw_ellipse(ax, p, cov_w, facecolor='tab:red', alpha=0.25, edgecolor='tab:red',
                     label=ellipse_label() if first else None)
        ax.plot(*p, 'x', color='tab:red', markersize=8, label='SLAM landmark' if first else None)
        if math.isfinite(psi_w):
            ax.annotate('', xy=(p[0] + 0.15 * math.cos(psi_w), p[1] + 0.15 * math.sin(psi_w)),
                        xytext=p, arrowprops={'arrowstyle': '->', 'color': 'tab:red'})
        ax.annotate(f'tag {tag}', p, textcoords='offset points', xytext=(5, 5), fontsize=8,
                    color='tab:red')
        first = False


def finish(ax, title):
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small')
    ax.set_title(title)


def plot_maps(run, gt, out, arena, raw_rows, aligned_rows, transform_aligned):
    plt = plt_module()
    written = []

    # 1. The map as the filter built it, in its own (start) frame: needs no ground truth
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_trajectory(ax, plt, run, lambda x, y: (x, y))
    draw_slam_map(ax, run.final_map)
    finish(ax, f'{run.name}: SLAM map and trajectory in the start frame\n'
               f'({len(run.final_map)} landmarks discovered)')
    written.append(save(fig, out, 'map_start_frame.png'))

    if gt and run.start is not None:
        def tf(x, y):
            return tuple(to_world(run.start, x, y)[0])
        fig, ax = plt.subplots(figsize=(9, 7))
        draw_gt(ax, gt, arena)
        draw_trajectory(ax, plt, run, tf)
        draw_slam_map(ax, raw_rows)
        extra = {t: m for t, m in run.final_map.items() if t not in gt}
        if extra:
            draw_slam_map(ax, extra, lambda x, y, psi, cov: to_world(run.start, x, y, psi, cov))
        s = stats(raw_rows)
        finish(ax, f'{run.name}: world frame via the measured start pose\n'
                   f'mean landmark error {100 * s["mean"]:.1f} cm (RMSE {100 * s["rmse"]:.1f})'
                   if s['n'] else run.name)
        written.append(save(fig, out, 'map_world.png'))

    if gt and transform_aligned is not None:
        theta, t = transform_aligned
        R = rot(theta)
        base = run.start if run.start is not None else (0.0, 0.0, 0.0)

        def tf_al(x, y):
            p = to_world(base, x, y)[0]
            return tuple(R @ p + t)
        fig, ax = plt.subplots(figsize=(9, 7))
        draw_gt(ax, gt, arena)
        draw_trajectory(ax, plt, run, tf_al)
        draw_slam_map(ax, aligned_rows)
        s = stats(aligned_rows)
        finish(ax, f'{run.name}: after best-fit rigid alignment to the ground truth\n'
                   f'mean landmark error {100 * s["mean"]:.1f} cm (RMSE {100 * s["rmse"]:.1f})')
        written.append(save(fig, out, 'map_aligned.png'))
    return written


def _overlay_panel(ax, plt, run, gt, rows, transform, arena, title):
    """Ground-truth arena and tags with the SLAM map and trajectory drawn on top of it."""
    if arena:
        ax.add_patch(plt.Rectangle((0, 0), arena, arena, fill=True, facecolor='0.96',
                                   edgecolor='0.35', linewidth=2, label=f'arena {arena:g} m'))
    traj = run.trajectory
    if traj:
        pts = np.array([transform(p['x'], p['y']) for p in traj])
        ax.plot(pts[:, 0], pts[:, 1], color='tab:blue', linewidth=1.0, alpha=0.6,
                label='SLAM trajectory')
    s = transform(0.0, 0.0)
    ax.plot(*s, marker='*', markersize=16, color='gold', markeredgecolor='black',
            linestyle='none', label='start', zorder=10)
    by_tag = {r['tag']: r for r in rows}
    first_gt = first_est = True
    for tag, (gx, gy, gyaw) in sorted(gt.items()):
        half = 0.0865
        ax.plot([gx - half * math.sin(gyaw), gx + half * math.sin(gyaw)],
                [gy + half * math.cos(gyaw), gy - half * math.cos(gyaw)], color='black',
                linewidth=5, solid_capstyle='butt',
                label='actual tag (measured)' if first_gt else None)
        ax.plot(gx, gy, 'o', color='tab:green', markersize=6, markeredgecolor='black',
                zorder=5)
        first_gt = False
        # labels pushed into the arena, away from the wall
        ix, iy = math.cos(gyaw), math.sin(gyaw)
        r = by_tag.get(tag)
        text = f'tag {tag}\nactual ({gx:.2f}, {gy:.2f})'
        if r is not None:
            text += f"\nSLAM ({r['x']:.2f}, {r['y']:.2f})\nerror {100 * r['err']:.1f} cm"
        ax.annotate(text, (gx, gy), xytext=(gx + 0.55 * ix - 0.25 * abs(iy),
                                            gy + 0.55 * iy - 0.12 * abs(ix)),
                    fontsize=6.5, color='black', va='center',
                    bbox={'boxstyle': 'round,pad=0.2', 'fc': 'white', 'ec': '0.7',
                          'alpha': 0.85})
        if r is None:
            continue
        draw_ellipse(ax, (r['x'], r['y']), r['cov'], facecolor='tab:red', alpha=0.25,
                     edgecolor='tab:red', label=ellipse_label() if first_est else None)
        ax.plot(r['x'], r['y'], 'x', color='tab:red', markersize=9, markeredgewidth=2,
                label='SLAM estimate' if first_est else None, zorder=6)
        ax.plot([r['x'], gx], [r['y'], gy], color='tab:red', linestyle='--', linewidth=1,
                label='estimate -> actual' if first_est else None)
        first_est = False
    errs = [r['err'] for r in rows]
    ax.set_title(f'{title}\nmean landmark error {100 * np.mean(errs):.1f} cm over {len(errs)} tags'
                 if errs else title, fontsize=10)
    ax.set_xlabel('x [m] (east)')
    ax.set_ylabel('y [m] (north)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if arena:
        ax.set_xlim(-0.6, arena + 0.6)
        ax.set_ylim(-0.6, arena + 0.6)


def plot_overlay(run, gt, out, arena, raw_rows, aligned_rows, transform_aligned):
    """map_overlay.png: actual (measured) map with the SLAM map overlaid, two alignments."""
    if not gt or not (raw_rows or aligned_rows):
        return None
    plt = plt_module()
    panels = []
    if raw_rows and run.start is not None:
        panels.append((raw_rows, lambda x, y: tuple(to_world(run.start, x, y)[0]),
                       'SLAM map placed with the measured start pose ({:.2f}, {:.2f}, {:.0f} deg)'
                       .format(run.start[0], run.start[1], math.degrees(run.start[2]))))
    if aligned_rows and transform_aligned is not None:
        theta, t = transform_aligned
        R = rot(theta)
        base = run.start if run.start is not None else (0.0, 0.0, 0.0)
        out_tags = [r['tag'] for r in aligned_rows if not r.get('fit_inlier', True)]
        panels.append((aligned_rows, lambda x, y: tuple(R @ to_world(base, x, y)[0] + t),
                       f'after best-fit rigid alignment (shape error; '
                       f'{math.degrees(theta):+.1f} deg'
                       + (f', fit ignores outlier tag {out_tags}' if out_tags else '') + ')'))
    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 8.5), squeeze=False)
    for ax, (rows, tf, title) in zip(axes[0], panels):
        _overlay_panel(ax, plt, run, gt, rows, tf, arena, title)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 7), fontsize=8,
               frameon=False)
    fig.suptitle(f'{run.name}: actual map (black / green) with the SLAM map overlaid (red)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return save(fig, out, 'map_overlay.png')


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def lap_boundaries(run):
    """Return (lap, time) at which each lap ended, from the lap snapshots."""
    return [(lap, t) for k, (lap, t, _) in run.snapshots.items() if k.startswith('lap')]


def plot_timeline(run, out):
    plt = plt_module()
    traj = run.trajectory
    t = np.array([p['t'] for p in traj])
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax = axes[0]
    ax.step(t, [p['n_landmarks'] for p in traj], where='post', color='tab:blue')
    for e in run.events:
        ax.annotate(f"{int(e['tag'])}", (e['t'], e['n_landmarks']), textcoords='offset points',
                    xytext=(2, 4), fontsize=7)
    ax.set_ylabel('landmarks N in state')
    ax.set_title(f'{run.name}: state grows only when a new tag is confirmed')
    ax = axes[1]
    ax.plot(t, [p['std_x'] for p in traj], label='std x')
    ax.plot(t, [p['std_y'] for p in traj], label='std y')
    ax.set_ylabel('robot position std [m]')
    ax2 = ax.twinx()
    ax2.plot(t, [math.degrees(p['std_theta']) for p in traj], color='tab:red', alpha=0.6,
             label='std theta')
    ax2.set_ylabel('robot heading std [deg]', color='tab:red')
    ax.legend(loc='upper left', fontsize='small')
    ax = axes[2]
    ax.plot(t, [p['map_std'] for p in traj], color='tab:purple')
    ax.set_ylabel('mean landmark std [m]')
    ax.set_title('map uncertainty (drops at loop closures, i.e. re-observations)')
    ax = axes[3]
    ax.bar(t, [p['sightings'] for p in traj], width=0.4, color='0.7', label='sightings')
    ax.bar(t, [p['updates'] for p in traj], width=0.4, color='tab:green',
           label='accepted re-observations')
    ax.set_ylabel('per log period')
    ax.set_xlabel('time [s]')
    ax.legend(loc='upper left', fontsize='small')
    for lap, tl in lap_boundaries(run):
        for a in axes:
            a.axvline(tl, color='black', linestyle=':', linewidth=1)
        axes[0].annotate(f'lap {lap} end', (tl, 0), textcoords='offset points', xytext=(2, 2),
                         fontsize=7)
    for a in axes:
        a.grid(True, alpha=0.3)
    return save(fig, out, 'timeline.png')


def plot_laps(lap_table, out, name):
    if not lap_table:
        return None
    plt = plt_module()
    labels = [r['label'] for r in lap_table]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, lab in (('raw_mean', 'mean error (measured start)'),
                     ('aligned_mean', 'mean error (best-fit aligned)')):
        vals = [r.get(key) for r in lap_table]
        if any(v is not None for v in vals):
            ax1.plot(x, [100 * v if v is not None else np.nan for v in vals], 'o-', label=lab)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel('landmark error [cm]')
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.3)
    ax2.plot(x, [1e4 * r['mean_area95'] for r in lap_table], 'o-', color='tab:red')
    ax2.set_xticks(x, labels)
    ax2.set_ylabel('mean 95% ellipse area [cm^2]')
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f'{name}: does the map improve with more laps?')
    return save(fig, out, 'laps.png')


def plot_convergence(run, gt, out):
    """Each landmark's error (best-fit aligned per snapshot) and std at every waypoint."""
    plt = plt_module()
    labels = [k for k in run.snapshots if k.startswith('wp') or k == 'final']
    if not labels:
        return None
    times = [run.snapshots[k][1] for k in labels]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    tags = sorted(run.final_map)
    for tag in tags:
        stds = [math.sqrt(np.trace(run.snapshots[k][2][tag][3]) / 2)
                if tag in run.snapshots[k][2] else np.nan for k in labels]
        ax2.plot(times, stds, '.-', label=f'tag {tag}')
    if gt:
        errs = {tag: [] for tag in tags}
        for k in labels:
            rows, _ = (map_errors(run.snapshots[k][2], gt, run.start)
                       if run.start is not None
                       else map_errors(run.snapshots[k][2], gt, align=True))
            by_tag = {r['tag']: r['err'] for r in rows}
            for tag in tags:
                errs[tag].append(by_tag.get(tag, np.nan))
        for tag in tags:
            ax1.plot(times, [100 * e for e in errs[tag]], '.-', label=f'tag {tag}')
        ax1.set_ylabel('error [cm] ' + ('(measured start)' if run.start is not None
                                        else '(aligned)'))
        ax1.legend(fontsize='x-small', ncol=4)
    ax2.set_ylabel('position std sqrt(tr/2) [m]')
    ax2.set_xlabel('time [s] (one point per waypoint)')
    for lap, tl in lap_boundaries(run):
        for a in (ax1, ax2):
            a.axvline(tl, color='black', linestyle=':', linewidth=1)
    for a in (ax1, ax2):
        a.grid(True, alpha=0.3)
    ax1.set_title(f'{run.name}: landmark convergence')
    return save(fig, out, 'convergence.png')


def plot_nis(run, out):
    upd = [m for m in run.measurements if m['kind'] in ('update', 'rejected')]
    if not upd:
        return None
    plt = plt_module()
    dof = int(run.info.get('params', {}).get('landmark_dof', 3))
    d2 = np.array([m['mahalanobis2'] for m in upd])
    ok = np.array([m['kind'] == 'update' for m in upd])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(d2[ok], bins=40, range=(0, 15), density=True, alpha=0.6, label='accepted')
    xs = np.linspace(0.05, 15, 200)
    ax1.plot(xs, [chi2_pdf(x, dof) for x in xs], 'k-', label=f'chi2({dof})')
    ax1.set_xlabel('innovation Mahalanobis^2 (NIS)')
    ax1.legend()
    ax1.set_title(f'mean NIS {d2[ok].mean():.2f} (consistent filter: {dof})')
    ax2.plot([m['t'] for m in upd], d2, '.', markersize=3)
    ax2.axhline(dof, color='green', label='expected mean')
    gate = 11.34 if dof == 3 else 9.21
    ax2.axhline(gate, color='red', label='gate')
    ax2.set_yscale('log')
    ax2.set_xlabel('time [s]')
    ax2.legend(fontsize='small')
    return save(fig, out, 'nis.png')


# --- Report -------------------------------------------------------------------------------------

def lap_table_for(run, gt):
    table = []
    for label in run.lap_labels + (['final'] if 'final' in run.snapshots else []):
        lap, t, m = run.snapshots[label]
        row = {'label': label, 't': t, 'n_landmarks': len(m),
               'mean_area95': float(np.mean([ellipse_area(v[3]) for v in m.values()]))
               if m else float('nan'),
               'mean_std': float(np.mean([math.sqrt(np.trace(v[3]) / 2) for v in m.values()]))
               if m else float('nan')}
        if gt:
            if run.start is not None:
                row['raw_mean'] = stats(map_errors(m, gt, run.start)[0])['mean']
            row['aligned_mean'] = stats(map_errors(m, gt, run.start, align=True)[0])['mean']
        table.append(row)
    return table


def cm(v):
    return f'{100 * v:.1f}' if v is not None and math.isfinite(v) else 'n/a'


def write_report(run, gt, out, raw_rows, aligned_rows, transform, lap_table, figures):
    lines = [f'# SLAM results: {run.name}', '']
    info, summ = run.info, run.info.get('summary', {})
    traj = run.trajectory
    lines += [
        f'- Path: **{info.get("path", "?")}**, {info.get("waypoints_per_lap", "?")} waypoints '
        f'per lap, {len(run.lap_labels)} laps completed '
        f'(all waypoints: {summ.get("completed_all_waypoints")}); simulated: '
        f'{info.get("simulated")}',
        f'- Duration {summ.get("duration_s", traj[-1]["t"] if traj else 0):.0f} s, '
        f'distance driven (filter estimate) {path_length(traj):.2f} m',
        f'- Landmarks discovered: **{len(run.final_map)}** {sorted(run.final_map)}'
        + (f'; ground truth has {len(gt)}' if gt else ''),
        f'- Sightings: {summ.get("counts", {})}',
        f'- Mean NIS of accepted updates: {summ.get("mean_nis") or float("nan"):.2f} '
        f'(expected {summ.get("nis_dof", 3)} for a consistent filter)',
        f'- landmark_dof = {info.get("params", {}).get("landmark_dof")}, '
        f'use_imu = {info.get("params", {}).get("use_imu")}',
        f'- Start pose (world): {run.start if run.start is not None else "NOT MEASURED"}',
        '',
    ]
    if traj:
        end = traj[-1]
        lines += [f'- Final pose estimate (start frame): ({end["x"]:.3f}, {end["y"]:.3f}, '
                  f'{math.degrees(end["theta"]):.1f} deg), std ({end["std_x"]:.3f}, '
                  f'{end["std_y"]:.3f} m, {math.degrees(end["std_theta"]):.1f} deg); dead '
                  f'reckoning ended at ({end["odom_x"]:.3f}, {end["odom_y"]:.3f}, '
                  f'{math.degrees(end["odom_theta"]):.1f} deg). The path returns to the start, '
                  f'so the true final pose is (0, 0, 0) up to controller tolerance.', '']
    if run.truth and traj:
        tru = {round(p['t'], 1): p for p in run.truth}
        errs = [math.hypot(p['x'] - tru[round(p['t'], 1)]['x'],
                           p['y'] - tru[round(p['t'], 1)]['y'])
                for p in traj if round(p['t'], 1) in tru]
        odom_errs = [math.hypot(p['odom_x'] - tru[round(p['t'], 1)]['x'],
                                p['odom_y'] - tru[round(p['t'], 1)]['y'])
                     for p in traj if round(p['t'], 1) in tru]
        lines += [f'- (Simulation) trajectory error: SLAM mean {cm(np.mean(errs))} cm, max '
                  f'{cm(np.max(errs))} cm; dead reckoning mean {cm(np.mean(odom_errs))} cm, '
                  f'max {cm(np.max(odom_errs))} cm', '']

    if raw_rows:
        lines += ['## Actual vs estimated landmark positions (world frame, measured start pose)',
                  '', '| tag | actual x [m] | actual y [m] | actual yaw [deg] | SLAM x [m] | '
                  'SLAM y [m] | SLAM yaw [deg] | dx [cm] | dy [cm] | error [cm] |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for r in raw_rows:
            yaw = f'{math.degrees(r["psi"]):.1f}' if math.isfinite(r['psi']) else 'n/a'
            lines.append(f'| {r["tag"]} | {r["gt_x"]:.3f} | {r["gt_y"]:.3f} | '
                         f'{math.degrees(r["gt_yaw"]):.1f} | {r["x"]:.3f} | {r["y"]:.3f} | '
                         f'{yaw} | '
                         f'{100 * r["err_x"]:+.1f} | {100 * r["err_y"]:+.1f} | {cm(r["err"])} |')
        lines.append('')
    for title, rows in (('measured start pose', raw_rows), ('best-fit rigid alignment',
                                                            aligned_rows)):
        if not rows:
            continue
        s = stats(rows)
        lines += [f'## Landmark errors, {title}', '',
                  f'Mean error **{cm(s["mean"])} cm**, RMSE {cm(s["rmse"])} cm, median '
                  f'{cm(s["median"])} cm, max {cm(s["max"])} cm over {s["n"]} landmarks; mean '
                  f'NEES {s["mean_nees"]:.2f} (2 expected if the ellipses are honest).', '',
                  '| tag | est x | est y | GT x | GT y | error [cm] | err psi [deg] | std x [cm] '
                  '| std y [cm] | 95% area [cm^2] | updates |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            epsi = f'{r["err_psi_deg"]:+.1f}' if r['err_psi_deg'] is not None else 'n/a'
            lines.append(f'| {r["tag"]} | {r["x"]:.3f} | {r["y"]:.3f} | {r["gt_x"]:.3f} | '
                         f'{r["gt_y"]:.3f} | {cm(r["err"])} | {epsi} | {cm(r["std_x"])} | '
                         f'{cm(r["std_y"])} | {1e4 * r["area95"]:.1f} | {int(r["sightings"])} |')
        lines.append('')
    if transform is not None:
        lines += [f'Best-fit alignment correction on top of the start pose: rotation '
                  f'{math.degrees(transform[0]):+.2f} deg, translation '
                  f'({cm(transform[1][0])}, {cm(transform[1][1])}) cm. A large value means the '
                  f'start pose measurement (not the map shape) dominates the raw error.', '']
        outliers = [r['tag'] for r in aligned_rows if not r.get('fit_inlier', True)]
        if outliers:
            inl = [r for r in aligned_rows if r.get('fit_inlier', True)]
            raw_in = [r for r in raw_rows if r['tag'] not in outliers]
            lines += [f'Tags ignored by the alignment fit as outliers: {outliers}. Without them: '
                      f'mean error {cm(stats(raw_in)["mean"]) if raw_in else "-"} cm via the '
                      f'start pose, {cm(stats(inl)["mean"])} cm after alignment (map shape) over '
                      f'{len(inl)} tags.', '']
    if gt and len(set(run.final_map) & set(gt)) >= 2:
        common = sorted(set(run.final_map) & set(gt))
        base = run.start or (0.0, 0.0, 0.0)
        a = [to_world(base, *run.final_map[k][:2])[0] for k in common]
        _, _, s = best_fit(a, [gt[k][:2] for k in common], scale=True)
        lines += [f'Best-fit similarity scale {s:.3f} (1.0 = correct; a map uniformly too large '
                  f'or small suggests a wrong tag size or camera intrinsics).', '']
    if gt:
        missed = sorted(set(gt) - set(run.final_map))
        extra = sorted(set(run.final_map) - set(gt))
        lines += [f'Ground-truth tags never mapped: {missed or "none"}. '
                  f'Mapped tags not in the ground truth (spurious?): {extra or "none"}.', '']

    if lap_table:
        lines += ['## Map after each lap', '',
                  '| snapshot | t [s] | N | mean error, measured start [cm] | mean error, aligned '
                  '[cm] | mean std [cm] | mean 95% area [cm^2] |', '|---|---|---|---|---|---|---|']
        for r in lap_table:
            lines.append(f'| {r["label"]} | {r["t"]:.0f} | {r["n_landmarks"]} | '
                         f'{cm(r.get("raw_mean"))} | {cm(r.get("aligned_mean"))} | '
                         f'{cm(r["mean_std"])} | {1e4 * r["mean_area95"]:.1f} |')
        lines.append('')

    lines += ['## Landmark discovery', '', '| t [s] | tag | N after | robot std theta [deg] '
              '| initial std x, y [cm] |', '|---|---|---|---|---|']
    for e in run.events:
        lines.append(f'| {e["t"]:.1f} | {int(e["tag"])} | {int(e["n_landmarks"])} | '
                     f'{math.degrees(e["robot_std_theta"]):.1f} | {cm(e["std_x"])}, '
                     f'{cm(e["std_y"])} |')
    lines += ['', '## Sightings per tag', '',
              '| tag | total | update | rejected | candidate | new | incidence | mean NIS |',
              '|---|---|---|---|---|---|---|---|']
    for tag in sorted({int(m['tag']) for m in run.measurements}):
        ms = [m for m in run.measurements if int(m['tag']) == tag]
        k = {kind: sum(m['kind'] == kind for m in ms)
             for kind in ('update', 'rejected', 'candidate', 'new', 'incidence')}
        nis = [m['mahalanobis2'] for m in ms if m['kind'] == 'update']
        lines.append(f'| {tag} | {len(ms)} | {k["update"]} | {k["rejected"]} | '
                     f'{k["candidate"]} | {k["new"]} | {k["incidence"]} | '
                     f'{np.mean(nis) if nis else float("nan"):.2f} |')
    lines += ['', '## Figures', ''] + [f'![{os.path.basename(f)}]({os.path.basename(f)})'
                                       for f in figures if f]
    path = os.path.join(out, 'results.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def sensor_noise(run):
    """Per-tag spread of (range, bearing, orientation) while standing still: fit R (plan 2.7)."""
    print('Robot must have been standing still for this. Per tag:')
    print('  tag   n   mean r [m]  std r [m]  std bearing [rad]  std orient [rad]')
    pts = []
    for tag in sorted({int(m['tag']) for m in run.measurements}):
        ms = [m for m in run.measurements if int(m['tag']) == tag]
        if len(ms) < 10:
            continue
        r = np.array([m['range'] for m in ms])
        b = np.unwrap([m['bearing'] for m in ms])
        o = np.unwrap([m['orientation'] for m in ms])
        print(f'  {tag:<4} {len(ms):>4}  {r.mean():9.3f}  {r.std():9.4f}  {b.std():16.4f}  '
              f'{o.std():15.4f}')
        pts.append((r.mean(), r.std(), b.std(), o.std()))
    if not pts:
        print('Not enough sightings (need >= 10 per tag).')
        return
    pts = np.array(pts)
    if len(pts) >= 2 and np.ptp(pts[:, 0]) > 0.3:
        b1, b0 = np.polyfit(pts[:, 0], pts[:, 1], 1)
        print(f'Fit std(r) = {b0:.4f} + {b1:.4f} r')
    else:
        b0, b1 = pts[:, 1].mean(), 0.0
        print('Need tags at several ranges for the range slope; using a constant.')
    print('Frame-to-frame spread misses bias: inflate 2-3x. Suggested hw3.yaml values (x2.5):')
    print(f'  range_std: {2.5 * max(b0, 0.005):.3f}\n  range_std_per_m: {2.5 * max(b1, 0):.3f}')
    print(f'  bearing_std: {2.5 * pts[:, 2].mean():.3f}\n'
          f'  orientation_std: {2.5 * pts[:, 3].mean():.3f}')


def evaluate(run, gt, out, arena):
    os.makedirs(out, exist_ok=True)
    raw_rows = map_errors(run.final_map, gt, run.start)[0] if gt and run.start else []
    aligned_rows, transform = (map_errors(run.final_map, gt, run.start, align=True)
                               if gt else ([], None))
    lap_table = lap_table_for(run, gt)
    figures = plot_maps(run, gt, out, arena, raw_rows, aligned_rows, transform)
    figures.append(plot_overlay(run, gt, out, arena, raw_rows, aligned_rows, transform))
    if run.trajectory:
        figures.append(plot_timeline(run, out))
    figures.append(plot_laps(lap_table, out, run.name))
    figures.append(plot_convergence(run, gt, out))
    figures.append(plot_nis(run, out))
    report = write_report(run, gt, out, raw_rows, aligned_rows, transform, lap_table, figures)
    summary = {
        'run': run.dir, 'path': run.info.get('path'), 'start_pose': run.start,
        'n_landmarks': len(run.final_map), 'landmark_ids': sorted(run.final_map),
        'raw': stats(raw_rows), 'aligned': stats(aligned_rows),
        'per_landmark_raw': [{k: v for k, v in r.items() if k != 'cov'} for r in raw_rows],
        'per_landmark_aligned': [{k: v for k, v in r.items() if k != 'cov'}
                                 for r in aligned_rows],
        'laps': lap_table, 'run_summary': run.info.get('summary', {}),
        'alignment': ({'theta_deg': math.degrees(transform[0]), 'tx': float(transform[1][0]),
                       'ty': float(transform[1][1])} if transform else None),
        'distance_m': path_length(run.trajectory),
    }
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'\n{run.name}: {len(run.final_map)} landmarks {sorted(run.final_map)}')
    if raw_rows:
        print(f'  mean landmark error (measured start): {cm(stats(raw_rows)["mean"])} cm')
    if aligned_rows:
        print(f'  mean landmark error (aligned):        {cm(stats(aligned_rows)["mean"])} cm')
    if gt and run.start is None:
        print(f'  start pose not measured: fill in {os.path.join(run.dir, "start_pose.txt")} '
              f'or pass --start for the raw error')
    for r in lap_table:
        print(f'  {r["label"]:>6}: N={r["n_landmarks"]} raw {cm(r.get("raw_mean"))} cm, '
              f'aligned {cm(r.get("aligned_mean"))} cm, mean std {cm(r["mean_std"])} cm')
    print(f'  wrote {report} and {len([f for f in figures if f])} figures in {out}')
    return summary


def write_comparison(summaries, out):
    os.makedirs(out, exist_ok=True)
    lines = ['# SLAM run comparison', '',
             '| run | path | N | mean err raw [cm] | RMSE raw [cm] | mean err aligned [cm] | '
             'mean NIS | distance [m] | laps: aligned mean error [cm] |',
             '|---|---|---|---|---|---|---|---|---|']
    for s in summaries:
        laps = ', '.join(cm(r.get('aligned_mean')) for r in s['laps'])
        lines.append(f'| {os.path.basename(os.path.normpath(s["run"]))} | {s["path"]} | '
                     f'{s["n_landmarks"]} | {cm(s["raw"]["mean"])} | {cm(s["raw"]["rmse"])} | '
                     f'{cm(s["aligned"]["mean"])} | '
                     f'{s["run_summary"].get("mean_nis") or float("nan"):.2f} | '
                     f'{s["distance_m"]:.1f} | {laps} |')
    path = os.path.join(out, 'comparison.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWrote {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='*',
                        help='run folders or names in <workspace>/hw3_logs (default: newest)')
    parser.add_argument('--gt', help='ground-truth landmarks file (world frame)')
    parser.add_argument('--start', type=float, nargs=3, metavar=('X0', 'Y0', 'THETA0'),
                        help='world start pose (overrides RUN/start_pose.txt; one run only)')
    parser.add_argument('--arena', type=float, default=3.0, help='arena side [m], 0 to hide')
    parser.add_argument('--out', help='output folder (one run: RUN/eval; several: comparison)')
    parser.add_argument('--ellipse-scale', type=float, default=1.0,
                        help='magnify the drawn 95%% ellipses (e.g. 5 when they are tiny)')
    parser.add_argument('--sensor-noise', action='store_true',
                        help='fit measurement noise from a run with the robot standing still')
    args = parser.parse_args()
    global ELLIPSE_SCALE
    ELLIPSE_SCALE = args.ellipse_scale

    from robot_control import hw3_paths
    runs = [hw3_paths.resolve_run(r) for r in (args.runs or ['latest'])]
    gt = load_landmarks(args.gt) if args.gt else {}
    summaries = []
    for run_dir in runs:
        run = Run(run_dir)
        if args.sensor_noise:
            sensor_noise(run)
            continue
        if args.start:
            run.start = tuple(args.start)
        out = args.out if (args.out and len(runs) == 1) else os.path.join(run_dir, 'eval')
        summaries.append(evaluate(run, gt, out, args.arena))
    if len(summaries) > 1:
        write_comparison(summaries, args.out or os.path.join(os.path.dirname(
            os.path.normpath(runs[0])), 'comparison'))


if __name__ == '__main__':
    main()
