"""
Pose plots of a velocity_control run: every estimated pose drawn as an arrow (position + heading).

  ros2 run robot_control pose_plots [RUN_DIR]     (default: the newest ~/hw2_logs/run_*)

Writes into the run folder, each as .png and as zoomable .svg:
  final_1_corrected     corrected state (EKF), a pose every 0.5 s
  final_2_landmark      the pose each tag sighting implies by itself
  final_3_motion_model  motion model applied to the previous corrected pose, a pose every
                        0.5 s, the corrections that followed, and their size over time
  final_4_overlay       all of them together, with uncorrected dead reckoning
and poses.html: the same poses, interactive; hover any arrow for its time, x, y and heading.
"""

import argparse
import glob
import json
import math
import os

import numpy as np

from robot_control.localization import load_landmarks
from robot_control.visualize_run import read_csv
from robot_control.waypoint_controller import load_waypoints

# Reference categorical palette, fixed slot order; context marks stay in neutral ink
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID, SURFACE = '#1f1f1e', '#6b6a63', '#e4e3df', '#fcfcfb'
ARROW = 0.12  # m, length of a pose arrow
PERIOD = 0.5  # s between drawn poses of the 20 Hz estimates


def load_run(run_dir):
    ticks = read_csv(os.path.join(run_dir, 'ticks.csv'))
    if not ticks:
        raise SystemExit(f'no ticks.csv rows in {run_dir}')
    sightings = [r for r in read_csv(os.path.join(run_dir, 'measurements.csv'))
                 if r.get('status') != 'out_of_range']

    def col(rows, *keys):
        return np.array([[r[k] for k in keys] for r in rows], dtype=float).reshape(-1, len(keys))
    t = col(ticks, 't')[:, 0]
    return {
        't': t,
        'corr': col(ticks, 'corr_x', 'corr_y', 'corr_theta'),
        'model': col(ticks, 'model_x', 'model_y', 'model_theta'),
        'odom': col(ticks, 'odom_x', 'odom_y', 'odom_theta'),
        'every': np.r_[True, np.diff(np.floor((t - t[0]) / PERIOD)) > 0],
        'vis': col(sightings, 'vision_x', 'vision_y', 'vision_theta'),
        'vis_t': col(sightings, 't')[:, 0],
        'vis_tag': [int(r['tag']) for r in sightings],
        'accepted': np.array([r['status'] in ('accepted', 'forced') for r in sightings],
                             dtype=bool),
        'arrivals': read_csv(os.path.join(run_dir, 'waypoints.csv')),
        'landmarks': load_landmarks(os.path.join(run_dir, 'landmarks.txt')),
        'waypoints': load_waypoints(os.path.join(run_dir, 'waypoints.txt')),
    }


def style(plt):
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
        'ytick.color': MUTED, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 10, 'legend.frameon': False})


def draw_map(ax, run, xlim=(-1.0, 2.8)):
    """Landmarks (bar + facing arrow) and waypoints with their headings."""
    wp = np.array(run['waypoints'])
    ax.plot(wp[:, 0], wp[:, 1], color=MUTED, linestyle=':', linewidth=1, zorder=1)
    for x, y, th in run['waypoints']:
        ax.annotate('', xy=(x + 0.22 * math.cos(th), y + 0.22 * math.sin(th)), xytext=(x, y),
                    arrowprops={'arrowstyle': '->', 'color': INK, 'lw': 1.2}, zorder=4)
        ax.scatter([x], [y], s=150, facecolors='none', edgecolors=INK, linewidths=1.4, zorder=4)
    closed = len(wp) > 1 and np.allclose(wp[0, :2], wp[-1, :2])
    for i, (x, y, _) in enumerate(run['waypoints'][:-1] if closed else run['waypoints']):
        label = f'W0 / W{len(wp) - 1}' if (i == 0 and closed) else f'W{i}'
        ax.annotate(label, (x, y), textcoords='offset points', xytext=(10, -16), fontsize=9,
                    fontweight='bold')
    for tag, (x, y, yaw) in run['landmarks'].items():
        h = 0.09
        ax.plot([x - h * math.sin(yaw), x + h * math.sin(yaw)],
                [y + h * math.cos(yaw), y - h * math.cos(yaw)], color=INK, linewidth=5,
                solid_capstyle='butt', zorder=3)
        ax.annotate('', xy=(x + 0.2 * math.cos(yaw), y + 0.2 * math.sin(yaw)), xytext=(x, y),
                    arrowprops={'arrowstyle': '->', 'color': MUTED, 'lw': 1}, zorder=3)
        ax.annotate(f'tag {tag}', (x, y), textcoords='offset points', xytext=(6, 6),
                    fontsize=8, color=MUTED)
    ax.plot([], [], color=INK, linewidth=5, label='landmark (arrow: facing)')
    ax.scatter([], [], s=150, facecolors='none', edgecolors=INK, label='waypoint')
    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.8, 3.1)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')


def poses(ax, p, color, label, zorder=6):
    """One arrow per pose: tail at (x, y), pointing along theta."""
    if not len(p):
        return
    ax.quiver(p[:, 0], p[:, 1], np.cos(p[:, 2]), np.sin(p[:, 2]), color=color, angles='xy',
              scale_units='xy', scale=1.0 / ARROW, width=0.0035, headwidth=4, headlength=4.5,
              headaxislength=4, zorder=zorder, label=label)
    ax.scatter(p[:, 0], p[:, 1], s=6, color=color, zorder=zorder)


def time_labels(ax, t, p, step=5.0):
    """Label every step seconds where the robot has moved on, to find a time on the map."""
    last = None
    for mark in np.arange(step, t[-1], step):
        i = int(np.argmin(np.abs(t - mark)))
        if last is not None and np.hypot(*(p[i, :2] - last)) < 0.15:
            continue  # stopped or turning in place: the label would sit on the previous one
        last = p[i, :2]
        ax.annotate(f'{mark:.0f}s', p[i, :2], textcoords='offset points', xytext=(-14, 6),
                    fontsize=7, color=MUTED, zorder=8)


def save(fig, run_dir, name, written):
    for ext in ('png', 'svg'):
        path = os.path.join(run_dir, f'{name}.{ext}')
        fig.savefig(path, dpi=170)
        written.append(path)


def pose_figures(run_dir, run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)
    title = os.path.basename(os.path.normpath(run_dir))
    t, corr, model, odom, every = run['t'], run['corr'], run['model'], run['odom'], run['every']
    vis, accepted = run['vis'], run['accepted']
    written = []

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_map(ax, run)
    ax.plot(corr[:, 0], corr[:, 1], color=BLUE, linewidth=1, alpha=0.5, zorder=5)
    poses(ax, corr[every], BLUE, f'corrected pose, every {PERIOD:g} s')
    time_labels(ax, t, corr)
    for r in run['arrivals']:
        ax.scatter([r['x']], [r['y']], marker='x', s=70, color=INK, zorder=9)
    ax.scatter([], [], marker='x', s=70, color=INK, label='estimate at waypoint arrival')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f'Corrected state (EKF): pose every {PERIOD:g} s\n{title}', loc='left')
    fig.tight_layout()
    save(fig, run_dir, 'final_1_corrected', written)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9))
    draw_map(ax, run)
    ax.plot(corr[:, 0], corr[:, 1], color=GRID, linewidth=2.5, zorder=2,
            label='corrected path (context)')
    poses(ax, vis[accepted], ORANGE, f'landmark-only pose, accepted ({int(accepted.sum())})')
    poses(ax, vis[~accepted], MUTED, f'landmark-only pose, rejected ({int((~accepted).sum())})')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f'Pose from each landmark sighting alone\n{title}', loc='left')
    fig.tight_layout()
    save(fig, run_dir, 'final_2_landmark', written)
    plt.close(fig)

    shift = corr[1:, :2] - model[:-1, :2]
    size = np.hypot(shift[:, 0], shift[:, 1])
    turn = np.degrees(np.arctan2(np.sin(corr[1:, 2] - model[:-1, 2]),
                                 np.cos(corr[1:, 2] - model[:-1, 2])))
    fig = plt.figure(figsize=(15, 8.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
    ax = fig.add_subplot(grid[:, 0])
    draw_map(ax, run)
    ax.plot(model[:, 0], model[:, 1], color=AQUA, linewidth=1, alpha=0.5, zorder=5)
    poses(ax, model[every], AQUA, f'motion-model pose from the previous corrected pose, '
                                  f'every {PERIOD:g} s')
    big = np.nonzero(size > 0.005)[0]
    for k in big:
        ax.annotate('', xy=corr[k + 1, :2], xytext=model[k, :2],
                    arrowprops={'arrowstyle': '->', 'color': INK, 'lw': 1}, zorder=7)
    ax.plot([], [], color=INK, linewidth=1, label=f'correction by a sighting (> 5 mm, {len(big)})')
    time_labels(ax, t, model)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title(f'Motion-model prediction: pose every {PERIOD:g} s\n{title}', loc='left')
    a1 = fig.add_subplot(grid[0, 1])
    a1.vlines(t[1:], 0, size * 100, color=AQUA, linewidth=1.2)
    a1.set_ylabel('position correction [cm]')
    a1.set_title('Correction applied after each prediction', loc='left')
    a2 = fig.add_subplot(grid[1, 1], sharex=a1)
    a2.vlines(t[1:], 0, turn, color=AQUA, linewidth=1.2)
    a2.axhline(0, color=MUTED, linewidth=0.8)
    a2.set_ylabel('heading correction [deg]')
    a2.set_xlabel('time [s]')
    for a in (a1, a2):
        for r in run['arrivals']:
            a.axvline(r['t'], color=MUTED, linestyle=':', linewidth=1)
            a.annotate(f"W{int(r['index'])}", (r['t'], 1.0), xycoords=('data', 'axes fraction'),
                       fontsize=8, color=MUTED, va='top')
    fig.tight_layout()
    save(fig, run_dir, 'final_3_motion_model', written)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9))
    draw_map(ax, run, xlim=(-1.0, max(2.8, float(np.nanmax(odom[:, 0])) + 0.4)))
    ax.plot(odom[:, 0], odom[:, 1], color=YELLOW, linewidth=2, linestyle='--', zorder=4,
            label='dead reckoning only (never corrected)')
    poses(ax, model[every], AQUA, 'motion-model pose', zorder=5)
    poses(ax, vis[accepted], ORANGE, 'landmark-only pose (accepted)', zorder=6)
    poses(ax, corr[every], BLUE, 'corrected pose (EKF)', zorder=7)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f'All estimates together\n{title}', loc='left')
    fig.tight_layout()
    save(fig, run_dir, 'final_4_overlay', written)
    plt.close(fig)
    return written


def pose_html(run_dir, run):
    """Write poses.html: pose arrows per layer, hover for time, x, y and heading."""
    def rows(t, p, extra=None):
        return [[round(float(tt), 2), round(float(x), 4), round(float(y), 4),
                 round(math.degrees(float(th)), 1)] + ([extra[i]] if extra else [])
                for i, (tt, (x, y, th)) in enumerate(zip(t, p))]
    t, every = run['t'], run['every']
    acc = run['accepted']
    tags = run['vis_tag']
    data = {
        'title': os.path.basename(os.path.normpath(run_dir)),
        'layers': [
            {'name': 'Corrected state (EKF)', 'color': 'series-1',
             'poses': rows(t[every], run['corr'][every])},
            {'name': 'Landmark-only pose (accepted)', 'color': 'series-2',
             'poses': rows(run['vis_t'][acc], run['vis'][acc],
                           [f'tag {tags[i]}' for i in np.nonzero(acc)[0]])},
            {'name': 'Landmark-only pose (rejected)', 'color': 'muted',
             'poses': rows(run['vis_t'][~acc], run['vis'][~acc],
                           [f'tag {tags[i]}, rejected' for i in np.nonzero(~acc)[0]])},
            {'name': 'Motion-model prediction', 'color': 'series-3',
             'poses': rows(t[every], run['model'][every])},
            {'name': 'Dead reckoning only', 'color': 'series-4',
             'poses': rows(t[every], run['odom'][every])},
        ],
        'landmarks': [[tag, x, y, yaw] for tag, (x, y, yaw) in run['landmarks'].items()],
        'waypoints': [list(w) for w in run['waypoints']],
    }
    path = os.path.join(run_dir, 'poses.html')
    with open(path, 'w') as f:
        f.write(HTML.replace('__DATA__', json.dumps(data)))
    return path


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Poses</title>
<style>
:root { --surface-1:#fcfcfb; --text-primary:#1f1f1e; --text-secondary:#6b6a63;
  --grid:#e4e3df; --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --series-4:#eda100; --muted:#8b8a83; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --surface-1:#1a1a19; --text-primary:#f0efec; --text-secondary:#c3c2b7; --grid:#33332f;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --series-4:#c98500;
  --muted:#8b8a83; } }
:root[data-theme="dark"] { --surface-1:#1a1a19; --text-primary:#f0efec;
  --text-secondary:#c3c2b7; --grid:#33332f; --series-1:#3987e5; --series-2:#d95926;
  --series-3:#199e70; --series-4:#c98500; --muted:#8b8a83; }
body { margin:0; padding:16px; background:var(--surface-1); color:var(--text-primary);
  font:14px system-ui, sans-serif; }
h1 { font-size:18px; margin:0 0 4px; }
p { margin:0 0 12px; color:var(--text-secondary); }
#layers { display:flex; flex-wrap:wrap; gap:6px 18px; margin-bottom:10px; }
#layers label { display:flex; align-items:center; gap:6px; cursor:pointer; }
.swatch { width:14px; height:3px; border-radius:2px; display:inline-block; }
svg { width:100%; max-width:900px; height:auto; display:block; touch-action:none; }
#tip { position:fixed; pointer-events:none; background:var(--surface-1);
  border:1px solid var(--grid); border-radius:6px; padding:6px 8px; font-size:12px;
  box-shadow:0 2px 8px rgba(0,0,0,.15); display:none; white-space:nowrap; }
#tip b { display:block; margin-bottom:2px; }
</style></head><body>
<h1 id="title"></h1>
<p>Each arrow is one estimated pose (tail = position, arrow = heading).
Hover an arrow for its time, x, y and heading; toggle layers below.</p>
<div id="layers"></div>
<svg id="map" role="img" aria-label="Estimated poses on the map"></svg>
<div id="tip"></div>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue('--' + n).trim();
const svg = document.getElementById('map'), tip = document.getElementById('tip');
const NS = 'http://www.w3.org/2000/svg', W = 900, H = 820, M = 44, L = 0.12;
const pts = D.layers.flatMap(l => l.poses).concat(D.landmarks.map(l => [0, l[1], l[2]]));
const xs = pts.map(p => p[1]), ys = pts.map(p => p[2]);
const x0 = Math.min(...xs) - 0.3, x1 = Math.max(...xs) + 0.3;
const y0 = Math.min(...ys) - 0.3, y1 = Math.max(...ys) + 0.3;
const s = Math.min((W - 2 * M) / (x1 - x0), (H - 2 * M) / (y1 - y0));
const X = x => M + (x - x0) * s, Y = y => H - M - (y - y0) * s;
svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
const el = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
document.getElementById('title').textContent = 'Estimated poses, ' + D.title;
function draw() {
  svg.innerHTML = '';
  for (let g = Math.ceil(x0 * 2) / 2; g <= x1; g += 0.5) {
    el('line', {x1: X(g), x2: X(g), y1: M / 2, y2: H - M, stroke: css('grid')});
    el('text', {x: X(g), y: H - M + 16, 'text-anchor': 'middle', 'font-size': 11,
      fill: css('text-secondary')}).textContent = g.toFixed(1);
  }
  for (let g = Math.ceil(y0 * 2) / 2; g <= y1; g += 0.5) {
    el('line', {x1: M, x2: W - M / 2, y1: Y(g), y2: Y(g), stroke: css('grid')});
    el('text', {x: M - 6, y: Y(g) + 4, 'text-anchor': 'end', 'font-size': 11,
      fill: css('text-secondary')}).textContent = g.toFixed(1);
  }
  el('text', {x: W - M, y: H - 8, 'text-anchor': 'end', 'font-size': 11,
    fill: css('text-secondary')}).textContent = 'x [m]';
  el('text', {x: 8, y: M / 2 - 4, 'font-size': 11, fill: css('text-secondary')})
    .textContent = 'y [m]';
  el('polyline', {points: D.waypoints.map(w => `${X(w[0])},${Y(w[1])}`).join(' '),
    fill: 'none', stroke: css('text-secondary'), 'stroke-dasharray': '2 4'});
  const n = D.waypoints.length, f = D.waypoints[0], z = D.waypoints[n - 1];
  const closed = n > 1 && f[0] === z[0] && f[1] === z[1];
  D.waypoints.forEach((w, i) => {
    el('circle', {cx: X(w[0]), cy: Y(w[1]), r: 9, fill: 'none', stroke: css('text-primary'),
      'stroke-width': 1.5});
    if (closed && i === n - 1) return;
    el('text', {x: X(w[0]) + 12, y: Y(w[1]) + 18, 'font-size': 12, 'font-weight': 600,
      fill: css('text-primary')}).textContent = (closed && i === 0) ? `W0 / W${n - 1}` : 'W' + i;
  });
  D.landmarks.forEach(([id, x, y, yaw]) => {
    const h = 0.09;
    el('line', {x1: X(x - h * Math.sin(yaw)), y1: Y(y + h * Math.cos(yaw)),
      x2: X(x + h * Math.sin(yaw)), y2: Y(y - h * Math.cos(yaw)),
      stroke: css('text-primary'), 'stroke-width': 6});
    el('text', {x: X(x) + 8, y: Y(y) - 8, 'font-size': 11, fill: css('text-secondary')})
      .textContent = 'tag ' + id;
  });
  D.layers.forEach(l => { if (l.hidden) return; const c = css(l.color);
    l.poses.forEach(p => { const [t, x, y, deg] = p, a = deg * Math.PI / 180;
      const ex = X(x + L * Math.cos(a)), ey = Y(y + L * Math.sin(a));
      const hx = Math.cos(a), hy = -Math.sin(a), k = 6;
      el('line', {x1: X(x), y1: Y(y), x2: ex, y2: ey, stroke: c, 'stroke-width': 2,
        'stroke-linecap': 'round'});
      el('path', {d: `M${ex},${ey} L${ex - k * hx - k * 0.5 * hy},${ey - k * hy + k * 0.5 * hx}`
        + ` L${ex - k * hx + k * 0.5 * hy},${ey - k * hy - k * 0.5 * hx} Z`, fill: c});
      el('circle', {cx: X(x), cy: Y(y), r: 2.5, fill: c});
    }); });
}
const box = document.getElementById('layers');
D.layers.forEach(l => { const lab = document.createElement('label');
  const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = true;
  cb.onchange = () => { l.hidden = !cb.checked; draw(); };
  const sw = document.createElement('span'); sw.className = 'swatch';
  sw.style.background = `var(--${l.color})`;
  lab.append(cb, sw, `${l.name} (${l.poses.length})`); box.appendChild(lab); });
svg.addEventListener('pointermove', e => {
  const r = svg.getBoundingClientRect(), k = W / r.width;
  const mx = (e.clientX - r.left) * k, my = (e.clientY - r.top) * k;
  let best = null, bd = 14 * k;
  D.layers.forEach(l => { if (l.hidden) return; l.poses.forEach(p => {
    const d = Math.hypot(X(p[1]) - mx, Y(p[2]) - my);
    if (d < bd) { bd = d; best = [l, p]; } }); });
  if (!best) { tip.style.display = 'none'; return; }
  const [l, p] = best;
  tip.innerHTML = `<b>${l.name}</b>t = ${p[0].toFixed(2)} s<br>x = ${p[1].toFixed(3)} m, `
    + `y = ${p[2].toFixed(3)} m<br>heading = ${p[3].toFixed(1)}&deg;`
    + (p[4] ? `<br>${p[4]}` : '');
  tip.style.display = 'block'; tip.style.left = (e.clientX + 14) + 'px';
  tip.style.top = (e.clientY + 14) + 'px';
});
svg.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
draw();
</script></body></html>
"""


def make_pose_plots(run_dir):
    run = load_run(run_dir)
    written = pose_figures(run_dir, run) + [pose_html(run_dir, run)]
    for path in written:
        print(f'Wrote {path}')
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', nargs='?', help='a run_* folder (default: the newest)')
    args = parser.parse_args()
    make_pose_plots(args.run_dir or max(glob.glob(os.path.expanduser('~/hw2_logs/run_*'))))


if __name__ == '__main__':
    main()
