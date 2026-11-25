"""
HW4 command line: check the world file, plan, simulate, evaluate runs, summarise.

  ros2 run robot_control hw4 check                  # validate config/hw4_world.yaml, tags in view
  ros2 run robot_control hw4 plan                   # fast + safe + weighted plans and figures
  ros2 run robot_control hw4 plan --tradeoff        # also the alpha sweep (safety vs length)
  ros2 run robot_control hw4 sim fast --seeds 3     # offline rehearsal (controller + EKF + noise)
  ros2 run robot_control hw4 eval latest --end 0.47 2.52   # tape-measured final x y [theta]
  ros2 run robot_control hw4 summary                # comparison table of all evaluated runs

Plans and figures go to <workspace>/hw4_plans, runs to <workspace>/hw4_logs (see hw4_plan.py).
Every command takes --world (default: config/hw4_world.yaml in the source tree, so edits take
effect without a rebuild).
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

from robot_control import hw4_plan
from robot_control.hw4_world import load_world
from robot_control.path_tools import clearance_profile, point_segment_distance
from robot_control.waypoint_controller import wrap_angle

# Reference categorical palette (light mode), slots 1-3: validated all-pairs for <= 3 series.
COLORS = {'fast': '#2a78d6', 'safe': '#eb6834', 'weighted': '#1baf7a'}
STYLES = {'fast': '-', 'safe': '-', 'weighted': (0, (4, 2))}
MARKERS = {'fast': 'o', 'safe': 's', 'weighted': '^'}
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#8a8984'
GRID = '#e5e4e0'
SURFACE = '#fcfcfb'
OBSTACLE = '#52514e'
COBST = '#d9d8d4'
SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


# ---- plotting helpers ---------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
        'ytick.color': INK2, 'text.color': INK, 'axes.titlecolor': INK,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
        'axes.titlesize': 10, 'legend.frameon': False, 'lines.linewidth': 2.0,
    })
    return plt


def draw_world(ax, world, tags=True, cobst_rho=None):
    from matplotlib.patches import FancyBboxPatch, Rectangle
    b = world.box
    ax.add_patch(Rectangle((world.x_min, world.y_min), world.x_max - world.x_min,
                           world.y_max - world.y_min, fill=False, ec=INK, lw=1.5, zorder=2))
    if cobst_rho:  # true C-obstacle: the box grown by rho (a rounded rectangle)
        ax.add_patch(FancyBboxPatch((b.cx - b.hx, b.cy - b.hy), b.sx, b.sy,
                                    boxstyle=f'round,pad={cobst_rho}', fc=COBST, ec='none',
                                    zorder=1))
    ax.add_patch(Rectangle((b.cx - b.hx, b.cy - b.hy), b.sx, b.sy, fc=OBSTACLE, ec='none',
                           zorder=3))
    if tags:
        for tag, (x, y, yaw) in sorted(world.landmarks.items()):
            ax.plot([x], [y], marker='s', ms=5, color=INK2, zorder=6, ls='none')
            ax.annotate('', (x + 0.12 * math.cos(yaw), y + 0.12 * math.sin(yaw)), (x, y),
                        arrowprops={'arrowstyle': '->', 'color': INK2, 'lw': 0.8}, zorder=6)
            ax.text(x - 0.07 * math.cos(yaw), y - 0.07 * math.sin(yaw), str(tag),
                    color=INK2, fontsize=7, ha='center', va='center', zorder=6)
    sx, sy, st = world.start
    ax.plot([sx], [sy], marker='o', ms=9, mfc=SURFACE, mec=INK, mew=1.5, zorder=8)
    ax.annotate('', (sx + 0.18 * math.cos(st), sy + 0.18 * math.sin(st)), (sx, sy),
                arrowprops={'arrowstyle': '->', 'color': INK, 'lw': 1.2}, zorder=8)
    ax.text(sx + 0.06, sy - 0.12, 'start', color=INK, fontsize=8, zorder=8)
    gx, gy, gt = world.goal
    ax.plot([gx], [gy], marker='*', ms=13, mfc=SURFACE, mec=INK, mew=1.2, zorder=8)
    ax.text(gx + 0.08, gy + 0.06, 'goal', color=INK, fontsize=8, zorder=8)
    pad = 0.12
    ax.set_xlim(world.x_min - pad, world.x_max + pad)
    ax.set_ylim(world.y_min - pad, world.y_max + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m] (east)')
    ax.set_ylabel('y [m] (north)')


def draw_path(ax, path, mode, label=None, waypoints=None, lw=2.0, alpha=1.0, number=True):
    xs, ys = zip(*path)
    ax.plot(xs, ys, color=COLORS[mode], ls=STYLES[mode], lw=lw, alpha=alpha, label=label,
            zorder=7, solid_capstyle='round')
    if waypoints is not None:
        for i, (x, y, t) in enumerate(waypoints):
            passing = math.isnan(t)
            ax.plot([x], [y], marker=MARKERS[mode], ms=8 if not passing else 6,
                    mfc=COLORS[mode] if not passing else SURFACE, mec=SURFACE if not passing
                    else COLORS[mode], mew=1.5, zorder=9, ls='none')
            if number and i < len(waypoints) - 1:  # the last one is the goal
                ax.text(x + 0.05, y + 0.05, str(i), fontsize=7, color=INK2, zorder=9)


def _legend_below(ax):
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=7)


def _metrics_text(m):
    return (f'length {m["length_m"]:.2f} m · est. {m["est_time_s"]:.1f} s · '
            f'{m["stops"]} stops · min body gap {m["min_body_gap_m"]:.2f} m · '
            f'mean clearance {m["mean_clearance_m"]:.2f} m')


def figure_fast(world, plan, out):
    plt = _plt()
    d = plan.detail
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    draw_world(ax, world, cobst_rho=d.rho)
    poly = d.polygon + [d.polygon[0]]
    ax.plot(*zip(*poly), color=MUTED, lw=1.0, zorder=4, label=f'C-obstacle polygon '
            f'(rho = {d.rho:.2f} m, k = {world.fast["polygon_k"]})')
    for i, j in d.edges:
        (x1, y1), (x2, y2) = d.nodes[i], d.nodes[j]
        ax.plot([x1, x2], [y1, y2], color=MUTED, lw=0.6, alpha=0.7, zorder=4)
    ax.plot([], [], color=MUTED, lw=0.6, label=f'visibility graph ({len(d.nodes)} nodes, '
            f'{len(d.edges)} edges)')
    nx_, ny_ = zip(*d.nodes[2:])
    ax.plot(nx_, ny_, 'o', ms=4, color=INK2, zorder=5)
    if len(d.raw_path) != len(d.path):
        ax.plot(*zip(*d.raw_path), color=COLORS['fast'], lw=1.0, ls=(0, (2, 2)), zorder=6,
                label='A* path before corner merge')
    draw_path(ax, d.path, 'fast', 'fast plan (A*, time cost)', plan.waypoints, lw=2.5)
    ax.set_title('Minimum time: visibility graph in configuration space\n'
                 + _metrics_text(plan.metrics), fontsize=9)
    _legend_below(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure_safe(world, plan, out):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    d = plan.detail
    g = d.grid
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE)
    D = np.where(g.D > 0, g.D, np.nan)
    im = ax.imshow(D, origin='lower', extent=g.extent(), cmap=cmap, zorder=0,
                   interpolation='nearest')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('clearance d [m] (distance to nearest obstacle)')
    cb.outline.set_visible(False)
    draw_world(ax, world, tags=True)
    X = g.x0 + np.arange(g.nx) * g.res
    Y = g.y0 + np.arange(g.ny) * g.res
    ax.contour(X, Y, g.D, levels=[world.robot_radius], colors=[INK], linewidths=0.8,
               linestyles='--', zorder=4)
    ax.plot([], [], color=INK, ls='--', lw=0.8, label=f'C-obstacle boundary '
            f'(d = r = {world.robot_radius:.2f} m)')
    iy, ix = np.nonzero(d.gvd)
    ax.plot(g.x0 + ix * g.res, g.y0 + iy * g.res, 's', ms=1.2, color=INK, alpha=0.55,
            zorder=5, label=f'GVD cells ({len(ix)})')
    ax.plot(*zip(*d.raw_path), color=COLORS['safe'], lw=1.0, ls=(0, (2, 2)), zorder=6,
            label='Dijkstra chain on the GVD')
    draw_path(ax, d.path, 'safe', 'safe plan (RDP-simplified)', plan.waypoints, lw=2.5)
    ax.set_title('Maximum clearance: generalised Voronoi diagram on a clearance grid\n'
                 + _metrics_text(plan.metrics), fontsize=9)
    _legend_below(ax)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure_overview(world, plans, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    draw_world(ax, world)
    for mode, plan in plans.items():
        m = plan.metrics
        draw_path(ax, plan.path, mode, f'{mode}: {m["length_m"]:.2f} m, gap '
                  f'{m["min_body_gap_m"]:.2f} m, ~{m["est_time_s"]:.0f} s', plan.waypoints,
                  number=False)
        mid = plan.path[len(plan.path) // 2]
        ax.text(mid[0] + 0.05, mid[1] - 0.12, mode, color=INK, fontsize=8, zorder=9,
                bbox={'fc': SURFACE, 'ec': COLORS[mode], 'lw': 1.0,
                      'boxstyle': 'round,pad=0.2'})
    ax.set_title('Planned paths (filled marker = stop, hollow = pass-through)')
    _legend_below(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure_profiles(world, plans, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for mode, plan in plans.items():
        s, clear, _ = clearance_profile(world, plan.path)
        ax.plot(s, clear - world.robot_radius, color=COLORS[mode], ls=STYLES[mode],
                label=mode)
        k = int(np.argmin(clear))
        ax.plot([s[k]], [clear[k] - world.robot_radius], marker=MARKERS[mode], ms=7,
                color=COLORS[mode], mec=SURFACE, mew=1.5, ls='none')
    ax.axhline(0.0, color=INK, lw=1.0)
    ax.text(0.02, 0.01, 'contact', color=INK2, fontsize=7, va='bottom')
    ax.set_xlabel('distance along the path [m]')
    ax.set_ylabel('body gap to nearest obstacle [m]')
    ax.set_title('Clearance along each plan (marker = minimum)')
    ax.legend(loc='upper center', ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def figure_tradeoff(world, sweep, plans, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    L = [r['length_m'] for r in sweep]
    G = [r['min_body_gap_m'] for r in sweep]
    ax.plot(L, G, color=COLORS['weighted'], ls=STYLES['weighted'], marker=MARKERS['weighted'],
            ms=7, mec=SURFACE, label='weighted A*, alpha sweep')
    groups = {}  # alphas that give the same path share one label
    for r in sweep:
        groups.setdefault((round(r['length_m'], 4), round(r['min_body_gap_m'], 4)),
                          []).append(r['alpha'])
    for (length, gap), alphas in groups.items():
        ax.annotate('a=' + ', '.join(f'{a:g}' for a in alphas), (length, gap), fontsize=7,
                    color=INK2, xytext=(4, -10), textcoords='offset points')
    for mode in ('fast', 'safe'):
        if mode in plans:
            m = plans[mode].metrics
            ax.plot([m['length_m']], [m['min_body_gap_m']], marker=MARKERS[mode], ms=10,
                    color=COLORS[mode], mec=SURFACE, mew=1.5, ls='none', label=mode)
            ax.annotate(mode, (m['length_m'], m['min_body_gap_m']), fontsize=8, color=INK,
                        xytext=(6, 4), textcoords='offset points')
    ax.set_xlabel('path length [m]')
    ax.set_ylabel('minimum body gap [m]')
    ax.set_title('Safety vs distance (the box gap; start/goal clearance excluded)')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ---- check --------------------------------------------------------------------------------
def visible_tags(world, pose):
    return world.tags_in_view(pose)


def cmd_check(args):
    world = load_world(args.world)
    print(world.describe())
    problems = world.check()
    print('\nProblems:' if problems else '\nNo problems found in the world file.')
    for p in problems:
        print(f'  - {p}')
    print(f'\nTags in view (camera half FOV {world.camera["half_fov"]:.2f} rad, range '
          f'{world.camera["max_range"]:.1f} m, box occludes):')
    print(f'  at the start heading {world.start[2]:+.2f}: {visible_tags(world, world.start)}')
    gx, gy, gt = world.goal
    plans = {}
    for mode in ('fast', 'safe'):
        try:
            plans[mode] = hw4_plan.make_plan(world, mode)
        except Exception as e:  # noqa: BLE001
            print(f'  {mode} plan FAILED: {e}')
    for mode, plan in plans.items():
        print(f'  {mode} plan stops (heading -> tags in view):')
        for i, (x, y, th) in enumerate(plan.waypoints):
            if not math.isnan(th):
                print(f'    [{i}] ({x:.2f}, {y:.2f}) {th:+.2f}: {visible_tags(world, (x, y, th))}')
    good = []
    for deg in range(-180, 180, 15):
        th = math.radians(deg)
        n = visible_tags(world, (gx, gy, th))
        if n:
            good.append(f'{th:+.2f} rad ({deg} deg): {n}')
    print('  goal headings that see a tag (with goal theta null and camera.confirm_heading the '
          'planner picks the one nearest the arrival heading):')
    for line in good or ['  none: check camera.half_fov / max_range']:
        print(f'    {line}')
    if plans:
        print()
        _print_table(world, plans)
    return 1 if problems else 0


def _print_table(world, plans):
    keys = [('length_m', 'length [m]'), ('waypoints', 'waypoints'), ('stops', 'stops'),
            ('min_body_gap_m', 'min body gap [m]'), ('min_box_gap_m', 'min box gap [m]'),
            ('mean_clearance_m', 'mean clearance [m]'), ('est_time_s', 'est. time [s]')]
    head = f'{"":>20}' + ''.join(f'{m:>11}' for m in plans)
    print(head)
    for k, name in keys:
        row = f'{name:>20}'
        for plan in plans.values():
            v = plan.metrics[k]
            row += f'{v:>11.3f}' if isinstance(v, float) else f'{v:>11}'
        print(row)


# ---- plan ---------------------------------------------------------------------------------
def cmd_plan(args):
    world = load_world(args.world)
    for p in world.check():
        print(f'WARNING: {p}')
    out = os.path.abspath(args.out or hw4_plan.plans_dir())
    os.makedirs(out, exist_ok=True)
    plans = {}
    for mode in args.modes:
        plan = hw4_plan.make_plan(world, mode)
        plans[mode] = plan
        hw4_plan.write_plan(plan, world, out, waypoints_name=f'waypoints_hw4_{mode}.txt',
                            plan_name=f'plan_{mode}.json', landmarks_name='landmarks_hw4.txt')
        print(f'{mode:>8}: {hw4_plan.DESCRIPTIONS[mode]}')
        for i, (x, y, t) in enumerate(plan.waypoints):
            print(f'          [{i}] {x:.3f}, {y:.3f}, {"pass" if math.isnan(t) else f"{t:.3f}"}')
    print()
    _print_table(world, plans)
    sweep = []
    if args.tradeoff:
        from robot_control.gvd_planner import build_grid
        from robot_control.path_tools import make_waypoints, path_metrics
        from robot_control.gvd_planner import plan_weighted
        grid = build_grid(world, world.weighted['resolution'])
        for a in world.weighted['tradeoff_alphas']:
            path, _ = plan_weighted(world, a, grid)
            m = path_metrics(world, path, make_waypoints(path, world.goal[2], 'stop'))
            sweep.append({'alpha': a, **m})
            print(f'  alpha {a:5.2f}: length {m["length_m"]:.3f} m, min box gap '
                  f'{m["min_box_gap_m"]:.3f} m, est {m["est_time_s"]:.1f} s')
        with open(os.path.join(out, 'tradeoff.json'), 'w') as f:
            json.dump(sweep, f, indent=2)
    if not args.no_figures:
        if 'fast' in plans:
            figure_fast(world, plans['fast'], os.path.join(out, 'plan_fast.png'))
        if 'safe' in plans:
            figure_safe(world, plans['safe'], os.path.join(out, 'plan_safe.png'))
        figure_overview(world, plans, os.path.join(out, 'plans_overview.png'))
        figure_profiles(world, plans, os.path.join(out, 'clearance_profiles.png'))
        if sweep:
            for r in sweep:  # the box gap is the trade-off; start/goal sit at fixed clearance
                r['min_body_gap_m'] = r['min_box_gap_m']
            figure_tradeoff(world, sweep, {m: _box_gap_view(p) for m, p in plans.items()},
                            os.path.join(out, 'tradeoff.png'))
    _write_plan_summary(world, plans, out)
    print(f'\nWrote plans, landmarks_hw4.txt and figures to {out}')
    return 0


def _box_gap_view(plan):
    m = dict(plan.metrics)
    m['min_body_gap_m'] = m['min_box_gap_m']
    return hw4_plan.Plan(plan.mode, plan.path, plan.waypoints, m)


def _write_plan_summary(world, plans, out):
    lines = ['# HW4 plans', '', '```', world.describe(), '```', '',
             '| metric | ' + ' | '.join(plans) + ' |', '|---|' + '---|' * len(plans)]
    for k in ('length_m', 'waypoints', 'stops', 'pass_through', 'min_body_gap_m',
              'min_box_gap_m', 'mean_clearance_m', 'est_drive_s', 'est_turn_s',
              'est_vertex_s', 'est_time_s'):
        vals = [plans[m].metrics[k] for m in plans]
        lines.append(f'| {k} | ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v)
                                              for v in vals) + ' |')
    lines.append('')
    for mode, plan in plans.items():
        lines.append(f'## {mode}: {hw4_plan.DESCRIPTIONS[mode]}')
        lines.append(f'extras: {plan.extras}')
        lines.append('')
        for i, (x, y, t) in enumerate(plan.waypoints):
            lines.append(f'- [{i}] ({x:.3f}, {y:.3f}) '
                         + ('pass-through' if math.isnan(t) else f'theta {t:.3f}'))
        lines.append('')
    with open(os.path.join(out, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines))


# ---- sim ----------------------------------------------------------------------------------
def write_run_info(run_dir, info):
    with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


def cmd_sim(args):
    from robot_control.hw4_sim import load_params_yaml, simulate, write_sim_logs
    world = load_world(args.world)
    params = load_params_yaml(args.params) if os.path.exists(args.params) else {}
    modes = ['fast', 'safe'] if args.mode == 'both' else [args.mode]
    worst = 0
    for mode in modes:
        plan = hw4_plan.make_plan(world, mode)
        for seed in range(args.seed, args.seed + args.seeds):
            run_dir = os.path.join(hw4_plan.logs_dir(), hw4_plan.sim_run_name(mode, seed))
            hw4_plan.write_plan(plan, world, run_dir)
            write_run_info(run_dir, {'mode': mode, 'simulated': True, 'seed': seed,
                                     'world_file': world.source, 'params_file': args.params})
            result = simulate(world, plan.waypoints, params, seed=seed)
            write_sim_logs(run_dir, result)
            res = evaluate(run_dir, quiet=True)
            gap = result['min_true_clearance'] - world.robot_radius
            print(f'{mode:>5} seed {seed}: finished={result["done"]} '
                  f'time {res["duration_s"]:.1f} s, true min body gap {gap:.3f} m'
                  f'{"  COLLISION" if result["collided"] else ""}, true final error '
                  f'{res.get("true_final_error_m", float("nan")):.3f} m -> {run_dir}')
            if result['collided'] or not result['done']:
                worst = 1
    return worst


# ---- eval ---------------------------------------------------------------------------------
def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def read_end_pose(run_dir):
    path = os.path.join(run_dir, 'end_pose.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line:
                vals = [float(v) for v in line.replace(',', ' ').split()]
                if len(vals) in (2, 3):
                    return vals
    return None


def _cross_track(path, x, y):
    return min(point_segment_distance((x, y), a, b) for a, b in zip(path[:-1], path[1:]))


def evaluate(run_dir, end=None, quiet=False):
    """Metrics and figures for one run folder; writes RUN/eval/results.{json,md} and PNGs."""
    with open(os.path.join(run_dir, 'plan.json')) as f:
        plan = json.load(f)
    world_file = os.path.join(run_dir, 'world.yaml')
    world = load_world(world_file if os.path.exists(world_file) else plan['world_file'])
    mode = plan['mode']
    path = [tuple(p) for p in plan['path']]
    traj = read_csv(os.path.join(run_dir, 'trajectory.csv'))
    reached = read_csv(os.path.join(run_dir, 'waypoints.csv'))
    meas = read_csv(os.path.join(run_dir, 'measurements.csv'))
    truth = read_csv(os.path.join(run_dir, 'truth.csv'))
    info = {}
    if os.path.exists(os.path.join(run_dir, 'run_info.json')):
        with open(os.path.join(run_dir, 'run_info.json')) as f:
            info = json.load(f)
    if end is not None:
        with open(os.path.join(run_dir, 'end_pose.txt'), 'w') as f:
            f.write('# tape-measured final pose: x, y[, theta] (world frame)\n'
                    + ', '.join(f'{v:.4f}' for v in end) + '\n')
    else:
        end = read_end_pose(run_dir)
    if not traj:
        raise RuntimeError(f'{run_dir} has no trajectory.csv rows')

    r = world.robot_radius
    n_wp = len(plan['waypoints'])
    finished = len(reached) >= n_wp
    moving = [row for row in traj if abs(row.get('v', 0.0)) > 1e-4 or abs(row.get('w', 0.0))
              > 1e-4]
    t_start = moving[0]['t'] if moving else traj[0]['t']
    t_end = reached[-1]['t'] if finished else traj[-1]['t']
    seg = [row for row in traj if t_start - 1e-9 <= row['t'] <= t_end + 1e-9] or traj
    xs = np.array([row['x'] for row in seg])
    ys = np.array([row['y'] for row in seg])
    clear = world.clearance(xs, ys)
    boxd = world.box.distance(xs, ys)
    driven = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    xt = [_cross_track(path, x, y) for x, y in zip(xs, ys)]
    pos_std = [math.hypot(row['std_x'], row['std_y']) for row in seg]
    final = traj[-1] if not finished else min(traj, key=lambda row: abs(row['t'] - t_end))
    gx, gy, gt = world.goal
    goal_theta = plan['waypoints'][-1][2]
    res = {
        'run': os.path.basename(run_dir), 'mode': mode, 'simulated': bool(truth),
        'finished': finished, 'waypoints_reached': len(reached), 'waypoints_total': n_wp,
        'duration_s': t_end - t_start, 'driven_length_m': driven,
        'planned_length_m': plan['metrics']['length_m'],
        'planned_est_time_s': plan['metrics']['est_time_s'],
        'planned_min_body_gap_m': plan['metrics']['min_body_gap_m'],
        'est_min_body_gap_m': float(clear.min() - r),
        'est_mean_clearance_m': float(clear.mean()),
        'est_min_box_gap_m': float(boxd.min() - r),
        'mean_cross_track_m': float(np.mean(xt)), 'max_cross_track_m': float(np.max(xt)),
        'max_pos_std_m': float(max(pos_std)), 'mean_pos_std_m': float(np.mean(pos_std)),
        'est_final_error_m': math.hypot(final['x'] - gx, final['y'] - gy),
        'est_final_heading_error_rad': (abs(wrap_angle(final['theta'] - goal_theta))
                                        if goal_theta is not None else None),
        'tag_sightings': len(meas),
        'tag_accepted': int(sum(m.get('accepted', 0) for m in meas)),
    }
    if os.path.exists(os.path.join(run_dir, 'run_end.json')):
        with open(os.path.join(run_dir, 'run_end.json')) as f:
            res['guard_tripped'] = json.load(f).get('guard_tripped', False)
    if end is not None:
        res['measured_end'] = end
        res['true_final_error_m'] = math.hypot(end[0] - gx, end[1] - gy)
        res['est_vs_measured_end_m'] = math.hypot(end[0] - final['x'], end[1] - final['y'])
    if truth:
        tx = np.array([row['x'] for row in truth])
        ty = np.array([row['y'] for row in truth])
        res['true_min_body_gap_m'] = float(world.clearance(tx, ty).min() - r)
        last = truth[-1]
        res['true_final_error_m'] = math.hypot(last['x'] - gx, last['y'] - gy)
        res['est_vs_true_end_m'] = math.hypot(last['x'] - traj[-1]['x'],
                                              last['y'] - traj[-1]['y'])
    res['info'] = info

    out = os.path.join(run_dir, 'eval')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump(res, f, indent=2)
    _write_results_md(res, out)
    _eval_figures(world, plan, traj, truth, reached, seg, t_start, out)
    if not quiet:
        print(open(os.path.join(out, 'results.md')).read())
        print(f'Figures in {out}')
    return res


def _write_results_md(res, out):
    def f(v):
        if isinstance(v, float):
            return f'{v:.3f}'
        return str(v)
    lines = [f'# {res["run"]} ({res["mode"]}{", simulated" if res["simulated"] else ""})', '',
             '| metric | value |', '|---|---|']
    for k, v in res.items():
        if k in ('run', 'mode', 'info'):
            continue
        lines.append(f'| {k} | {f(v)} |')
    with open(os.path.join(out, 'results.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def _eval_figures(world, plan, traj, truth, reached, seg, t_start, out):
    plt = _plt()
    from matplotlib.patches import Ellipse
    mode = plan['mode']
    r = world.robot_radius
    fig, ax = plt.subplots(figsize=(6.4, 7.4))
    draw_world(ax, world)
    xs, ys = zip(*plan['path'])
    ax.plot(xs, ys, color=COLORS[mode], ls=(0, (3, 2)), lw=1.5, zorder=6, label='plan')
    ax.plot([row['x'] for row in traj], [row['y'] for row in traj], color=COLORS[mode], lw=2.0,
            zorder=7, label='EKF estimate')
    if truth:
        ax.plot([row['x'] for row in truth], [row['y'] for row in truth], color=INK, lw=1.0,
                zorder=7, label='true (simulation)')
    last = -math.inf
    for row in traj:  # 2-sigma position ellipses every ~2 s
        if row['t'] - last < 2.0:
            continue
        last = row['t']
        ax.add_patch(Ellipse((row['x'], row['y']), 4 * row['std_x'], 4 * row['std_y'],
                             fill=False, ec=INK2, lw=0.6, zorder=6))
    ax.plot([], [], color=INK2, lw=0.6, label='2-sigma (every 2 s)')
    if reached:
        ax.plot([w['x'] for w in reached], [w['y'] for w in reached], ls='none',
                marker=MARKERS[mode], ms=7, color=COLORS[mode], mec=SURFACE, mew=1.5,
                zorder=9, label='waypoint reached (estimate)')
    ax.set_title(f'{mode} run: plan vs estimated trajectory')
    _legend_below(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'trajectory.png'), dpi=160)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    t = np.array([row['t'] for row in traj]) - t_start
    c = world.clearance([row['x'] for row in traj], [row['y'] for row in traj]) - r
    a1.plot(t, c, color=COLORS[mode], label='estimate')
    if truth:
        tt = np.array([row['t'] for row in truth]) - t_start
        a1.plot(tt, world.clearance([row['x'] for row in truth], [row['y'] for row in truth])
                - r, color=INK, lw=1.0, label='true (simulation)')
    a1.axhline(0.0, color=INK, lw=1.0)
    a1.axhline(plan['metrics']['min_body_gap_m'], color=INK2, lw=0.8, ls='--')
    a1.text(t[0], plan['metrics']['min_body_gap_m'], ' planned minimum', color=INK2,
            fontsize=7, va='bottom')
    a1.set_ylabel('body gap [m]')
    a1.set_title('Clearance and localisation uncertainty over the run (t = 0: first motion)')
    a1.legend(loc='upper right', fontsize=7)
    a2.plot(t, [100 * math.hypot(row['std_x'], row['std_y']) for row in traj],
            color=COLORS[mode], label='position std [cm]')
    a2.plot(t, [math.degrees(row['std_theta']) for row in traj], color=INK2, lw=1.2,
            ls=(0, (3, 2)), label='heading std [deg]')
    a2.set_xlabel('time [s]')
    a2.set_ylabel('std (cm or deg)')
    a2.legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'clearance_uncertainty.png'), dpi=160)
    plt.close(fig)


def cmd_eval(args):
    for spec in args.runs or ['latest']:
        run = hw4_plan.resolve_run(spec)
        evaluate(run, end=args.end, quiet=args.quiet)
    return 0


# ---- summary ------------------------------------------------------------------------------
def cmd_summary(args):
    runs = hw4_plan.all_runs(include_sim=args.sim)
    rows = []
    for run in runs:
        path = os.path.join(run, 'eval', 'results.json')
        try:
            res = evaluate(run, quiet=True) if args.reeval or not os.path.exists(path) else \
                json.load(open(path))
        except Exception as e:  # noqa: BLE001
            print(f'skipping {os.path.basename(run)}: {e}')
            continue
        rows.append(res)
    cols = [('run', 'run'), ('mode', 'mode'), ('finished', 'done'),
            ('duration_s', 'time [s]'), ('driven_length_m', 'driven [m]'),
            ('est_min_body_gap_m', 'min gap est [m]'),
            ('true_min_body_gap_m', 'min gap true [m]'),
            ('est_mean_clearance_m', 'mean clear [m]'),
            ('mean_cross_track_m', 'x-track [m]'), ('est_final_error_m', 'end err est [m]'),
            ('true_final_error_m', 'end err meas [m]')]
    lines = ['| ' + ' | '.join(c[1] for c in cols) + ' |', '|' + '---|' * len(cols)]
    for res in rows:
        cells = []
        for k, _ in cols:
            v = res.get(k)
            cells.append('-' if v is None else f'{v:.3f}' if isinstance(v, float) else str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    table = '\n'.join(lines)
    print(table)
    out = os.path.join(hw4_plan.logs_dir(), 'summary.md')
    with open(out, 'w') as f:
        f.write(table + '\n')
    print(f'\n{len(rows)} runs; table written to {out}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='hw4', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--world', default=hw4_plan.default_world_file(),
                        help='world file (default: config/hw4_world.yaml)')
    sub = parser.add_subparsers(dest='cmd', required=True)

    sub.add_parser('check', help='validate the world file; tags in view at start and goal')

    p = sub.add_parser('plan', help='plan and write waypoints + figures')
    p.add_argument('modes', nargs='*', metavar='MODE',
                   help=f'any of {", ".join(hw4_plan.MODES)} (default: all)')
    p.add_argument('--out', help='output folder (default: <workspace>/hw4_plans)')
    p.add_argument('--tradeoff', action='store_true', help='also sweep weighted.alpha')
    p.add_argument('--no-figures', action='store_true')

    p = sub.add_parser('sim', help='simulate runs offline and evaluate them')
    p.add_argument('mode', choices=list(hw4_plan.MODES) + ['both'])
    p.add_argument('--seeds', type=int, default=1, help='number of runs')
    p.add_argument('--seed', type=int, default=0, help='first seed')
    p.add_argument('--params', default=hw4_plan.default_params_file(),
                   help='node parameters (default: config/hw4.yaml)')

    p = sub.add_parser('eval', help='evaluate run folders (default: latest robot run)')
    p.add_argument('runs', nargs='*', help="run folder, 'latest', 'latest-sim' or a name part")
    p.add_argument('--end', type=float, nargs='+', metavar='V',
                   help='tape-measured final pose: x y [theta] (saved to RUN/end_pose.txt)')
    p.add_argument('--quiet', action='store_true')

    p = sub.add_parser('summary', help='table of all evaluated runs')
    p.add_argument('--sim', action='store_true', help='include simulations')
    p.add_argument('--reeval', action='store_true', help='re-evaluate every run first')

    args = parser.parse_args(argv)
    if args.cmd == 'plan':
        args.modes = args.modes or list(hw4_plan.MODES)
        bad = [m for m in args.modes if m not in hw4_plan.MODES]
        if bad:
            parser.error(f'unknown mode(s) {bad}; choose from {hw4_plan.MODES}')
    if getattr(args, 'end', None) is not None and len(args.end) not in (2, 3):
        parser.error('--end takes x y [theta]')
    return {'check': cmd_check, 'plan': cmd_plan, 'sim': cmd_sim, 'eval': cmd_eval,
            'summary': cmd_summary}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
