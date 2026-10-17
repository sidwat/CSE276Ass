"""Plan the trajectory with the robot's parameters, plot it and animate a simulated run."""

import argparse
from dataclasses import fields
import math
import os

import numpy as np
from robot_control.trajectory_planner import plan_trajectory, PlannerConfig
from robot_control.waypoint_controller import body_to_wheel_cmds, load_waypoints
import yaml


def default_config_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('robot_control'), 'config')
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')


def load_params(path):
    with open(path) as f:
        params = yaml.safe_load(f)['trajectory_control_node']['ros__parameters']
    config = PlannerConfig(**{f.name: f.type(params[f.name])
                              for f in fields(PlannerConfig) if f.name in params})
    return config, float(params.get('control_rate', 20.0))


def simulate_open_loop(trajectory, config, dt, right_gain=1.0):
    """Integrate the wheel commands the robot would send, with an optional right-wheel error."""
    times = np.arange(0.0, trajectory.duration + dt, dt)
    x, y, theta = 0.0, 0.0, trajectory.sample(0.0).heading
    poses, cmds = [], []
    for t in times:
        sample = trajectory.sample(t)
        left, right = body_to_wheel_cmds(
            sample.v, sample.w, config.speed_per_unit, config.wheel_base, config.max_wheel_cmd)
        cmds.append((left, right))
        poses.append((x, y, theta))
        v_left = left * config.speed_per_unit
        v_right = right * config.speed_per_unit * right_gain
        v, w = (v_left + v_right) / 2.0, (v_right - v_left) / config.wheel_base
        x += v * math.cos(theta + w * dt / 2.0) * dt
        y += v * math.sin(theta + w * dt / 2.0) * dt
        theta += w * dt
    return times, np.array(poses), np.array(cmds)


def corridor_outline(a, b, width, n=40):
    """Stadium-shaped region within width of the segment a-b."""
    d = math.atan2(b[1] - a[1], b[0] - a[0])
    arc_b = d + np.linspace(-math.pi / 2, math.pi / 2, n)
    arc_a = d + np.linspace(math.pi / 2, 3 * math.pi / 2, n)
    pts = [b + width * np.array([math.cos(t), math.sin(t)]) for t in arc_b]
    pts += [a + width * np.array([math.cos(t), math.sin(t)]) for t in arc_a]
    return np.array(pts)


def print_summary(waypoints, trajectory):
    report = trajectory.report
    print(f'Duration {trajectory.duration:.1f} s, max curvature {report.max_curvature:.2f} '
          f'(limit {report.curvature_limit:.2f}) 1/m')
    for i, ((wx, wy, wt), t) in enumerate(zip(waypoints, trajectory.waypoint_times)):
        s = trajectory.sample(t)
        print(f'  wp[{i}] ({wx:5.2f}, {wy:5.2f}) t={t:5.1f}s planned ({s.x:5.2f}, {s.y:5.2f}) '
              f'heading {math.degrees(s.heading):7.1f} deg (file {math.degrees(wt):6.1f})')
    print('  segments: ' + ', '.join(
        ('straight' if s else 'curve') + (' fwd' if d > 0 else ' rev')
        for d, s in zip(report.directions, report.straight)))
    if report.stops:
        print(f'  touches zero speed to change direction at waypoints {report.stops}')
    print('  corridor deviation per segment [m]: '
          + ', '.join(f'{d:.3f}' for d in report.max_deviation))
    for violation in report.violations:
        print(f'  WARNING: {violation}')


def main():
    config_dir = default_config_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--waypoints', default=os.path.join(config_dir, 'waypoints.txt'))
    parser.add_argument('--params', default=os.path.join(config_dir, 'trajectory_control.yaml'))
    parser.add_argument('--mismatch', type=float, default=0.03,
                        help='right-wheel speed error for the drift simulation (0.03 = +3%%)')
    parser.add_argument('--speedup', type=float, default=4.0, help='animation speed vs real time')
    parser.add_argument('--save', help='save the static figure to this file')
    parser.add_argument('--no-show', action='store_true', help='do not open a window')
    args = parser.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    waypoints = load_waypoints(args.waypoints)
    config, control_rate = load_params(args.params)
    trajectory = plan_trajectory(waypoints, config)
    print_summary(waypoints, trajectory)

    dt = 1.0 / control_rate
    times, ideal, cmds = simulate_open_loop(trajectory, config, dt)
    _, drift, _ = simulate_open_loop(trajectory, config, dt, 1.0 + args.mismatch)
    planned = [trajectory.sample(t) for t in times]
    speed = np.array([abs(s.v) for s in planned])
    curvature = np.array([s.curvature for s in planned])
    final_err = math.hypot(*(ideal[-1, :2] - np.array(waypoints[-1][:2])))
    drift_err = math.hypot(*(drift[-1, :2] - np.array(waypoints[-1][:2])))
    print(f'Open-loop end error: {final_err * 100:.1f} cm ideal, '
          f'{drift_err * 100:.1f} cm with {args.mismatch:+.0%} right-wheel error')

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.3, 1])
    ax_xy = fig.add_subplot(grid[:, 0])
    ax_speed = fig.add_subplot(grid[0, 1])
    ax_kappa = fig.add_subplot(grid[1, 1], sharex=ax_speed)
    ax_cmd = fig.add_subplot(grid[2, 1], sharex=ax_speed)

    lo = np.array([config.arena_min_x, config.arena_min_y])
    ax_xy.add_patch(plt.Rectangle(lo, config.arena_size, config.arena_size, fill=False,
                                  ec='k', lw=1.5, ls='--', label='7 ft arena'))
    for i in range(len(waypoints) - 1):
        a, b = np.array(waypoints[i][:2]), np.array(waypoints[i + 1][:2])
        outline = corridor_outline(a, b, config.corridor_width)
        ax_xy.fill(outline[:, 0], outline[:, 1], color='tab:green', alpha=0.07,
                   label='1 ft corridor' if i == 0 else None)
        ax_xy.plot([a[0], b[0]], [a[1], b[1]], color='tab:green', lw=0.8, ls=':')

    labelled = set()
    for leg in trajectory.legs:
        label = 'planned (forward)' if leg.direction > 0 else 'planned (reverse)'
        ax_xy.plot(leg.xy[:, 0], leg.xy[:, 1], lw=2.2,
                   color='tab:blue' if leg.direction > 0 else 'tab:orange',
                   label=None if label in labelled else label)
        labelled.add(label)
    for n, leg in enumerate(trajectory.legs[1:]):
        ax_xy.plot(*leg.xy[0], 's', ms=11, mfc='none', mec='tab:red', mew=1.8,
                   label='stop to change direction' if n == 0 else None)
    ax_xy.plot(drift[:, 0], drift[:, 1], color='tab:red', lw=1, ls='--',
               label=f'sim, {args.mismatch:+.0%} right wheel')

    for i, ((wx, wy, wt), t) in enumerate(zip(waypoints, trajectory.waypoint_times)):
        h = trajectory.sample(t).heading
        ax_xy.annotate('', (wx + 0.22 * math.cos(wt), wy + 0.22 * math.sin(wt)), (wx, wy),
                       arrowprops={'arrowstyle': '-|>', 'color': '0.55', 'lw': 3.5})
        ax_xy.annotate('', (wx + 0.18 * math.cos(h), wy + 0.18 * math.sin(h)), (wx, wy),
                       arrowprops={'arrowstyle': '->', 'color': 'tab:blue', 'lw': 1.5})
        ax_xy.plot(wx, wy, 'o', color='k', ms=6)
        ax_xy.annotate(f'{i}', (wx, wy), textcoords='offset points', xytext=(6, -12))
    ax_xy.plot([], [], color='0.55', lw=3.5, label='heading in file')
    ax_xy.plot([], [], color='tab:blue', lw=1.5, label='planned heading')
    ax_xy.set_aspect('equal')
    ax_xy.set_xlabel('x [m]')
    ax_xy.set_ylabel('y [m]')
    ax_xy.set_title('Planned trajectory (robot heading at each waypoint vs file)')
    ax_xy.grid(alpha=0.3)
    ax_xy.legend(loc='upper left', fontsize=8)

    ax_speed.plot(times, speed, color='tab:blue')
    ax_speed.set_ylabel('|v| [m/s]')
    ax_speed.set_title('Speed, curvature and wheel commands')
    ax_kappa.plot(times, curvature, color='tab:purple')
    lim = trajectory.report.curvature_limit
    ax_kappa.axhline(lim, color='r', lw=0.8, ls=':')
    ax_kappa.axhline(-lim, color='r', lw=0.8, ls=':')
    ax_kappa.set_ylabel('curvature [1/m]')
    ax_cmd.plot(times, cmds[:, 0], label='L')
    ax_cmd.plot(times, cmds[:, 1], label='R')
    ax_cmd.axhline(config.max_wheel_cmd, color='r', lw=0.8, ls=':')
    ax_cmd.axhline(-config.max_wheel_cmd, color='r', lw=0.8, ls=':')
    ax_cmd.set_ylabel('wheel cmd')
    ax_cmd.set_xlabel('time [s]')
    ax_cmd.legend(loc='lower left', fontsize=8)
    for ax in (ax_speed, ax_kappa, ax_cmd):
        ax.grid(alpha=0.3)
        for t in trajectory.waypoint_times:
            ax.axvline(t, color='0.7', lw=0.6)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f'Saved {args.save}')
    if args.no_show:
        return 0

    robot = ax_xy.quiver([0.0], [0.0], [1.0], [0.0], color='k', scale=12, width=0.012, zorder=5)
    trail, = ax_xy.plot([], [], color='k', lw=1, alpha=0.6)
    cursors = [ax.axvline(0.0, color='k', lw=1) for ax in (ax_speed, ax_kappa, ax_cmd)]
    step = max(1, int(round(args.speedup * 0.05 / dt)))
    frames = range(0, len(times), step)

    def update(i):
        x, y, theta = ideal[i]
        robot.set_offsets([[x, y]])
        robot.set_UVC([math.cos(theta)], [math.sin(theta)])
        trail.set_data(ideal[:i + 1, 0], ideal[:i + 1, 1])
        for cursor in cursors:
            cursor.set_xdata([times[i], times[i]])
        return [robot, trail, *cursors]

    fig.animation = FuncAnimation(fig, update, frames=frames, interval=50, repeat=True)
    plt.show()
    return 0


if __name__ == '__main__':
    main()
