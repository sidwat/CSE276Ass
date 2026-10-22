"""
Record motor_commands and imu_data during a run; on exit save CSVs and a diagnostic plot.

run_recorder (ROS node) writes <output_dir>/<tag>_<timestamp>_{cmd,imu}.csv, _meta.json and .png.
plot_run <prefix> re-draws the plot from those files.
"""

import csv
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from robot_control.waypoint_controller import load_waypoints
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

WATCHDOG = 0.15  # s, motor_control stops the wheels if no command arrives within this
MAX_CMD = 0.5    # motor_control clips each wheel command to +/- this
CALIBRATION_DEFAULTS = {
    'speed_per_unit': 1.26, 'wheel_base': 0.72, 'wheel_trim': -0.06,
    'reverse_speed_per_unit': 1.26, 'reverse_trim': -0.06,
    'ccw_deg_per_s': 83.3, 'ccw_offset_deg': 10.0, 'cw_deg_per_s': 93.3, 'cw_offset_deg': 20.0,
    'imu_yaw_sign': 1.0,
}


def _spin_rates(sent_left, sent_right, dt, meta):
    """Yaw rate of each in-place spin run, from the calibrated rate plus start/stop offset."""
    w = np.zeros(len(dt))
    spin = sent_left * sent_right < 0
    i = 0
    while i < len(dt):
        if not spin[i]:
            i += 1
            continue
        ccw = sent_right[i] > 0
        j = i
        while j < len(dt) and spin[j] and (sent_right[j] > 0) == ccw:
            j += 1
        duration = max(float(np.sum(dt[i:j])), 1e-6)
        if ccw:
            angle = meta['ccw_deg_per_s'] * duration + meta['ccw_offset_deg']
        else:
            angle = -(meta['cw_deg_per_s'] * duration + meta['cw_offset_deg'])
        w[i:j] = math.radians(angle) / duration
        i = j
    return spin, w


def dead_reckon(cmd, meta):
    """Pose the robot should have from the commands actually sent (t, L, R rows)."""
    t = cmd[:, 0]
    dt = np.minimum(np.diff(t, append=t[-1]), WATCHDOG)
    sent_left = np.clip(cmd[:, 1], -MAX_CMD, MAX_CMD)
    sent_right = np.clip(cmd[:, 2], -MAX_CMD, MAX_CMD)
    trim, spu = meta['wheel_trim'], meta['speed_per_unit']
    # Undo the trim: trim only corrects motor mismatch, so the nominal command is the motion
    left = sent_left / (1.0 - trim)
    right = sent_right / (1.0 + trim)
    if meta.get('quintic', True):
        v = (left + right) / 2.0 * spu
        w = (right - left) * spu / meta['wheel_base']
    else:
        # Turn-drive-turn: spins follow the spin calibration, straights the drive calibration
        spin, w = _spin_rates(sent_left, sent_right, dt, meta)
        reverse = (sent_left < 0) & (sent_right < 0)
        r_trim = meta.get('reverse_trim', trim)
        r_left = sent_left / (1.0 - r_trim)
        r_right = sent_right / (1.0 + r_trim)
        v = np.where(reverse, (r_left + r_right) / 2.0 * meta.get('reverse_speed_per_unit', spu),
                     (left + right) / 2.0 * spu)
        v[spin] = 0.0
    start = (meta.get('waypoints') or [[0.0, 0.0, 0.0]])[0]
    x = np.full(len(t), float(start[0]))
    y = np.full(len(t), float(start[1]))
    theta = np.full(len(t), float(start[2]))
    for i in range(len(t) - 1):
        mid = theta[i] + w[i] * dt[i] / 2.0
        x[i + 1] = x[i] + v[i] * math.cos(mid) * dt[i]
        y[i + 1] = y[i] + v[i] * math.sin(mid) * dt[i]
        theta[i + 1] = theta[i] + w[i] * dt[i]
    return {'t': t, 'v': v, 'w': w, 'x': x, 'y': y, 'theta': theta}


def imu_heading(imu, t0, yaw_sign):
    """IMU yaw unwrapped and zeroed at the first command time."""
    if len(imu) == 0:
        return None
    yaw = np.unwrap(imu[:, 1])
    yaw0 = yaw[int(np.argmin(np.abs(imu[:, 0] - t0)))]
    return {'t': imu[:, 0], 'theta': yaw_sign * (yaw - yaw0), 'gz': imu[:, 2]}


def summarize(cmd, dr, imu_h):
    moving = np.flatnonzero((np.abs(cmd[:, 1]) > 1e-6) | (np.abs(cmd[:, 2]) > 1e-6))
    gaps = np.diff(cmd[:, 0])
    lines = [f'{len(cmd)} commands over {cmd[-1, 0] - cmd[0, 0]:.1f} s']
    if len(moving):
        lines.append(f'moving from t={cmd[moving[0], 0]:.1f}s to t={cmd[moving[-1], 0]:.1f}s')
    if len(gaps):
        lines.append(f'max gap between commands {gaps.max() * 1000:.0f} ms; '
                     f'{int(np.sum(gaps > WATCHDOG))} gaps > {WATCHDOG * 1000:.0f} ms watchdog')
    clipped = int(np.sum((np.abs(cmd[:, 1]) > MAX_CMD) | (np.abs(cmd[:, 2]) > MAX_CMD)))
    lines.append(f'{clipped} commands clipped at +/-{MAX_CMD} by motor_control')
    lines.append(f'final commanded pose x={dr["x"][-1]:.2f} m y={dr["y"][-1]:.2f} m '
                 f'theta={math.degrees(dr["theta"][-1]):.1f} deg')
    if imu_h is not None:
        lines.append(f'final IMU heading {math.degrees(imu_h["theta"][-1]):.1f} deg '
                     f'(commanded {math.degrees(dr["theta"][-1]):.1f} deg)')
    return lines


def plot_run(prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from robot_control.simulate_trajectory import corridor_outline

    with open(prefix + '_meta.json') as f:
        meta = json.load(f)
    cmd = np.loadtxt(prefix + '_cmd.csv', delimiter=',', skiprows=1, ndmin=2)
    imu = np.loadtxt(prefix + '_imu.csv', delimiter=',', skiprows=1, ndmin=2)
    if len(cmd) < 2:
        print('Fewer than 2 motor commands recorded, nothing to plot')
        return None
    imu = imu.reshape(-1, 3)

    dr = dead_reckon(cmd, meta)
    imu_h = imu_heading(imu, cmd[0, 0], meta['imu_yaw_sign'])
    t = dr['t']

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(5, 2, width_ratios=[1.15, 1])
    ax_xy = fig.add_subplot(grid[0:3, 0])
    ax_pos = fig.add_subplot(grid[3, 0])
    ax_gap = fig.add_subplot(grid[4, 0], sharex=ax_pos)
    ax_cmd = fig.add_subplot(grid[0, 1], sharex=ax_pos)
    ax_v = fig.add_subplot(grid[1, 1], sharex=ax_pos)
    ax_w = fig.add_subplot(grid[2, 1], sharex=ax_pos)
    ax_th = fig.add_subplot(grid[3, 1], sharex=ax_pos)
    ax_err = fig.add_subplot(grid[4, 1], sharex=ax_pos)

    waypoints = meta.get('waypoints') or []
    for i in range(len(waypoints) - 1):
        a, b = np.array(waypoints[i][:2]), np.array(waypoints[i + 1][:2])
        outline = corridor_outline(a, b, 0.3048)
        ax_xy.fill(outline[:, 0], outline[:, 1], color='tab:green', alpha=0.07,
                   label='1 ft corridor' if i == 0 else None)
        ax_xy.plot([a[0], b[0]], [a[1], b[1]], color='tab:green', lw=0.8, ls=':')
    for i, (wx, wy, wt) in enumerate(waypoints):
        ax_xy.annotate('', (wx + 0.2 * math.cos(wt), wy + 0.2 * math.sin(wt)), (wx, wy),
                       arrowprops={'arrowstyle': '-|>', 'color': '0.55', 'lw': 3})
        ax_xy.plot(wx, wy, 'ko', ms=5)
        ax_xy.annotate(str(i), (wx, wy), textcoords='offset points', xytext=(6, -12))

    fwd = np.where(dr['v'] >= 0, 1.0, np.nan)
    rev = np.where(dr['v'] < 0, 1.0, np.nan)
    ax_xy.plot(dr['x'] * fwd, dr['y'] * fwd, color='tab:blue', lw=2, label='commanded, forward')
    ax_xy.plot(dr['x'] * rev, dr['y'] * rev, color='tab:orange', lw=2,
               label='commanded, reverse')
    for tick in np.arange(math.ceil(t[0] / 5.0) * 5.0, t[-1], 5.0):
        i = int(np.argmin(np.abs(t - tick)))
        ax_xy.plot(dr['x'][i], dr['y'][i], 'k.', ms=6)
        ax_xy.annotate(f'{tick - t[0]:.0f}s', (dr['x'][i], dr['y'][i]),
                       textcoords='offset points', xytext=(4, 4), fontsize=7)
    ax_xy.set_aspect('equal')
    ax_xy.grid(alpha=0.3)
    ax_xy.set_xlabel('x [m]')
    ax_xy.set_ylabel('y [m]')
    ax_xy.set_title('Pose dead-reckoned from the published L/R (open-loop expectation)')
    ax_xy.legend(loc='best', fontsize=8)

    rel = t - t[0]
    ax_pos.plot(rel, dr['x'], label='x')
    ax_pos.plot(rel, dr['y'], label='y')
    ax_pos.set_ylabel('position [m]')
    ax_pos.legend(fontsize=8, loc='upper left')

    ax_gap.plot(rel[1:], np.diff(t) * 1000.0, '.', ms=3, color='tab:gray')
    ax_gap.axhline(WATCHDOG * 1000.0, color='r', ls=':', lw=1, label='watchdog')
    ax_gap.set_ylabel('cmd gap [ms]')
    ax_gap.set_xlabel('time since first command [s]')
    ax_gap.legend(fontsize=8, loc='upper left')

    ax_cmd.plot(rel, cmd[:, 1], label='L sent')
    ax_cmd.plot(rel, cmd[:, 2], label='R sent')
    for lim in (MAX_CMD, -MAX_CMD):
        ax_cmd.axhline(lim, color='r', ls=':', lw=1)
    ax_cmd.set_ylabel('wheel cmd')
    if meta.get('quintic', True):
        ax_cmd.set_title(f'QUINTIC  speed_per_unit={meta["speed_per_unit"]}  '
                         f'wheel_base={meta["wheel_base"]}  wheel_trim={meta["wheel_trim"]}')
    else:
        ax_cmd.set_title(f'TURN-DRIVE-TURN  spu={meta["speed_per_unit"]}  '
                         f'trim={meta["wheel_trim"]}  CCW {meta["ccw_deg_per_s"]}/s'
                         f'+{meta["ccw_offset_deg"]}  CW {meta["cw_deg_per_s"]}/s'
                         f'+{meta["cw_offset_deg"]}')
    ax_cmd.legend(fontsize=8, loc='upper left')

    ax_v.plot(rel, dr['v'], color='tab:blue')
    ax_v.axhline(0.0, color='k', lw=0.5)
    ax_v.set_ylabel('v [m/s]')

    ax_w.plot(rel, dr['w'], color='tab:purple', label='commanded w')
    ax_w.set_ylabel('w [rad/s]')
    if imu_h is not None:
        ax_gz = ax_w.twinx()
        ax_gz.plot(imu_h['t'] - t[0], imu_h['gz'], color='tab:red', lw=0.8, alpha=0.6)
        ax_gz.set_ylabel('IMU gz (board units)', color='tab:red')

    ax_th.plot(rel, np.degrees(dr['theta']), label='commanded')
    if imu_h is not None:
        ax_th.plot(imu_h['t'] - t[0], np.degrees(imu_h['theta']), label='IMU')
        err = imu_h['theta'] - np.interp(imu_h['t'], t, dr['theta'])
        ax_err.plot(imu_h['t'] - t[0], np.degrees(err), color='tab:red')
        ax_err.axhline(0.0, color='k', lw=0.5)
        ax_err.set_ylabel('IMU - cmd [deg]')
    else:
        path = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(dr['x']), np.diff(dr['y'])))])
        ax_err.plot(rel, path, color='tab:green')
        ax_err.set_ylabel('distance driven [m]')
    ax_th.set_ylabel('heading [deg]')
    ax_th.legend(fontsize=8, loc='upper left')
    ax_err.set_xlabel('time since first command [s]')

    for ax in (ax_pos, ax_gap, ax_cmd, ax_v, ax_w, ax_th, ax_err):
        ax.grid(alpha=0.3)
    ax_pos.set_xlim(rel[0], rel[-1])

    lines = summarize(cmd, dr, imu_h)
    fig.suptitle(os.path.basename(prefix) + '   |   ' + '   |   '.join(lines[-2:]), fontsize=9)
    fig.tight_layout()
    fig.savefig(prefix + '.png', dpi=110)
    plt.close(fig)
    for line in lines:
        print('  ' + line)
    print(f'Saved plot {prefix}.png', flush=True)
    return prefix + '.png'


class RunRecorder(Node):

    def __init__(self):
        super().__init__('run_recorder')
        for name, default in CALIBRATION_DEFAULTS.items():
            self.declare_parameter(name, default)
        self.declare_parameter('quintic', False)
        self.declare_parameter('use_imu', False)
        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('output_dir', '~/robot_runs')
        self.declare_parameter('tag', 'run')

        self.cmd_rows = []
        self.imu_rows = []
        self.create_subscription(Float32MultiArray, 'motor_commands', self.on_cmd, 50)
        if self.get_parameter('use_imu').value:
            self.create_subscription(Imu, 'imu_data', self.on_imu, 50)
        self.get_logger().info('Recording motor_commands; Ctrl-C to save and plot')

    def on_cmd(self, msg):
        if len(msg.data) >= 2:
            self.cmd_rows.append((time.monotonic(), float(msg.data[0]), float(msg.data[1])))

    def on_imu(self, msg):
        q = msg.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.imu_rows.append((time.monotonic(), yaw, float(msg.angular_velocity.z)))

    def save(self):
        """Write the CSVs and metadata (fast), then the plot. Returns the file prefix."""
        param = self.get_parameter
        out_dir = os.path.expanduser(param('output_dir').value)
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = os.path.join(out_dir, f'{param("tag").value}_{stamp}')

        t0 = self.cmd_rows[0][0] if self.cmd_rows else 0.0
        for suffix, header, rows in (('_cmd.csv', ('t', 'L', 'R'), self.cmd_rows),
                                     ('_imu.csv', ('t', 'yaw', 'gz'), self.imu_rows)):
            with open(prefix + suffix, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows((r[0] - t0, r[1], r[2]) for r in rows)

        waypoints_file = param('waypoints_file').value
        meta = {name: float(param(name).value) for name in CALIBRATION_DEFAULTS}
        meta['quintic'] = bool(param('quintic').value)
        meta['waypoints'] = load_waypoints(waypoints_file) if waypoints_file else []
        with open(prefix + '_meta.json', 'w') as f:
            json.dump(meta, f, indent=1)
        print(f'Saved {len(self.cmd_rows)} commands and {len(self.imu_rows)} IMU samples '
              f'to {prefix}_*.csv', flush=True)
        return prefix


def main(args=None):
    rclpy.init(args=args)
    node = RunRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        prefix = node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    plot_run(prefix)
    return 0


def plot_main():
    if len(sys.argv) != 2:
        print('usage: plot_run <prefix, e.g. ~/robot_runs/trajectory_20261017_120000>')
        return 1
    prefix = os.path.expanduser(sys.argv[1])
    for suffix in ('_cmd.csv', '_imu.csv', '_meta.json', '.png'):
        if prefix.endswith(suffix):
            prefix = prefix[:-len(suffix)]
    plot_run(prefix)
    return 0


if __name__ == '__main__':
    main()
