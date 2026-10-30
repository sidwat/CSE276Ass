"""Plot the navigator CSV; estimates and command-derived metrics are not ground truth."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheel-base', type=float, default=.68)
    args = parser.parse_args()
    with args.csv_path.open() as stream:
        records = list(csv.DictReader(stream))
    if len(records) < 2:
        parser.error('at least two samples required')
    args.output.mkdir(parents=True, exist_ok=False)
    t, x, y, yaw, v, w = [np.array([float(r[key]) for r in records])
                          for key in ['t', 'x', 'y', 'yaw', 'v', 'w']]
    dt = np.diff(t)
    if not np.all(dt > 0):
        parser.error('CSV time must increase strictly; split runs at clock resets')
    dots = np.r_[True, np.diff(np.floor((t-t[0])/.5)) > 0]
    travelled = float(np.hypot(np.diff(x[dots]), np.diff(y[dots])).sum())
    dyaw = np.diff(yaw[dots])
    rotation = float(np.abs(np.arctan2(np.sin(dyaw), np.cos(dyaw))).sum())
    left, right = v-w*args.wheel_base/2, v+w*args.wheel_base/2
    # Commands at row i act over the following sample interval.
    wheel_travel = float(np.sum((abs(left[:-1])+abs(right[:-1]))*.5*dt))
    summary = dict(duration_s=float(t[-1]-t[0]), estimated_path_length_m=travelled,
        estimated_rotation_rad=rotation, command_model_mean_wheel_travel_m=wheel_travel,
        max_command_acceleration_m_s2=float(np.max(abs(np.diff(v)/dt))),
        max_command_yaw_acceleration_rad_s2=float(np.max(abs(np.diff(w)/dt))),
        note='Not physical ground truth; pose noise inflates travel; proposed commands may be passive')
    (args.output/'metrics.json').write_text(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x[dots], y[dots], '.-', label='estimated 0.5 s samples')
    axes[0].plot([0, 1, 1, 0], [0, 0, 2, 0], 'kx--', label='goals')
    axes[0].set(xlabel='x [m]', ylabel='y [m]', aspect='equal')
    axes[0].legend()
    axes[1].plot(t-t[0], v, label='proposed v [m/s]')
    axes[1].plot(t-t[0], w, label='proposed w [rad/s]')
    axes[1].set(xlabel='time [s]')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output/'debug.png', dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
