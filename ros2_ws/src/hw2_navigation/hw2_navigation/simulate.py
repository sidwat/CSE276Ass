"""Seeded algorithm smoke experiment, NOT a camera/visibility or hardware simulator."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from robot_control.waypoint_controller import Gains, WaypointController
from hw2_navigation.core import PoseFilter, wrap

WAYPOINTS = [(0, 0, 0), (1, 0, 0), (1, 2, math.pi), (0, 0, 0)]


def run(strategy='filtered', seed=1, dropout=0.15, blackout=(25.0, 27.0)):
    rng = np.random.default_rng(seed)
    truth = np.zeros(3)
    estimator = PoseFilter()
    controller = WaypointController(WAYPOINTS, Gains(max_v=.12, max_w=.6, min_v=.03, min_w=.15))
    dt = .05
    last_seen = -float('inf')
    v = w = 0.0
    rows, events = [], []
    distance = rotation = wheel_travel = 0.0
    for step in range(6000):
        t = step*dt
        estimator.predict(v, w, dt)
        # Independent synthetic truth has scale bias and turn bias absent from the predictor.
        tv, tw = v*.9, w*1.07 + (.025 if abs(v) > 0 else 0)
        h = truth[2]+tw*dt/2
        truth += [tv*dt*math.cos(h), tv*dt*math.sin(h), tw*dt]
        truth[2] = wrap(truth[2])
        distance += abs(tv)*dt
        rotation += abs(tw)*dt
        wheel_travel += (abs(tv-tw*.68/2)+abs(tv+tw*.68/2))*dt/2
        if step % 4 == 0 and not blackout[0] <= t <= blackout[1] and rng.random() > dropout:
            noise = np.array([.02, .02, .04])
            z = truth+rng.normal(0, noise)
            if strategy == 'visual':
                estimator.x = z.copy()
                estimator.x[2] = wrap(estimator.x[2])
                last_seen = t
            elif strategy == 'filtered':
                accepted, _ = estimator.correct(z, np.diag(noise**2))
                if accepted:
                    last_seen = t
        reached = None
        if strategy != 'baseline' and t-last_seen > .5:
            v = w = 0.0
        else:
            v, w, reached = controller.update(*estimator.x)
        if reached is not None:
            goal = WAYPOINTS[reached]
            events.append(dict(waypoint=reached, t=t,
                translation_error_m=math.hypot(truth[0]-goal[0], truth[1]-goal[1]),
                yaw_error_rad=wrap(truth[2]-goal[2])))
        rows.append((t, *truth, *estimator.x, v, w, t-last_seen, reached))
        if controller.done:
            break
    return rows, dict(strategy=strategy, seed=seed, dropout=dropout, complete=controller.done,
        duration_s=t, distance_m=distance, rotation_rad=rotation,
        mean_wheel_travel_m=wheel_travel, waypoint_errors=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', choices=['baseline', 'visual', 'filtered'], default='filtered')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=.15)
    parser.add_argument('--output', type=Path, default=Path('/tmp/hw2-sim'))
    args = parser.parse_args()
    if not 0 <= args.dropout <= 1:
        parser.error('dropout must be between 0 and 1')
    args.output.mkdir(parents=True, exist_ok=False)
    rows, summary = run(args.strategy, args.seed, args.dropout)
    with (args.output/'trajectory.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t', 'true_x', 'true_y', 'true_yaw', 'x', 'y', 'yaw', 'v', 'w', 'visual_age', 'reached'])
        writer.writerows(rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = np.asarray([row[:10] for row in rows], dtype=float)
    fig, ax = plt.subplots()
    ax.plot(data[:, 1], data[:, 2], label='synthetic truth')
    ax.plot(data[::10, 4], data[::10, 5], '.', label='estimate at 0.5 s')
    ax.plot(*np.asarray(WAYPOINTS)[:, :2].T, 'kx--', label='goals')
    ax.set(xlabel='x [m]', ylabel='y [m]', title=args.strategy, aspect='equal')
    ax.legend()
    fig.savefig(args.output/'trajectory.png', dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
