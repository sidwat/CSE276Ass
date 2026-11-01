"""Plot a velocity_control run log (log_dir/run_*/) and print the numbers for the HW2 report."""

import argparse
import csv
import glob
import math
import os

from robot_control.localization import load_landmarks
from robot_control.waypoint_controller import wrap_angle


def read_csv(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def print_tag_check(measurements):
    """Compare each tag sighting with what the pose estimate predicted, to catch setup errors."""
    print('\nTag  sightings  accepted  range/expected  bearing sign agrees')
    for tag in sorted({int(m['tag']) for m in measurements}):
        rows = [m for m in measurements if int(m['tag']) == tag]
        ratios = sorted(m['range'] / m['exp_range'] for m in rows)
        # Only sightings well off the image centre say anything about the sign
        off_centre = [m for m in rows if abs(m['exp_bearing']) > 0.15]
        agree = sum((m['bearing'] > 0) == (m['exp_bearing'] > 0) for m in off_centre)
        sign = f'{agree}/{len(off_centre)}' if off_centre else 'n/a (always near centre)'
        print(f'  {tag:<4} {len(rows):>6}  {int(sum(m["accepted"] for m in rows)):>8}'
              f'  {ratios[len(ratios) // 2]:>13.3f}   {sign}')
    print('A bearing sign that mostly disagrees means the image is upside down '
          '(camera_upside_down: true).\nA range ratio away from 1 for every tag means the tag '
          'size is wrong; for one tag, its position in landmarks.txt.')


def pose_from_sighting(m, landmark):
    """Robot pose implied by one tag sighting alone (sensor model, no motion model)."""
    lx, ly, lyaw = landmark
    theta = wrap_angle(lyaw - m['orientation'])
    heading = theta + m['bearing']
    return (lx - m['range'] * math.cos(heading), ly - m['range'] * math.sin(heading), theta)


def draw_map(ax, landmarks, goals):
    for gx, gy, gt in goals:
        ax.annotate('', xy=(gx + 0.15 * math.cos(gt), gy + 0.15 * math.sin(gt)), xytext=(gx, gy),
                    arrowprops={'arrowstyle': '->', 'color': 'blue'})
    ax.scatter([g[0] for g in goals], [g[1] for g in goals], s=80, facecolors='none',
               edgecolors='blue', linewidths=1.5, zorder=3, label='waypoint')
    for tag_id, (lx, ly, lyaw) in landmarks.items():
        half = 0.09
        ax.plot([lx - half * math.sin(lyaw), lx + half * math.sin(lyaw)],
                [ly + half * math.cos(lyaw), ly - half * math.cos(lyaw)],
                color='black', linewidth=4)
        ax.annotate(f'tag {tag_id}', (lx, ly), textcoords='offset points', xytext=(6, 6))


def finish(ax, title):
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize='small')
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', nargs='?',
                        help='a run_* folder (default: the newest one in ~/hw2_logs)')
    parser.add_argument('--output', help='image file (default: <run_dir>/trajectory.png)')
    args = parser.parse_args()

    run_dir = args.run_dir or max(glob.glob(os.path.expanduser('~/hw2_logs/run_*')))
    trajectory = read_csv(os.path.join(run_dir, 'trajectory.csv'))
    reached = read_csv(os.path.join(run_dir, 'waypoints.csv'))
    landmarks_path = os.path.join(run_dir, 'landmarks.txt')
    landmarks = load_landmarks(landmarks_path) if os.path.exists(landmarks_path) else {}

    distance = sum(math.hypot(b['x'] - a['x'], b['y'] - a['y'])
                   for a, b in zip(trajectory, trajectory[1:]))
    rotation = sum(abs(wrap_angle(b['theta'] - a['theta']))
                   for a, b in zip(trajectory, trajectory[1:]))
    print(f'Run {run_dir}: {trajectory[-1]["t"]:.1f} s')
    print(f'Distance driven {distance:.2f} m, rotation {rotation:.2f} rad '
          f'({math.degrees(rotation):.0f} deg), estimated from the {len(trajectory)} logged poses')
    print('\nWaypoint  goal (x, y, theta)       estimate at arrival      est. error (m, deg)')
    for w in reached:
        err_pos = math.hypot(w['x'] - w['goal_x'], w['y'] - w['goal_y'])
        err_rot = math.degrees(wrap_angle(w['theta'] - w['goal_theta']))
        print(f'  [{int(w["index"])}]'
              f'   ({w["goal_x"]:5.2f}, {w["goal_y"]:5.2f}, {w["goal_theta"]:5.2f})'
              f'   ({w["x"]:5.2f}, {w["y"]:5.2f}, {w["theta"]:5.2f})'
              f'   {err_pos:.3f}, {err_rot:+.1f}')
    print('The estimate errors only show controller tolerance; measure the true pose at each '
          'waypoint marker with a tape for the localization error.')

    measurements_path = os.path.join(run_dir, 'measurements.csv')
    if os.path.exists(measurements_path):
        print_tag_check(read_csv(measurements_path))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    goals = [(w['goal_x'], w['goal_y'], w['goal_theta']) for w in reached]
    fused = ([p['x'] for p in trajectory], [p['y'] for p in trajectory])

    # Report figure: the EKF estimate, one dot per log_period
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_map(ax, landmarks, goals)
    ax.plot(*fused, color='0.8', linewidth=1, zorder=1)
    ax.scatter(*fused, s=14, color='red', zorder=2, label='EKF estimate (every log_period)')
    finish(ax, 'Estimated trajectory')
    output = args.output or os.path.join(run_dir, 'trajectory.png')
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f'\nWrote {output}')

    # Motion model alone, sensor model alone, and the EKF that fuses them
    has_odom = 'odom_x' in trajectory[0]
    odom = ([p['odom_x'] for p in trajectory], [p['odom_y'] for p in trajectory]) \
        if has_odom else ([], [])
    sightings = read_csv(measurements_path) if os.path.exists(measurements_path) else []
    sightings = [m for m in sightings if int(m['tag']) in landmarks]
    for m in sightings:
        m['pose'] = pose_from_sighting(m, landmarks[int(m['tag'])])
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    (ax_odom, ax_sensor), (ax_fused, ax_all) = axes
    for ax in axes.ravel():
        draw_map(ax, landmarks, goals)

    ax_odom.plot(*odom, color='tab:green', marker='.', markersize=5, linewidth=1)
    finish(ax_odom, 'Motion model only (commanded velocities)'
           if has_odom else 'Motion model: not in this log (older run)')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, tag in enumerate(sorted({int(m['tag']) for m in sightings})):
        for accepted, marker in ((1, 'o'), (0, 'x')):
            rows = [m for m in sightings
                    if int(m['tag']) == tag and int(m['accepted']) == accepted]
            if not rows:
                continue
            ax_sensor.scatter([m['pose'][0] for m in rows], [m['pose'][1] for m in rows],
                              s=10, marker=marker,
                              color=colors[i % len(colors)],
                              label=f'tag {tag}' + ('' if accepted else ' (rejected)'))
    finish(ax_sensor, 'Sensor model only (one dot per tag sighting)')

    ax_fused.plot(*fused, color='0.8', linewidth=1, zorder=1)
    ax_fused.scatter(*fused, s=14, color='red', zorder=2)
    finish(ax_fused, 'EKF: motion and sensor models fused')

    ax_all.plot(*odom, color='tab:green', linewidth=1.5, label='motion model')
    ax_all.scatter([m['pose'][0] for m in sightings], [m['pose'][1] for m in sightings],
                   s=6, color='tab:blue', alpha=0.4, label='sensor model')
    ax_all.plot(*fused, color='red', linewidth=2, label='EKF')
    finish(ax_all, 'All three')

    output = os.path.join(run_dir, 'models.png')
    fig.savefig(output, dpi=120, bbox_inches='tight')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
