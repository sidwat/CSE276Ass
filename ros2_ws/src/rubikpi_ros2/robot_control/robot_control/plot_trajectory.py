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

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([p['x'] for p in trajectory], [p['y'] for p in trajectory],
            color='0.8', linewidth=1, zorder=1)
    ax.scatter([p['x'] for p in trajectory], [p['y'] for p in trajectory],
               s=14, color='red', zorder=2, label='estimated pose (every log_period)')
    goals = [(w['goal_x'], w['goal_y'], w['goal_theta']) for w in reached]
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
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_title('Estimated trajectory')
    output = args.output or os.path.join(run_dir, 'trajectory.png')
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f'\nWrote {output}')


if __name__ == '__main__':
    main()
