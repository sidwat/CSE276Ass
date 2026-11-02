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


def draw_poses(ax, poses, color, label, marker='o', line=False, arrows=True):
    """Dots at (x, y), each with a short arrow along its heading."""
    if not poses:
        return
    xs, ys, thetas = zip(*poses)
    if line:
        ax.plot(xs, ys, color='0.8', linewidth=1, zorder=1)
    ax.scatter(xs, ys, s=12, color=color, marker=marker, zorder=2, label=label)
    if arrows:
        ax.quiver(xs, ys, [math.cos(t) for t in thetas], [math.sin(t) for t in thetas],
                  color=color, angles='xy', scale_units='xy', scale=10, width=0.003,
                  headwidth=4, zorder=2)  # scale=10: unit vector drawn 0.1 m long


def unwrapped_turn(thetas):
    return sum(wrap_angle(b - a) for a, b in zip(thetas, thetas[1:]))


def print_turn_check(trajectory):
    """Compare commanded rotation with the rotation the EKF (i.e. the tags) saw."""
    commanded = unwrapped_turn([p['odom_theta'] for p in trajectory])
    seen = unwrapped_turn([p['theta'] for p in trajectory])
    print(f'\nNet rotation: commanded (motion model) {math.degrees(commanded):.0f} deg, '
          f'EKF {math.degrees(seen):.0f} deg')
    if abs(seen) > math.radians(90):
        ratio = commanded / seen
        print(f'The robot turns {100 / ratio:.0f}% of what is commanded. If that is far from '
              f'100%, multiply wheel_base by {ratio:.2f} (and divide min_w and max_w by '
              f'{ratio:.2f} to keep the same wheel commands).')


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

    if 'odom_theta' in trajectory[0]:
        print_turn_check(trajectory)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    goals = [(w['goal_x'], w['goal_y'], w['goal_theta']) for w in reached]
    fused = [(p['x'], p['y'], p['theta']) for p in trajectory]

    # Report figure: the EKF estimate, one dot per log_period
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_map(ax, landmarks, goals)
    draw_poses(ax, fused, 'red', 'EKF estimate (every log_period)', line=True)
    finish(ax, 'Estimated trajectory')
    output = args.output or os.path.join(run_dir, 'trajectory.png')
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f'\nWrote {output}')

    # Motion model, sensor model, and the EKF that fuses them
    odom = [(p['odom_x'], p['odom_y'], p['odom_theta']) for p in trajectory] \
        if 'odom_x' in trajectory[0] else []
    predicted = [(p['pred_x'], p['pred_y'], p['pred_theta']) for p in trajectory[1:]] \
        if 'pred_x' in trajectory[0] else []
    sightings = read_csv(measurements_path) if os.path.exists(measurements_path) else []
    sightings = [m for m in sightings if int(m['tag']) in landmarks]
    for m in sightings:
        m['pose'] = pose_from_sighting(m, landmarks[int(m['tag'])])
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    (ax_motion, ax_sensor), (ax_fused, ax_all) = axes
    for ax in axes.ravel():
        draw_map(ax, landmarks, goals)

    # Each prediction starts from the EKF estimate one log_period earlier; the gap between a
    # green point and the next red one is the correction the tags made over that period.
    ax_motion.plot([p[0] for p in fused], [p[1] for p in fused], color='red', alpha=0.25,
                   linewidth=1, label='EKF estimate (prediction start points)')
    for start_pose, end_pose in zip(fused, predicted):
        ax_motion.plot([start_pose[0], end_pose[0]], [start_pose[1], end_pose[1]],
                       color='tab:green', linewidth=0.8)
    draw_poses(ax_motion, predicted, 'tab:green', 'motion model prediction from previous EKF')
    if odom:
        ax_motion.plot([p[0] for p in odom], [p[1] for p in odom], color='tab:green',
                       linestyle=':', alpha=0.5, label='dead reckoning from origin (no tags)')
    finish(ax_motion, 'Motion model (prediction step)' if predicted
           else 'Motion model: not in this log (older run)')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, tag in enumerate(sorted({int(m['tag']) for m in sightings})):
        for accepted, marker in ((1, 'o'), (0, 'x')):
            rows = [m for m in sightings
                    if int(m['tag']) == tag and int(m['accepted']) == accepted]
            if rows:
                draw_poses(ax_sensor, [m['pose'] for m in rows], colors[i % len(colors)],
                           f'tag {tag}' + ('' if accepted else ' (rejected)'), marker=marker)
    finish(ax_sensor, 'Sensor model only (pose implied by each tag sighting)')

    draw_poses(ax_fused, fused, 'red', 'EKF estimate', line=True)
    finish(ax_fused, 'EKF: motion and sensor models fused')

    draw_poses(ax_all, predicted, 'tab:green', 'motion model prediction', arrows=False)
    ax_all.scatter([m['pose'][0] for m in sightings], [m['pose'][1] for m in sightings],
                   s=6, color='tab:blue', alpha=0.4, label='sensor model')
    ax_all.plot([p[0] for p in fused], [p[1] for p in fused], color='red', linewidth=2,
                label='EKF')
    finish(ax_all, 'All three')

    output = os.path.join(run_dir, 'models.png')
    fig.savefig(output, dpi=120, bbox_inches='tight')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
