"""
Robot standing still at a known pose (default (0, 0, 0)): compare the tags with landmarks.txt.

Needs camera_node and apriltag_node running. Prints, per tag and pooled, the mean error and
standard deviation of range, bearing and tag orientation (measured - expected) and of the pose
each sighting implies by itself, and writes samples.csv, summary.csv and static_check.png to
--output. Use --pose X Y THETA when the robot stands somewhere else, e.g. on a waypoint mark.
"""

import argparse
import csv
import math
import os
import re
import time

import numpy as np
import rclpy
from rclpy.node import Node
from robot_control.localization import (
    config_dir, expected_measurement, load_landmarks, pose_from_measurement, tag_measurement)
from robot_control.waypoint_controller import wrap_angle
from tf2_msgs.msg import TFMessage

CONFIG = config_dir()


class Collector(Node):
    def __init__(self, landmarks, camera_pose, camera_frame, upside_down=False):
        super().__init__('static_check')
        self.landmarks, self.camera_pose, self.upside_down = landmarks, camera_pose, upside_down
        self.camera_frame = camera_frame
        self.samples = []
        self.create_subscription(TFMessage, '/tf', self.on_tf, 100)

    def on_tf(self, msg):
        for tf in msg.transforms:
            match = re.search(r'[_:](\d+)$', tf.child_frame_id)
            if tf.header.frame_id != self.camera_frame or not match:
                continue
            tag_id = int(match.group(1))
            if tag_id not in self.landmarks:
                continue
            t, q = tf.transform.translation, tf.transform.rotation
            normal = (2.0 * (q.x * q.z + q.w * q.y), 2.0 * (q.y * q.z - q.w * q.x),
                      1.0 - 2.0 * (q.x * q.x + q.y * q.y))
            translation = (t.x, t.y, t.z)
            z = tag_measurement(translation, normal, self.camera_pose, self.upside_down)
            pose = pose_from_measurement(z, self.landmarks[tag_id])
            self.samples.append((time.monotonic(), tag_id, *z, *pose, *translation))


def summarize(rows, landmarks, pose):
    """rows: array columns [tag, range, bearing, orientation, px, py, ptheta, cx, cy, cz]."""
    errors = []
    for r in rows:
        er, eb, eo = expected_measurement(pose, landmarks[int(r[0])])
        errors.append((r[1] - er, wrap_angle(r[2] - eb), wrap_angle(r[3] - eo),
                       math.hypot(r[4] - pose[0], r[5] - pose[1]), wrap_angle(r[6] - pose[2])))
    return np.array(errors)  # range [m], bearing, orientation, |pose error| [m], heading [rad]


def stats(errors):
    mean, std = errors.mean(axis=0), errors.std(axis=0)
    deg_mean, deg_std = np.degrees(mean), np.degrees(std)
    return (f'range {mean[0] * 100:+6.1f} +- {std[0] * 100:4.1f} cm | '
            f'bearing {deg_mean[1]:+5.1f} +- {deg_std[1]:3.1f} deg | '
            f'orient {deg_mean[2]:+6.1f} +- {deg_std[2]:4.1f} deg | '
            f'implied pose off {mean[3] * 100:5.1f} +- {std[3] * 100:4.1f} cm, '
            f'heading {deg_mean[4]:+5.1f} +- {deg_std[4]:3.1f} deg')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--landmarks', default=str(CONFIG / 'landmarks.txt'))
    parser.add_argument('--params', default=str(CONFIG / 'hw2.yaml'))
    parser.add_argument('--duration', type=float, default=20.0, help='seconds to collect')
    parser.add_argument('--pose', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=('X', 'Y', 'THETA'), help='true robot pose (m, m, rad)')
    parser.add_argument('--output', default='static_check')
    args = parser.parse_args()
    pose = tuple(args.pose)

    import yaml
    with open(args.params) as f:
        p = yaml.safe_load(f)['velocity_control_node']['ros__parameters']
    camera_pose = tuple(float(p.get(k, 0.0)) for k in ('camera_x', 'camera_y', 'camera_yaw'))
    landmarks = load_landmarks(args.landmarks)

    rclpy.init()
    node = Collector(landmarks, camera_pose, p.get('camera_frame', 'camera_frame'),
                     bool(p.get('camera_upside_down', False)))
    end = time.monotonic() + args.duration
    print(f'Collecting for {args.duration:.0f} s; do not move the robot')
    while time.monotonic() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

    if not node.samples:
        raise SystemExit('no tag sightings: are camera_node and apriltag_node running, and is a '
                         'landmark tag in view?')
    data = np.array(node.samples)
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'samples.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'tag', 'range', 'bearing', 'orientation', 'pose_x', 'pose_y',
                    'pose_theta', 'cam_x', 'cam_y', 'cam_z'])
        w.writerows(data)

    print(f'\nExpected at ({pose[0]:g}, {pose[1]:g}, {pose[2]:g}) from landmarks.txt, with the '
          f'camera {camera_pose[0]:g} m ahead:')
    seen = sorted({int(t) for t in data[:, 1]})
    all_errors = []
    summary = []
    for tag in sorted(landmarks):
        er, eb, eo = expected_measurement(pose, landmarks[tag])
        rows = data[data[:, 1] == tag][:, 1:]
        if not len(rows):
            print(f'  tag {tag}: expected range {er:.3f} m, bearing {math.degrees(eb):+.1f} deg: '
                  f'NOT SEEN')
            continue
        errors = summarize(rows, landmarks, pose)
        all_errors.append(errors)
        summary.append([tag, len(rows), er, rows[:, 1].mean(), *errors.mean(axis=0),
                        *errors.std(axis=0)])
        print(f'  tag {tag}: expected range {er:.3f} m, bearing {math.degrees(eb):+.1f} deg, '
              f'orient {math.degrees(eo):+.1f} deg; {len(rows)} samples '
              f'({len(rows) / args.duration:.1f}/s), measured range {rows[:, 1].mean():.3f} m')
        print(f'     error: {stats(errors)}')
    pooled = np.vstack(all_errors)
    print(f'\nPOOLED over {len(pooled)} samples, tags {seen} (measured - expected):')
    print(f'  {stats(pooled)}')
    with open(os.path.join(args.output, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pose_x', 'pose_y', 'pose_theta', 'tag', 'n', 'exp_range', 'range',
                    'err_range', 'err_bearing', 'err_orientation', 'err_pose', 'err_heading',
                    'std_range', 'std_bearing', 'std_orientation', 'std_pose', 'std_heading'])
        w.writerows([[*pose, *row] for row in summary])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    per_tag = [(tag, summarize(data[data[:, 1] == tag][:, 1:], landmarks, pose)) for tag in seen]
    axes[0].boxplot([e[:, 0] * 100 for _, e in per_tag], labels=[f'tag {t}' for t, _ in per_tag])
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].set_ylabel('range error [cm]')
    axes[1].boxplot([np.degrees(e[:, 2]) for _, e in per_tag],
                    labels=[f'tag {t}' for t, _ in per_tag])
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].set_ylabel('orientation error [deg]')
    for tag, e in per_tag:
        rows = data[data[:, 1] == tag]
        axes[2].scatter(rows[:, 5], rows[:, 6], s=8, label=f'tag {tag}')
    axes[2].scatter([pose[0]], [pose[1]], marker='*', s=200, color='k',
                    label=f'true ({pose[0]:g}, {pose[1]:g})')
    axes[2].set_xlabel('implied x [m]')
    axes[2].set_ylabel('implied y [m]')
    axes[2].set_aspect('equal')
    axes[2].legend(fontsize=8)
    axes[2].set_title('pose implied by each sighting')
    for ax in axes[:2]:
        ax.grid(True, alpha=0.3)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, 'static_check.png'), dpi=130)
    print(f'\nWrote {args.output}/samples.csv, summary.csv and static_check.png')


if __name__ == '__main__':
    main()
