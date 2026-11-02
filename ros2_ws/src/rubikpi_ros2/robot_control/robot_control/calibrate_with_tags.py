"""
Motion calibration against one AprilTag: measure, drive a fixed wheel command, stop, measure.

The motion between the two stops is recovered from the tag alone (its range, bearing and
orientation in the robot frame, averaged while standing still), so no tape measure is needed:

  straight  L = R = cmd        forward/left distance and heading drift -> speed_per_unit, trim
  spin      L = -cmd, R = cmd  rotation from the tag bearing          -> wheel_base
            (cmd > 0 spins counter-clockwise)

The trim is applied exactly as velocity_control applies it. Every run is appended to --log, so
the parameters can be fitted over many runs. Needs camera_node, apriltag_node and motor_control.
"""

import argparse
import csv
import math
import os
import re
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from robot_control.localization import config_dir, tag_measurement
from robot_control.waypoint_controller import wrap_angle
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage

CONFIG = config_dir()
FIELDS = ['time', 'mode', 'cmd', 'duration', 'elapsed', 'trim', 'speed_per_unit', 'wheel_base',
          'n0', 'range0', 'bearing0', 'orient0', 'n1', 'range1', 'bearing1', 'orient1',
          'forward', 'left', 'rot_orient', 'rot_bearing',
          'est_speed_per_unit', 'est_trim', 'est_wheel_base', 'note']


class Calibrator(Node):
    def __init__(self, tag, camera_pose, camera_frame, upside_down):
        super().__init__('calibrate_with_tags')
        self.tag, self.camera_pose = tag, camera_pose
        self.camera_frame, self.upside_down = camera_frame, upside_down
        self.samples = None  # a list only while collecting
        self.publisher = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.create_subscription(TFMessage, '/tf', self.on_tf, 100)

    def on_tf(self, msg):
        if self.samples is None:
            return
        for tf in msg.transforms:
            match = re.search(r'[_:](\d+)$', tf.child_frame_id)
            if (tf.header.frame_id != self.camera_frame or not match
                    or int(match.group(1)) != self.tag):
                continue
            t, q = tf.transform.translation, tf.transform.rotation
            normal = (2.0 * (q.x * q.z + q.w * q.y), 2.0 * (q.y * q.z - q.w * q.x),
                      1.0 - 2.0 * (q.x * q.x + q.y * q.y))
            self.samples.append(tag_measurement(
                (t.x, t.y, t.z), normal, self.camera_pose, self.upside_down))

    def collect(self, seconds):
        """Spin for seconds and return the tag sightings (also drains any queued during a move)."""
        self.samples = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        samples, self.samples = self.samples, None
        return np.array(samples)

    def send(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.publisher.publish(msg)

    def drive(self, left, right, duration):
        try:
            start = time.monotonic()
            while (elapsed := time.monotonic() - start) < duration:
                self.send(left, right)
                time.sleep(min(0.05, max(0.0, duration - elapsed)))
            return time.monotonic() - start
        finally:
            for _ in range(5):
                self.send(0.0, 0.0)
                time.sleep(0.05)


def mean_angle(a):
    return math.atan2(np.sin(a).mean(), np.cos(a).mean())


def summary(samples):
    """Return n, mean range, bearing and orientation, and their spreads (angles on the circle)."""
    r, b, o = samples[:, 0], mean_angle(samples[:, 1]), mean_angle(samples[:, 2])
    spread = (r.std(), np.std([wrap_angle(v - b) for v in samples[:, 1]]),
              np.std([wrap_angle(v - o) for v in samples[:, 2]]))
    return len(samples), r.mean(), b, o, spread


def relative_motion(before, after, rotation):
    """(forward, left) of the robot at the second stop, in the frame of the first stop."""
    _, r0, b0, _, _ = before
    _, r1, b1, _, _ = after
    c, s = math.cos(rotation), math.sin(rotation)
    p1x, p1y = r1 * math.cos(b1), r1 * math.sin(b1)
    # the tag is fixed: p0 = t + R(rotation) p1
    return (r0 * math.cos(b0) - (c * p1x - s * p1y),
            r0 * math.sin(b0) - (s * p1x + c * p1y))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['straight', 'spin'])
    parser.add_argument('--cmd', type=float, required=True,
                        help='wheel command (spin: > 0 counter-clockwise)')
    parser.add_argument('--duration', type=float, required=True, help='seconds to drive')
    parser.add_argument('--tag', type=int, default=0, help='tag id to measure against')
    parser.add_argument('--average', type=float, default=5.0, help='seconds of tag data per stop')
    parser.add_argument('--settle', type=float, default=1.5, help='seconds to wait after stopping')
    parser.add_argument('--params', default=str(CONFIG / 'hw2.yaml'))
    parser.add_argument('--speed-per-unit', type=float, help='override the value in --params')
    parser.add_argument('--wheel-base', type=float, help='override the value in --params')
    parser.add_argument('--trim', type=float, help='override wheel_trim in --params')
    parser.add_argument('--log', default='~/hw2_calib/calibration.csv')
    parser.add_argument('--note', default='', help='free text stored with the run')
    parser.add_argument('--force', action='store_true', help='skip the range/view safety checks')
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    import yaml
    with open(args.params) as f:
        p = yaml.safe_load(f)['velocity_control_node']['ros__parameters']
    spu = args.speed_per_unit or float(p['speed_per_unit'])
    base = args.wheel_base or float(p['wheel_base'])
    trim = float(p.get('wheel_trim', 0.0)) if args.trim is None else args.trim
    camera_pose = tuple(float(p.get(k, 0.0)) for k in ('camera_x', 'camera_y', 'camera_yaw'))
    c = args.cmd
    left, right = (c, c) if args.mode == 'straight' else (-c, c)
    pred_forward = (left + right) / 2 * spu * args.duration
    pred_rotation = (right - left) * spu / base * args.duration

    rclpy.init()
    node = Calibrator(args.tag, camera_pose, p.get('camera_frame', 'camera_frame'),
                      bool(p.get('camera_upside_down', False)))
    try:
        deadline = time.monotonic() + 10.0
        while node.publisher.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                raise SystemExit('motor_control is not subscribed to motor_commands: start it')
            rclpy.spin_once(node, timeout_sec=0.1)

        print(f'Measuring tag {args.tag} for {args.average:.0f} s, robot must be still...')
        samples = node.collect(args.average)
        if len(samples) < 5:
            raise SystemExit(f'only {len(samples)} sightings of tag {args.tag}: is it in view and '
                             'are camera_node and apriltag_node running?')
        before = summary(samples)
        _, r0, b0, _, _ = before
        if args.mode == 'straight' and r0 * math.cos(b0) - pred_forward < 0.6 and not args.force:
            raise SystemExit(f'would end {r0 * math.cos(b0) - pred_forward:.2f} m from the tag '
                             '(< 0.6 m): back up or shorten --duration')
        if (args.mode == 'spin' and abs(wrap_angle(b0 - pred_rotation)) > math.radians(45)
                and not args.force):
            raise SystemExit(f'tag would end at bearing {math.degrees(b0 - pred_rotation):+.0f} '
                             'deg, out of view: start with the tag on the other side, or shorten '
                             '--duration')

        sent_left, sent_right = left * (1.0 - trim), right * (1.0 + trim)
        print(f'Driving L={sent_left:+.3f} R={sent_right:+.3f} for {args.duration:.2f} s '
              f'(model: {pred_forward * 100:.1f} cm, {math.degrees(pred_rotation):+.1f} deg)')
        elapsed = node.drive(sent_left, sent_right, args.duration)
        node.collect(args.settle)  # discard: sightings from while the robot was moving
        print(f'Measuring tag {args.tag} again for {args.average:.0f} s...')
        samples = node.collect(args.average)
        if len(samples) < 5:
            raise SystemExit(f'only {len(samples)} sightings after the move: the tag left the '
                             'view, rerun with a shorter --duration')
        after = summary(samples)
    finally:
        node.send(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

    # tag orientation in the robot frame is lyaw - theta
    rot_orient = wrap_angle(before[3] - after[3])
    rot_bearing = wrap_angle(before[2] - after[2])  # valid only if the centre did not move (spin)
    rotation = rot_bearing if args.mode == 'spin' else rot_orient
    forward, side = relative_motion(before, after, rotation)
    est_spu = est_trim = est_base = ''
    if args.mode == 'straight':
        est_spu = forward / (c * elapsed)
        est_trim = trim - rotation / elapsed * base / (2.0 * c * spu)
    elif abs(rotation) > 1e-3:
        est_base = 2.0 * c * spu * elapsed / rotation

    for name, s in (('before', before), ('after', after)):
        n, r, b, o, (sr, sb, so) = s
        print(f'  {name:6s} n={n:3d}  range {r:.4f} +- {sr * 100:.2f} cm  bearing '
              f'{math.degrees(b):+7.2f} +- {math.degrees(sb):.2f} deg  orientation '
              f'{math.degrees(o):+7.2f} +- {math.degrees(so):.2f} deg')
    print(f'\nDrove {elapsed:.3f} s. Measured: forward {forward * 100:+.1f} cm, left '
          f'{side * 100:+.1f} cm, rotation {math.degrees(rotation):+.2f} deg '
          f'(orientation {math.degrees(rot_orient):+.2f}, '
          f'bearing {math.degrees(rot_bearing):+.2f})')
    print(f'Model said:   forward {pred_forward * 100:+.1f} cm, rotation '
          f'{math.degrees(pred_rotation):+.2f} deg')
    if args.mode == 'straight':
        print(f'=> speed_per_unit {est_spu:.3f} (now {spu}), '
              f'wheel_trim {est_trim:+.4f} (now {trim})')
    elif est_base:
        print(f'=> wheel_base {est_base:.3f} (now {base}), using speed_per_unit {spu}')
        if est_base < 0:
            print('   the robot turned the wrong way: check the motor wiring / cmd sign')

    log = os.path.expanduser(args.log)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    new = not os.path.exists(log)
    with open(log, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerow([time.strftime('%H:%M:%S'), args.mode, c, args.duration, f'{elapsed:.3f}', trim,
                    spu, base, before[0], f'{before[1]:.4f}', f'{before[2]:.5f}',
                    f'{before[3]:.5f}', after[0], f'{after[1]:.4f}', f'{after[2]:.5f}',
                    f'{after[3]:.5f}', f'{forward:.4f}', f'{side:.4f}', f'{rot_orient:.5f}',
                    f'{rot_bearing:.5f}', est_spu and f'{est_spu:.4f}',
                    est_trim if est_trim == '' else f'{est_trim:.4f}',
                    est_base and f'{est_base:.4f}', args.note])
    print(f'Appended to {log}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
