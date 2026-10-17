"""Drive one wheel command for an exact time, stop, and print the predicted motion."""

import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Float32MultiArray


def predicted_motion(left, right, duration, speed_per_unit, wheel_base):
    """Return (rotation [rad, CCW +], forward [m], left [m]) for constant wheel commands."""
    v = (left + right) / 2.0 * speed_per_unit
    w = (right - left) * speed_per_unit / wheel_base
    theta = w * duration
    if abs(w) < 1e-9:
        return 0.0, v * duration, 0.0
    radius = v / w
    return theta, radius * math.sin(theta), radius * (1.0 - math.cos(theta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=float, required=True, help='nominal left command')
    parser.add_argument('--right', type=float, required=True, help='nominal right command')
    parser.add_argument('--duration', type=float, required=True, help='seconds to drive')
    parser.add_argument('--trim', type=float, default=0.0, help='wheel_trim, as in the YAML')
    parser.add_argument('--speed-per-unit', type=float, default=1.4)
    parser.add_argument('--wheel-base', type=float, default=0.8)
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = Node('calibrate_motion')
    publisher = node.create_publisher(Float32MultiArray, 'motor_commands', 10)

    def send(left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        publisher.publish(msg)

    deadline = time.monotonic() + 10.0
    while publisher.get_subscription_count() == 0:
        if time.monotonic() > deadline:
            print('motor_control is not subscribed to motor_commands - is it running?')
            return 1
        time.sleep(0.1)
    time.sleep(0.5)

    sent_left = args.left * (1.0 - args.trim)
    sent_right = args.right * (1.0 + args.trim)
    print(f'Driving L={sent_left:.3f} R={sent_right:.3f} (nominal {args.left:.3f}, '
          f'{args.right:.3f}, trim {args.trim:+.3f}) for {args.duration:.2f} s')
    try:
        start = time.monotonic()
        while (elapsed := time.monotonic() - start) < args.duration:
            send(sent_left, sent_right)
            time.sleep(min(0.05, max(0.0, args.duration - elapsed)))
        elapsed = time.monotonic() - start
    finally:
        for _ in range(5):
            send(0.0, 0.0)
            time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()

    theta, forward, left = predicted_motion(
        args.left, args.right, elapsed, args.speed_per_unit, args.wheel_base)
    print(f'Drove for {elapsed:.3f} s. With speed_per_unit={args.speed_per_unit} and '
          f'wheel_base={args.wheel_base} the robot should have:')
    print(f'  rotated  {math.degrees(theta):8.1f} deg (+ = counter-clockwise)')
    print(f'  moved    {forward * 100:8.1f} cm forward, {left * 100:.1f} cm left (robot centre)')
    print('Measure the actual rotation and where the robot centre ended up.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
