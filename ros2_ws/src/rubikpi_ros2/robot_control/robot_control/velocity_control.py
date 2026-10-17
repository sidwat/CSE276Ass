import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

from robot_control.waypoint_controller import (
    body_to_wheel_cmds, Gains, load_waypoints, WaypointController, wheel_cmds_to_body, wrap_angle)


class VelocityControlNode(Node):
    """
    Follow the waypoints in a file, starting from pose (0, 0, 0).

    There are no wheel encoders, so position is dead-reckoned from the commanded wheel speeds
    (speed_per_unit and wheel_base must be calibrated). Heading is either dead-reckoned too or,
    with use_imu:=true, taken from the IMU yaw relative to its value at startup.
    """

    def __init__(self):
        super().__init__('velocity_control_node')

        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('speed_per_unit', 1.0)  # m/s of wheel speed per 1.0 of command
        self.declare_parameter('wheel_base', 0.2)      # m, effective track width
        self.declare_parameter('max_wheel_cmd', 0.5)   # motor_control clips to +/-0.5 anyway
        self.declare_parameter('wheel_trim', 0.0)      # >0 if it veers right going straight
        self.declare_parameter('control_rate', 20.0)   # Hz, must beat the 0.15s motor watchdog
        self.declare_parameter('start_delay', 2.0)     # s, wait for motor_control to come up
        self.declare_parameter('pause_at_waypoint', 1.0)  # s
        self.declare_parameter('use_imu', False)
        self.declare_parameter('imu_yaw_sign', 1.0)    # -1.0 if the IMU yaw is clockwise-positive
        defaults = Gains()
        for name, value in vars(defaults).items():
            self.declare_parameter(name, value)

        path = self.get_parameter('waypoints_file').value
        if not path:
            raise RuntimeError('waypoints_file parameter is required')
        waypoints = load_waypoints(path)
        gains = Gains(**{name: float(self.get_parameter(name).value) for name in vars(defaults)})
        self.controller = WaypointController(waypoints, gains)

        self.speed_per_unit = float(self.get_parameter('speed_per_unit').value)
        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.max_wheel_cmd = float(self.get_parameter('max_wheel_cmd').value)
        self.wheel_trim = float(self.get_parameter('wheel_trim').value)
        self.pause_at_waypoint = float(self.get_parameter('pause_at_waypoint').value)
        self.use_imu = bool(self.get_parameter('use_imu').value)
        self.imu_yaw_sign = float(self.get_parameter('imu_yaw_sign').value)

        # Pose estimate in the start frame, and the body velocity actually being commanded
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.w = 0.0

        self.imu_yaw = None
        self.imu_yaw_offset = None
        if self.use_imu:
            self.create_subscription(Imu, 'imu_data', self.imu_callback, 10)

        self.publisher = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

        now = time.monotonic()
        self.last_time = now
        self.hold_until = now + float(self.get_parameter('start_delay').value)
        self.finished = False
        self.timer = self.create_timer(
            1.0 / float(self.get_parameter('control_rate').value), self.control_loop)

        self.get_logger().info(f'Loaded {len(waypoints)} waypoints from {path}')
        for i, (wx, wy, wt) in enumerate(waypoints):
            self.get_logger().info(f'  [{i}] x={wx:.2f} y={wy:.2f} theta={wt:.2f}')

    def imu_callback(self, msg):
        q = msg.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.imu_yaw_offset is None:
            self.imu_yaw_offset = yaw
        self.imu_yaw = yaw

    def control_loop(self):
        now = time.monotonic()
        dt = now - self.last_time
        self.last_time = now

        self.integrate_pose(dt)

        if now < self.hold_until or self.finished:
            self.send(0.0, 0.0)
            return
        if self.use_imu and self.imu_yaw is None:
            self.get_logger().info('Waiting for IMU data...', throttle_duration_sec=2.0)
            self.send(0.0, 0.0)
            return

        v, w, reached = self.controller.update(self.x, self.y, self.theta)
        if reached is not None:
            wx, wy, wt = self.controller.waypoints[reached]
            self.get_logger().info(
                f'Reached waypoint [{reached}] ({wx:.2f}, {wy:.2f}, {wt:.2f}); estimated pose '
                f'({self.x:.2f}, {self.y:.2f}, {self.theta:.2f})')
            self.hold_until = now + self.pause_at_waypoint
        if self.controller.done:
            self.get_logger().info('All waypoints reached')
            self.finished = True
        self.send(v, w)

    def integrate_pose(self, dt):
        if self.use_imu and self.imu_yaw is not None:
            self.theta = wrap_angle(self.imu_yaw_sign * (self.imu_yaw - self.imu_yaw_offset))
            heading = self.theta
        else:
            heading = self.theta + self.w * dt / 2.0  # midpoint heading over the step
            self.theta = wrap_angle(self.theta + self.w * dt)
        self.x += self.v * math.cos(heading) * dt
        self.y += self.v * math.sin(heading) * dt

    def send(self, v, w):
        left, right = body_to_wheel_cmds(
            v, w, self.speed_per_unit, self.wheel_base, self.max_wheel_cmd)
        # Dead-reckon from what was actually sent, after saturation
        self.v, self.w = wheel_cmds_to_body(left, right, self.speed_per_unit, self.wheel_base)
        # The trim compensates for mismatched motors so the robot actually does what was
        # commanded, so it is applied after (and not included in) the dead reckoning above.
        msg = Float32MultiArray()
        msg.data = [float(left * (1.0 - self.wheel_trim)), float(right * (1.0 + self.wheel_trim))]
        self.publisher.publish(msg)

    def destroy_node(self):
        try:
            self.send(0.0, 0.0)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VelocityControlNode()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    main()
