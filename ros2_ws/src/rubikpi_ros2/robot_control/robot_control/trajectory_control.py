from dataclasses import fields
import math
import time

import rclpy
from rclpy.node import Node
from robot_control.trajectory_planner import plan_trajectory, PlannerConfig
from robot_control.waypoint_controller import body_to_wheel_cmds, load_waypoints
from std_msgs.msg import Float32MultiArray


class TrajectoryControlNode(Node):
    """
    Drive a smooth constant-speed trajectory through the waypoint poses, open loop.

    The whole path is planned at startup and then played back in time: each cycle sends the
    planned body velocity for the current time. Without encoders or IMU there is nothing to
    correct against, so accuracy depends entirely on the speed_per_unit/wheel_base calibration.
    """

    def __init__(self):
        super().__init__('trajectory_control_node')

        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('control_rate', 20.0)  # Hz, must beat the 0.15s motor watchdog
        self.declare_parameter('start_delay', 2.0)    # s, wait for motor_control to come up
        for f in fields(PlannerConfig):
            self.declare_parameter(f.name, f.default)

        path = self.get_parameter('waypoints_file').value
        if not path:
            raise RuntimeError('waypoints_file parameter is required')
        waypoints = load_waypoints(path)
        self.config = PlannerConfig(
            **{f.name: f.type(self.get_parameter(f.name).value) for f in fields(PlannerConfig)})

        self.get_logger().info(f'Planning through {len(waypoints)} waypoints from {path}...')
        started = time.monotonic()
        self.trajectory = plan_trajectory(waypoints, self.config)
        self.log_plan(waypoints, time.monotonic() - started)

        self.publisher = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.start_time = time.monotonic() + float(self.get_parameter('start_delay').value)
        self.next_waypoint = 0
        self.finished = False
        self.timer = self.create_timer(
            1.0 / float(self.get_parameter('control_rate').value), self.control_loop)

    def log_plan(self, waypoints, planning_time):
        traj, report = self.trajectory, self.trajectory.report
        log = self.get_logger()
        log.info(f'Plan ready in {planning_time:.1f}s: {traj.duration:.1f}s long, max curvature '
                 f'{report.max_curvature:.2f} (limit {report.curvature_limit:.2f}) 1/m')
        for i, ((wx, wy, wt), t) in enumerate(zip(waypoints, traj.waypoint_times)):
            heading = traj.sample(t).heading
            log.info(f'  [{i}] ({wx:.2f}, {wy:.2f}) at t={t:5.1f}s, heading '
                     f'{math.degrees(heading):6.1f} deg (file {math.degrees(wt):6.1f} deg)')
        log.info('  segments: ' + ', '.join(
            ('straight' if s else 'curve') + (' fwd' if d > 0 else ' rev')
            for d, s in zip(report.directions, report.straight)))
        if report.stops:
            log.info(f'  touches zero speed to change direction at waypoints {report.stops}')
        log.info('  corridor deviation per segment [m]: '
                 + ', '.join(f'{d:.3f}' for d in report.max_deviation))
        for violation in report.violations:
            log.warning(f'Plan constraint not met: {violation}')

    def control_loop(self):
        t = time.monotonic() - self.start_time
        if t < 0.0 or self.finished:
            self.send(0.0, 0.0)
            return
        if t >= self.trajectory.duration:
            self.get_logger().info('Trajectory complete')
            self.finished = True
            self.send(0.0, 0.0)
            return

        sample = self.trajectory.sample(t)
        times = self.trajectory.waypoint_times
        if self.next_waypoint < len(times) and t >= times[self.next_waypoint]:
            self.get_logger().info(
                f'Passing waypoint [{self.next_waypoint}] (planned pose {sample.x:.2f}, '
                f'{sample.y:.2f}, {math.degrees(sample.heading):.0f} deg)')
            self.next_waypoint += 1
        self.send(sample.v, sample.w)

    def send(self, v, w):
        cfg = self.config
        left, right = body_to_wheel_cmds(
            v, w, cfg.speed_per_unit, cfg.wheel_base, cfg.max_wheel_cmd)
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.publisher.publish(msg)

    def destroy_node(self):
        try:
            self.send(0.0, 0.0)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryControlNode()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    main()
