from dataclasses import fields
import math
import time

import rclpy
from rclpy.node import Node
from robot_control.segment_planner import plan_segments, SegmentConfig
from robot_control.trajectory_planner import plan_trajectory, PlannerConfig
from robot_control.waypoint_controller import body_to_wheel_cmds, load_waypoints
from std_msgs.msg import Float32MultiArray


class TrajectoryControlNode(Node):
    """
    Drive through the waypoint poses open loop, in one of two modes.

    quintic:=false (default) plays a turn-drive-turn plan of fixed, calibrated motions
    (segment_planner). quintic:=true plans a smooth constant-speed path (trajectory_planner) and
    plays back its body velocity. Without encoders or IMU there is nothing to correct against,
    so accuracy depends entirely on the calibration.
    """

    def __init__(self):
        super().__init__('trajectory_control_node')

        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('quintic', False)
        self.declare_parameter('control_rate', 50.0)  # Hz, must beat the 0.15s motor watchdog
        self.declare_parameter('start_delay', 2.0)    # s, wait for motor_control to come up
        for cls in (PlannerConfig, SegmentConfig):
            for f in fields(cls):
                if not self.has_parameter(f.name):
                    self.declare_parameter(f.name, f.default)

        path = self.get_parameter('waypoints_file').value
        if not path:
            raise RuntimeError('waypoints_file parameter is required')
        waypoints = load_waypoints(path)
        self.quintic = bool(self.get_parameter('quintic').value)

        if self.quintic:
            self.config = self.read_config(PlannerConfig)
            self.get_logger().info(f'QUINTIC mode: planning through {len(waypoints)} waypoints...')
            started = time.monotonic()
            self.trajectory = plan_trajectory(waypoints, self.config)
            self.duration = self.trajectory.duration
            self.log_quintic_plan(waypoints, time.monotonic() - started)
        else:
            self.config = self.read_config(SegmentConfig)
            self.motions, notes = plan_segments(waypoints, self.config)
            self.duration = self.motions[-1].start_time + self.motions[-1].duration
            self.log_segment_plan(waypoints, notes)

        self.publisher = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.start_time = time.monotonic() + float(self.get_parameter('start_delay').value)
        self.next_waypoint = 0
        self.current_motion = -1
        self.finished = False
        self.timer = self.create_timer(
            1.0 / float(self.get_parameter('control_rate').value), self.control_loop)

    def read_config(self, cls):
        return cls(**{f.name: f.type(self.get_parameter(f.name).value) for f in fields(cls)})

    def log_segment_plan(self, waypoints, notes):
        cfg, log = self.config, self.get_logger()
        log.info(f'TURN-DRIVE-TURN mode: {len(waypoints)} waypoints, {self.duration:.1f}s total; '
                 f'drive_cmd {cfg.drive_cmd}, spin_cmd {cfg.spin_cmd}, CCW {cfg.ccw_deg_per_s} '
                 f'deg/s +{cfg.ccw_offset_deg}, CW {cfg.cw_deg_per_s} deg/s +{cfg.cw_offset_deg}')
        for m, after in zip(self.motions, self.motions[1:] + [None]):
            if m.label != 'pause':
                reached = after.waypoint if after is not None else -1
                end = f'  -> waypoint [{reached}]' if reached >= 0 else ''
                log.info(f'  t={m.start_time:5.1f}s {m.label:16s} {m.duration:5.2f}s '
                         f'L={m.left:+.3f} R={m.right:+.3f}{end}')
        for note in notes:
            log.warning(note)

    def log_quintic_plan(self, waypoints, planning_time):
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
            self.publish(0.0, 0.0)
            return
        if t >= self.duration:
            self.get_logger().info('Trajectory complete')
            self.finished = True
            self.publish(0.0, 0.0)
            return
        if self.quintic:
            self.quintic_step(t)
        else:
            self.segment_step(t)

    def segment_step(self, t):
        index = self.current_motion
        while index + 1 < len(self.motions) and t >= self.motions[index + 1].start_time:
            index += 1
            motion = self.motions[index]
            if motion.label != 'pause':
                self.get_logger().info(f't={t:5.1f}s {motion.label}')
            elif motion.waypoint >= 0:
                self.get_logger().info(f't={t:5.1f}s Reached waypoint [{motion.waypoint}]')
        self.current_motion = index
        motion = self.motions[index]
        self.publish(motion.left, motion.right)

    def quintic_step(self, t):
        sample = self.trajectory.sample(t)
        times = self.trajectory.waypoint_times
        if self.next_waypoint < len(times) and t >= times[self.next_waypoint]:
            self.get_logger().info(
                f'Passing waypoint [{self.next_waypoint}] (planned pose {sample.x:.2f}, '
                f'{sample.y:.2f}, {math.degrees(sample.heading):.0f} deg)')
            self.next_waypoint += 1
        cfg = self.config
        left, right = body_to_wheel_cmds(
            sample.v, sample.w, cfg.speed_per_unit, cfg.wheel_base, math.inf)
        left *= 1.0 - cfg.wheel_trim
        right *= 1.0 + cfg.wheel_trim
        # Saturate after the trim, scaling both wheels so motor_control never clips one alone
        largest = max(abs(left), abs(right))
        if largest > cfg.max_wheel_cmd:
            left *= cfg.max_wheel_cmd / largest
            right *= cfg.max_wheel_cmd / largest
        self.publish(left, right)

    def publish(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.publisher.publish(msg)

    def destroy_node(self):
        try:
            self.publish(0.0, 0.0)
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
