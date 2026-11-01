"""Visual pose -> EKF -> existing waypoint controller; motion disabled by default."""
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as RosPath
from std_msgs.msg import Float32MultiArray, String
from robot_control.waypoint_controller import (
    Gains, WaypointController, body_to_wheel_cmds, wheel_cmds_to_body)
from hw2_navigation.core import PoseFilter, transform
from hw2_navigation.configuration import load_config
from hw2_navigation.visual_localization import VisualLocalizer
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time


class Navigator(Node):
    def __init__(self):
        super().__init__('hw2_navigator')
        self.declare_parameter('config', '')
        self.declare_parameter('detector_config', '')
        self.declare_parameter('enable_motion', False)
        self.declare_parameter('csv_path', '/tmp/hw2-live.csv')
        self.enabled = self.get_parameter('enable_motion').value
        self.cfg, self.tags, self.mount_ready = load_config(
            self.get_parameter('config').value, self.enabled,
            self.get_parameter('detector_config').value)
        self.filter = PoseFilter(self.cfg['filter']['process_std'])
        g = Gains(max_v=self.cfg['control']['max_v'], max_w=self.cfg['control']['max_w'],
                  min_v=0.03, min_w=0.15)
        self.controller = WaypointController(self.cfg['waypoints'], g)
        self.base_camera = (transform(**self.cfg['base_camera']) if self.mount_ready else None)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.localizer = VisualLocalizer(self.cfg, self.tags, self.base_camera)
        self.last_seen = None
        self.last_tick = time.monotonic()
        self.hold_until = 0.0
        self.v = self.w = 0.0
        self.accepted = self.rejected = 0
        self.last_reason = 'waiting_for_measurements'
        self.path = RosPath()
        self.path.header.frame_id = 'map'
        self.motor = (self.create_publisher(Float32MultiArray, '/motor_commands', 1)
                      if self.enabled else None)
        self.command = self.create_publisher(Twist, '/hw2/proposed_cmd_vel', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/hw2/pose', 10)
        self.path_pub = self.create_publisher(RosPath, '/hw2/path', 1)
        self.debug = self.create_publisher(String, '/hw2/status', 10)
        self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        path = Path(self.get_parameter('csv_path').value)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids silently replacing previous experimental data.
        self.log = path.open('x', newline='')
        self.writer = csv.writer(self.log)
        self.writer.writerow(['t', 'x', 'y', 'yaw', 'sigma_x', 'sigma_y', 'sigma_yaw',
                              'v', 'w', 'waypoint', 'reached', 'visual_age', 'accepted', 'rejected', 'state'])
        self.tick_count = 0
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f'Motion enabled: {self.enabled}; surveyed tags: {len(self.tags)}')

    def on_detections(self, msg):
        self.last_reason = self.localizer.enqueue(msg, self.get_clock().now().nanoseconds)

    def lookup_tag_transform(self, camera_frame, tag_frame, stamp):
        """Nonblocking lookup at capture time; missing TF is retried next tick."""
        try:
            return self.tf_buffer.lookup_transform(
                camera_frame, tag_frame, Time.from_msg(stamp))
        except TransformException:
            return None

    def update_visual_pose(self, now):
        result = self.localizer.process(
            self.get_clock().now().nanoseconds,
            self.lookup_tag_transform, self.filter)
        self.last_reason = result.reason
        self.rejected += result.rejected
        if result.accepted:
            # Account for capture/transport delay in the blind-stop interval.
            self.last_seen = now-result.age
            self.accepted += 1

    def tick(self):
        now = time.monotonic()
        dt, self.last_tick = now-self.last_tick, now
        healthy_dt = self.filter.predict(self.v, self.w, dt)
        self.update_visual_pose(now)
        age = now-self.last_seen if self.last_seen is not None else float('inf')
        state = 'tracking'
        reached = None
        v = w = 0.0
        if not healthy_dt:
            state = 'timer_gap_stop'
        elif age > self.cfg['control']['blind_timeout']:
            state = 'landmark_loss_stop'
        elif now < self.hold_until:
            state = 'waypoint_pause'
        else:
            v, w, reached = self.controller.update(*self.filter.x)
            if reached is not None:
                self.hold_until = now+self.cfg['control']['pause_at_waypoint']
            if self.controller.done:
                state = 'complete'
        m = self.cfg['motion']
        left, right = body_to_wheel_cmds(v, w, m['speed_per_unit'], m['wheel_base'], m['max_cmd'])
        left *= 1-m['wheel_trim']
        right *= 1+m['wheel_trim']
        scale = max(1.0, abs(left)/m['max_cmd'], abs(right)/m['max_cmd'])
        left, right = left/scale, right/scale
        # Prediction treats trim as actuator compensation, as in the HW1 controller.
        actual_v, actual_w = wheel_cmds_to_body(left/(1-m['wheel_trim']), right/(1+m['wheel_trim']),
                                               m['speed_per_unit'], m['wheel_base'])
        self.v, self.w = (actual_v, actual_w) if self.enabled else (0.0, 0.0)
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = actual_v, actual_w
        self.command.publish(cmd)
        if self.motor is not None:
            motors = Float32MultiArray()
            motors.data = [float(left), float(right)]
            self.motor.publish(motors)
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = map(float, self.filter.x[:2])
        pose.pose.orientation.z = math.sin(self.filter.x[2]/2)
        pose.pose.orientation.w = math.cos(self.filter.x[2]/2)
        self.pose_pub.publish(pose)
        if self.tick_count % 10 == 0:  # assignment: a dot every 0.5 s
            self.path.header.stamp = pose.header.stamp
            self.path.poses.append(pose)
            self.path.poses = self.path.poses[-2000:]
            self.path_pub.publish(self.path)
        self.tick_count += 1
        status = String()
        status.data = json.dumps(dict(state=state, phase=self.controller.phase,
            waypoint=self.controller.index, visual_age=age if math.isfinite(age) else None,
            accepted=self.accepted, rejected=self.rejected, reason=self.last_reason, enabled=self.enabled))
        self.debug.publish(status)
        self.writer.writerow([self.get_clock().now().nanoseconds*1e-9, *self.filter.x,
                              *np.sqrt(np.diag(self.filter.P)), actual_v, actual_w,
                              self.controller.index, reached, age, self.accepted, self.rejected, state])
        self.log.flush()

    def destroy_node(self):
        if self.motor is not None:
            stop = Float32MultiArray()
            stop.data = [0.0, 0.0]
            self.motor.publish(stop)
        self.log.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Navigator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
