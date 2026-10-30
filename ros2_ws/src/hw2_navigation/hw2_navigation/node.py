"""Visual pose -> EKF -> existing waypoint controller; motion disabled by default."""
import csv
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as RosPath
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String
from robot_control.waypoint_controller import (
    Gains, WaypointController, body_to_wheel_cmds, wheel_cmds_to_body)
from hw2_navigation.core import PoseFilter, base_pose, transform


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec*1e-9


def load_config(path, enable):
    with open(path) as stream:
        cfg = yaml.safe_load(stream)
    tags = [t for t in cfg['tags'] if all(t[k] is not None for k in ('id', 'size', 'xyz', 'rpy'))]
    if len({t['id'] for t in tags}) != len(tags):
        raise ValueError('Duplicate tag IDs')
    for t in tags:
        if t['size'] <= 0 or not np.isfinite(t['xyz'] + t['rpy']).all():
            raise ValueError('Invalid tag geometry')
    m = cfg['motion']
    if not (m['speed_per_unit'] > 0 and m['wheel_base'] > 0 and
            0 < m['max_cmd'] <= 0.5 and abs(m['wheel_trim']) < 1):
        raise ValueError('Invalid motion calibration')
    mount = cfg['base_camera']
    complete = mount['xyz'] is not None and mount['rpy'] is not None
    if enable and (not cfg['reviewed'] or not complete or len(tags) != 8):
        raise ValueError('Motion requires reviewed=true, measured camera mount and all eight tags')
    return cfg, {int(t['id']): t for t in tags}, complete


class Navigator(Node):
    def __init__(self):
        super().__init__('hw2_navigator')
        self.declare_parameter('config', '')
        self.declare_parameter('enable_motion', False)
        self.declare_parameter('csv_path', '/tmp/hw2-live.csv')
        self.enabled = self.get_parameter('enable_motion').value
        self.cfg, self.tags, self.mount_ready = load_config(self.get_parameter('config').value, self.enabled)
        self.filter = PoseFilter(self.cfg['filter']['process_std'])
        g = Gains(max_v=self.cfg['control']['max_v'], max_w=self.cfg['control']['max_w'],
                  min_v=0.03, min_w=0.15)
        self.controller = WaypointController(self.cfg['waypoints'], g)
        self.base_camera = (transform(**self.cfg['base_camera']) if self.mount_ready else None)
        self.info = None
        self.corners = []
        self.bridge = CvBridge()
        self.last_seen = None
        self.last_stamp = -float('inf')
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
        self.overlay = self.create_publisher(Image, '/hw2/overlay', 1)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, qos_profile_sensor_data)
        self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        self.create_subscription(Image, '/camera/image_rect', self.on_image, qos_profile_sensor_data)
        path = Path(self.get_parameter('csv_path').value)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids silently replacing previous experimental data.
        self.log = path.open('x', newline='')
        self.writer = csv.writer(self.log)
        self.writer.writerow(['t', 'x', 'y', 'yaw', 'sigma_x', 'sigma_y', 'sigma_yaw',
                              'v', 'w', 'waypoint', 'reached', 'visual_age', 'accepted', 'rejected', 'state'])
        self.tick_count = 0
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f'Motion enabled: {self.enabled}; measured tags: {len(self.tags)}/8')

    def on_info(self, msg):
        self.info = msg

    def on_detections(self, msg):
        self.corners = [(d.id, [(p.x, p.y) for p in d.corners]) for d in msg.detections]
        self.overlay_stamp = stamp_seconds(msg.header.stamp)
        age = self.get_clock().now().nanoseconds*1e-9 - self.overlay_stamp
        if (age < -0.02 or age > self.cfg['vision']['max_age'] or
                self.overlay_stamp <= self.last_stamp):
            self.last_reason = 'stale_or_out_of_order_image'
            return
        self.last_stamp = self.overlay_stamp
        if self.info is None or self.base_camera is None:
            self.last_reason = 'missing_camera_info_or_mount'
            return
        if (msg.header.frame_id != self.cfg['camera_frame'] or
                self.info.header.frame_id != msg.header.frame_id or
                [self.info.width, self.info.height] != self.cfg['calibration_resolution']):
            self.last_reason = 'camera_frame_or_resolution_mismatch'
            return
        k = np.asarray(self.info.p).reshape(3, 4)[:, :3].copy()
        if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            self.last_reason = 'invalid_intrinsics'
            return
        candidates = []
        for d in msg.detections:
            tag = self.tags.get(d.id)
            if tag is None or d.hamming != 0 or d.decision_margin < self.cfg['vision']['min_margin']:
                continue
            s = tag['size']/2
            obj = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=float)
            pixels = np.array([[p.x, p.y] for p in d.corners], dtype=float)
            ok, r, t = cv2.solvePnP(obj, pixels, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or t[2, 0] <= 0 or np.linalg.norm(t) > self.cfg['vision']['max_range']:
                continue
            projection, _ = cv2.projectPoints(obj, r, t, k, None)
            error = float(np.sqrt(np.mean(np.sum((projection.reshape(-1, 2)-pixels)**2, axis=1))))
            if error > self.cfg['vision']['max_reprojection_px']:
                continue
            ct = np.eye(4)
            ct[:3, :3], ct[:3, 3] = cv2.Rodrigues(r)[0], t.ravel()
            z = base_pose(transform(tag['xyz'], tag['rpy']), ct, self.base_camera)
            # Conservative range-dependent heuristic; replace with measured static covariance.
            std = np.asarray(self.cfg['filter']['visual_std'])*max(1.0, float(np.linalg.norm(t)))
            candidates.append((error, z, np.diag(std**2)))
        # One update per image avoids claiming independence for correlated same-image tags.
        # Try the lowest reprojection error first; retain EKF innovation rejection.
        for _, z, covariance in sorted(candidates, key=lambda item: item[0]):
            ok, nis = self.filter.correct(z, covariance, self.cfg['filter']['nis_gate'])
            if ok:
                self.last_seen = time.monotonic()
                self.accepted += 1
                self.last_reason = f'visual_accepted_nis={nis:.2f}'
                return
            self.rejected += 1
        self.last_reason = 'no_valid_visual_update'

    def on_image(self, msg):
        if self.overlay.get_subscription_count() == 0:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8').copy()
        if abs(stamp_seconds(msg.header.stamp)-getattr(self, 'overlay_stamp', -1)) < 0.1:
            for tag_id, points in self.corners:
                pts = np.asarray(points, dtype=np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                cv2.putText(img, str(tag_id), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
        cv2.putText(img, self.last_reason, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 200, 255), 1)
        result = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        result.header = msg.header
        self.overlay.publish(result)

    def tick(self):
        now = time.monotonic()
        dt, self.last_tick = now-self.last_tick, now
        healthy_dt = self.filter.predict(self.v, self.w, dt)
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
