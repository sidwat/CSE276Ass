import csv
import math
import os
import re
import shutil
import time

from geometry_msgs.msg import Point
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from robot_control.localization import (
    expected_measurement, load_landmarks, NoiseParams, PoseEKF, tag_measurement)
from robot_control.slam import (
    EKFSLAM, is_lap_end, MAP_COLUMNS, map_rows, MEASUREMENT_COLUMNS, measurement_row,
    SLAMParams, SNAPSHOT_COLUMNS)
from robot_control.waypoint_controller import (
    body_to_wheel_cmds, Gains, load_waypoints, WaypointController, wheel_cmds_to_body, wrap_angle)
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray


def parse_tag_id(frame):
    """Tag id from an apriltag_ros frame name ('tag_3' or '36h11:3'), or None."""
    match = re.search(r'[_:](\d+)$', frame)
    return int(match.group(1)) if match else None


class VelocityControlNode(Node):
    """
    Follow the waypoints in a file, starting from pose (0, 0, 0).

    There are no wheel encoders, so position is dead-reckoned from the commanded wheel speeds
    (speed_per_unit and wheel_base must be calibrated). Heading is either dead-reckoned too or,
    with use_imu:=true, taken from the IMU yaw relative to its value at startup.

    With use_vision:=true the dead reckoning becomes the prediction step of an EKF that is
    corrected by AprilTag sightings (camera_frame -> tag_N transforms on /tf from apriltag_node)
    of the landmarks listed in landmarks_file. With use_imu too, the prediction turns by the IMU
    yaw change instead of the commanded turn rate, so a turn that stalls is not taken as done.

    With use_slam:=true too (HW3), no landmarks_file is used: EKF-SLAM (robot_control/slam.py)
    estimates the landmarks along with the pose, adding each tag id the first time it is seen.
    The map is logged at the end of every lap (map_snapshots.csv) and at the end (map.csv).
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
        self.declare_parameter('use_vision', False)
        self.declare_parameter('landmarks_file', '')
        self.declare_parameter('camera_frame', 'camera_frame')
        self.declare_parameter('camera_x', 0.0)        # m, camera ahead of the rotation centre
        self.declare_parameter('camera_y', 0.0)        # m, camera left of the rotation centre
        self.declare_parameter('camera_yaw', 0.0)      # rad, 0 if the camera looks straight ahead
        self.declare_parameter('camera_upside_down', False)  # image rotated 180 degrees
        self.declare_parameter('max_tag_range', 3.0)   # m, ignore tags farther than this
        self.declare_parameter('relocalize_after', 15)  # rejections in a row before forcing one
        self.declare_parameter('log_dir', '')          # write trajectory/waypoint CSVs under here
        self.declare_parameter('log_period', 0.5)      # s between trajectory samples
        self.declare_parameter('use_slam', False)      # HW3: map the tags instead of loading them
        self.declare_parameter('waypoints_per_lap', 0)  # 0: a lap ends at each return to (0, 0)
        self.declare_parameter('publish_map', True)    # MarkerArray on slam_map, for foxglove
        self.declare_parameter('map_frame', 'slam_map')
        defaults = Gains()
        for name, value in vars(defaults).items():
            self.declare_parameter(name, value)
        noise_defaults = NoiseParams()
        for name, value in vars(noise_defaults).items():
            self.declare_parameter(name, value)
        slam_defaults = SLAMParams()
        for name, value in vars(slam_defaults).items():
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
        self.use_vision = bool(self.get_parameter('use_vision').value)
        self.use_slam = self.use_vision and bool(self.get_parameter('use_slam').value)
        self.waypoints_per_lap = int(self.get_parameter('waypoints_per_lap').value)

        # Pose estimate in the start frame, and the body velocity actually being commanded
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        # Motion model alone (commanded velocities, never corrected), logged for comparison, and
        # the motion model's prediction since the last log line, from the EKF pose at that line
        self.odom = [0.0, 0.0, 0.0]
        self.prediction = [0.0, 0.0, 0.0]
        self.v = 0.0
        self.w = 0.0

        self.imu_yaw = None
        self.imu_yaw_offset = None
        self.imu_yaw_last = None
        if self.use_imu:
            self.create_subscription(Imu, 'imu_data', self.imu_callback, 10)

        self.ekf = None
        self.slam = None
        self.lap = 0
        self.landmarks_file = None
        self.map_publisher = None
        self.path = []
        self.last_map_publish = -math.inf
        self.tags_seen = 0
        self.last_fix = -math.inf  # monotonic time of the last accepted tag sighting
        self.last_phase = None
        self.rejected_in_row = 0
        if self.use_vision:
            self.setup_vision(noise_defaults, slam_defaults)

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

        self.trajectory_log = self.waypoint_log = self.measurement_log = None
        self.snapshot_log = None
        self.run_dir = None
        log_dir = self.get_parameter('log_dir').value
        if log_dir:
            self.setup_logs(os.path.expanduser(log_dir))

    def setup_vision(self, noise_defaults, slam_defaults):
        noise = NoiseParams(
            **{name: float(self.get_parameter(name).value) for name in vars(noise_defaults)})
        if self.use_slam:
            # The filter is given neither the number of landmarks nor where they are
            params = SLAMParams(**{name: type(value)(self.get_parameter(name).value)
                                   for name, value in vars(slam_defaults).items()})
            self.slam = self.ekf = EKFSLAM(noise, params)
            self.landmarks = {}
            if bool(self.get_parameter('publish_map').value):
                self.map_frame = self.get_parameter('map_frame').value
                self.map_publisher = self.create_publisher(MarkerArray, 'slam_map', 1)
        else:
            path = self.get_parameter('landmarks_file').value
            if not path:
                raise RuntimeError('landmarks_file parameter is required with use_vision')
            self.landmarks = load_landmarks(path)
            self.landmarks_file = path
            self.ekf = PoseEKF(noise=noise)
        self.camera_frame = self.get_parameter('camera_frame').value
        self.camera_pose = tuple(float(self.get_parameter(name).value)
                                 for name in ('camera_x', 'camera_y', 'camera_yaw'))
        self.camera_upside_down = bool(self.get_parameter('camera_upside_down').value)
        self.max_tag_range = float(self.get_parameter('max_tag_range').value)
        self.relocalize_after = int(self.get_parameter('relocalize_after').value)
        self.create_subscription(TFMessage, '/tf', self.tf_callback, 100)
        if self.use_slam:
            self.get_logger().info(
                f'EKF-SLAM: empty map, landmark_dof {self.slam.dof}, a tag is added after '
                f'{self.slam.params.min_sightings} agreeing sightings')
            return
        self.get_logger().info(f'Localizing with {len(self.landmarks)} landmarks from {path}')
        for tag_id, (lx, ly, lyaw) in sorted(self.landmarks.items()):
            self.get_logger().info(f'  tag {tag_id}: x={lx:.2f} y={ly:.2f} yaw={lyaw:.2f}')

    def setup_logs(self, log_dir):
        run_dir = os.path.join(log_dir, time.strftime('run_%Y%m%d_%H%M%S'))
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.trajectory_file = open(os.path.join(run_dir, 'trajectory.csv'), 'w', newline='')
        self.trajectory_log = csv.writer(self.trajectory_file)
        self.trajectory_log.writerow(
            ['t', 'x', 'y', 'theta', 'std_x', 'std_y', 'std_theta', 'waypoint', 'tags_seen',
             'odom_x', 'odom_y', 'odom_theta', 'pred_x', 'pred_y', 'pred_theta', 'n_landmarks'])
        self.waypoint_file = open(os.path.join(run_dir, 'waypoints.csv'), 'w', newline='')
        self.waypoint_log = csv.writer(self.waypoint_file)
        self.waypoint_log.writerow(
            ['index', 't', 'goal_x', 'goal_y', 'goal_theta', 'x', 'y', 'theta'])
        if self.slam is not None:
            self.measurement_file = open(
                os.path.join(run_dir, 'measurements.csv'), 'w', newline='')
            self.measurement_log = csv.writer(self.measurement_file)
            self.measurement_log.writerow(MEASUREMENT_COLUMNS)
            self.snapshot_file = open(os.path.join(run_dir, 'map_snapshots.csv'), 'w', newline='')
            self.snapshot_log = csv.writer(self.snapshot_file)
            self.snapshot_log.writerow(SNAPSHOT_COLUMNS)
        elif self.ekf is not None:
            shutil.copy(self.landmarks_file, os.path.join(run_dir, 'landmarks.txt'))
            # Every tag sighting next to what the current estimate predicted, for debugging
            self.measurement_file = open(
                os.path.join(run_dir, 'measurements.csv'), 'w', newline='')
            self.measurement_log = csv.writer(self.measurement_file)
            self.measurement_log.writerow(
                ['t', 'tag', 'range', 'bearing', 'orientation', 'exp_range', 'exp_bearing',
                 'exp_orientation', 'x', 'y', 'theta', 'accepted', 'mahalanobis2'])
        self.log_period = float(self.get_parameter('log_period').value)
        self.log_start = time.monotonic()
        self.last_log = -math.inf
        self.get_logger().info(f'Logging to {run_dir}')

    def imu_callback(self, msg):
        q = msg.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.imu_yaw_offset is None:
            self.imu_yaw_offset = yaw
        self.imu_yaw = yaw

    def tf_callback(self, msg):
        for tf in msg.transforms:
            if tf.header.frame_id != self.camera_frame:
                continue
            tag_id = parse_tag_id(tf.child_frame_id)
            if tag_id is None or (not self.use_slam and tag_id not in self.landmarks):
                continue
            t, q = tf.transform.translation, tf.transform.rotation
            normal = (2.0 * (q.x * q.z + q.w * q.y),
                      2.0 * (q.y * q.z - q.w * q.x),
                      1.0 - 2.0 * (q.x * q.x + q.y * q.y))  # tag z axis in the camera frame
            measurement = tag_measurement(
                (t.x, t.y, t.z), normal, self.camera_pose, self.camera_upside_down)
            if measurement[0] > self.max_tag_range:
                continue

            stamp = Time.from_msg(tf.header.stamp).nanoseconds * 1e-9
            if self.use_slam:
                self.slam_sighting(tag_id, measurement, stamp)
                continue
            force = self.rejected_in_row >= self.relocalize_after
            if force:
                self.get_logger().warn(f'{self.rejected_in_row} sightings rejected in a row; '
                                       f're-localizing on tag {tag_id}')
                self.ekf.inflate((0.3, 0.3, 0.3))
            pose = self.ekf.pose
            accepted, distance = self.ekf.correct(
                measurement, self.landmarks[tag_id], stamp, force=force)
            if self.measurement_log:
                expected = expected_measurement(pose, self.landmarks[tag_id])
                self.measurement_log.writerow(
                    [f'{time.monotonic() - self.log_start:.2f}', tag_id]
                    + [f'{v:.4f}' for v in (*measurement, *expected, *pose)]
                    + [int(accepted), f'{distance:.1f}'])
            if accepted:
                self.rejected_in_row = 0
                self.tags_seen += 1
                self.last_fix = time.monotonic()
            else:
                self.rejected_in_row += 1
                self.get_logger().info(
                    f'Rejected tag {tag_id} sighting (Mahalanobis^2 {distance:.1f})',
                    throttle_duration_sec=1.0)

    def slam_sighting(self, tag_id, measurement, stamp):
        """
        Feed one sighting to EKF-SLAM.

        There is no forced re-localization here: with SLAM it would overwrite the map. Only an
        accepted re-observation of a mapped tag confirms the heading; a freshly added landmark
        agrees with the current pose by construction, so it confirms nothing.
        """
        pose = self.slam.pose_at(stamp)
        obs = self.slam.observe(tag_id, measurement, stamp, w=self.w)
        if self.measurement_log:
            self.measurement_log.writerow(measurement_row(
                time.monotonic() - self.log_start, tag_id, measurement, obs, pose,
                self.slam.n_landmarks))
        if obs.event == 'updated':
            self.tags_seen += 1
            self.last_fix = time.monotonic()
        elif obs.event == 'added':
            m, _ = self.slam.landmarks()[tag_id]
            self.get_logger().info(
                f'New landmark: tag {tag_id} at ({m[0]:.2f}, {m[1]:.2f}); '
                f'{self.slam.n_landmarks} in the map')
        elif obs.event == 'rejected':
            self.get_logger().info(
                f'Rejected tag {tag_id} sighting (Mahalanobis^2 {obs.d2:.1f})',
                throttle_duration_sec=1.0)

    def control_loop(self):
        now = time.monotonic()
        dt = now - self.last_time
        self.last_time = now

        self.integrate_pose(dt)
        self.log_trajectory(now)
        self.publish_map(now)

        if now < self.hold_until or self.finished:
            self.send(0.0, 0.0)
            return
        if self.use_imu and self.imu_yaw is None:
            self.get_logger().info('Waiting for IMU data...', throttle_duration_sec=2.0)
            self.send(0.0, 0.0)
            return

        # Without the camera, any heading counts; with it, only one a tag was seen at
        confirmed = (self.ekf is None
                     or now - self.last_fix <= max(self.controller.gains.settle_time, 0.5))
        v, w, reached = self.controller.update(
            self.x, self.y, self.theta, dt, heading_confirmed=confirmed)
        phase = self.controller.phase
        if phase == WaypointController.SEARCH and self.last_phase != phase:
            self.get_logger().info('No tag confirms the heading; searching for one')
            if not self.use_slam:  # with SLAM this would also loosen every new landmark
                self.ekf.inflate((0.0, 0.0, 1.0))  # heading unknown: trust the next sighting
        self.last_phase = phase
        if reached is not None:
            wx, wy, wt = self.controller.waypoints[reached]
            self.get_logger().info(
                f'Reached waypoint [{reached}] ({wx:.2f}, {wy:.2f}, {wt:.2f}); estimated pose '
                f'({self.x:.2f}, {self.y:.2f}, {self.theta:.2f})')
            if self.waypoint_log:
                self.waypoint_log.writerow(
                    [reached, f'{now - self.log_start:.2f}', wx, wy, wt,
                     f'{self.x:.4f}', f'{self.y:.4f}', f'{self.theta:.4f}'])
                self.waypoint_file.flush()
            if self.slam is not None and is_lap_end(reached, (wx, wy, wt), self.waypoints_per_lap):
                self.lap += 1
                self.save_map(now, lap=self.lap)
            self.hold_until = now + self.pause_at_waypoint
        if self.controller.done:
            self.get_logger().info('All waypoints reached')
            self.finished = True
        self.send(v, w)

    def integrate_pose(self, dt):
        for pose in (self.odom, self.prediction):
            heading = pose[2] + self.w * dt / 2.0
            pose[0] += self.v * math.cos(heading) * dt
            pose[1] += self.v * math.sin(heading) * dt
            pose[2] = wrap_angle(pose[2] + self.w * dt)
        if self.ekf is not None:
            w = self.w
            if self.use_imu and self.imu_yaw is not None:
                if self.imu_yaw_last is not None and dt > 0.0:
                    w = self.imu_yaw_sign * wrap_angle(self.imu_yaw - self.imu_yaw_last) / dt
                self.imu_yaw_last = self.imu_yaw
            self.ekf.predict(self.v, w, dt, self.get_clock().now().nanoseconds * 1e-9)
            self.x, self.y, self.theta = self.ekf.pose
            return
        if self.use_imu and self.imu_yaw is not None:
            self.theta = wrap_angle(self.imu_yaw_sign * (self.imu_yaw - self.imu_yaw_offset))
            heading = self.theta
        else:
            heading = self.theta + self.w * dt / 2.0  # midpoint heading over the step
            self.theta = wrap_angle(self.theta + self.w * dt)
        self.x += self.v * math.cos(heading) * dt
        self.y += self.v * math.sin(heading) * dt

    def log_trajectory(self, now):
        if not self.trajectory_log or now - self.last_log < self.log_period:
            return
        self.last_log = now
        std = self.ekf.std if self.ekf is not None else (0.0, 0.0, 0.0)
        self.trajectory_log.writerow(
            [f'{now - self.log_start:.2f}', f'{self.x:.4f}', f'{self.y:.4f}', f'{self.theta:.4f}',
             f'{std[0]:.4f}', f'{std[1]:.4f}', f'{std[2]:.4f}',
             self.controller.index, self.tags_seen]
            + [f'{v:.4f}' for v in (*self.odom, *self.prediction)]
            + [self.slam.n_landmarks if self.slam is not None else 0])
        self.trajectory_file.flush()
        self.tags_seen = 0
        self.prediction = [self.x, self.y, self.theta]

    def save_map(self, now, lap=None):
        """Write map.csv (the current map) and, at the end of a lap, a map_snapshots entry."""
        rows = map_rows(self.slam)
        self.get_logger().info(
            (f'Lap {lap} done: ' if lap else 'Final map: ')
            + f'{len(rows)} landmarks, ids {sorted(self.slam.index)}')
        if not self.run_dir:
            return
        if lap is not None and self.snapshot_log:
            for row in rows:
                self.snapshot_log.writerow(
                    [lap, f'{now - self.log_start:.2f}']
                    + [f'{v:.4f}' for v in self.slam.pose] + row)
            self.snapshot_file.flush()
        with open(os.path.join(self.run_dir, 'map.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(MAP_COLUMNS)
            writer.writerows(rows)

    def publish_map(self, now):
        """Landmarks with 95% ellipses, the robot and its path as markers, at 2 Hz."""
        if self.map_publisher is None or now - self.last_map_publish < 0.5:
            return
        self.last_map_publish = now
        self.path.append((self.x, self.y))
        stamp = self.get_clock().now().to_msg()

        def marker(ns, marker_id, kind, rgba):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = ns, marker_id, kind, Marker.ADD
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.pose.orientation.w = 1.0
            return m

        def set_yaw(m, yaw):
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)

        markers = MarkerArray()
        for k, (tag_id, (mean, cov)) in enumerate(self.slam.landmarks().items()):
            evals, evecs = np.linalg.eigh(cov[:2, :2])
            ellipse = marker('ellipses', k, Marker.CYLINDER, (0.1, 0.4, 1.0, 0.5))
            ellipse.pose.position.x, ellipse.pose.position.y = float(mean[0]), float(mean[1])
            set_yaw(ellipse, math.atan2(evecs[1, 1], evecs[0, 1]))
            # 95% ellipse: semi-axes sqrt(5.991 lambda); at least 2 cm so it stays visible
            ellipse.scale.x = max(2 * math.sqrt(5.991 * max(evals[1], 0.0)), 0.02)
            ellipse.scale.y = max(2 * math.sqrt(5.991 * max(evals[0], 0.0)), 0.02)
            ellipse.scale.z = 0.01
            label = marker('labels', k, Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0, 1.0))
            label.pose.position.x, label.pose.position.y = float(mean[0]), float(mean[1])
            label.pose.position.z = 0.1
            label.scale.z = 0.1
            label.text = f'tag {tag_id}'
            markers.markers += [ellipse, label]
        robot = marker('robot', 0, Marker.ARROW, (1.0, 0.2, 0.2, 1.0))
        robot.pose.position.x, robot.pose.position.y = self.x, self.y
        set_yaw(robot, self.theta)
        robot.scale.x, robot.scale.y, robot.scale.z = 0.2, 0.05, 0.05
        path = marker('path', 0, Marker.LINE_STRIP, (1.0, 0.5, 0.0, 0.8))
        path.scale.x = 0.01
        for px, py in self.path[-2000:]:
            path.points.append(Point(x=px, y=py, z=0.0))
        markers.markers += [robot, path]
        self.map_publisher.publish(markers)

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
        if self.trajectory_log:
            self.trajectory_file.close()
            self.waypoint_file.close()
        if self.measurement_log:
            self.measurement_file.close()
        if self.slam is not None:
            try:
                self.save_map(time.monotonic())
            except Exception as e:  # never lose the logs above over this
                self.get_logger().error(f'Could not write map.csv: {e}')
            if self.snapshot_log:
                self.snapshot_file.close()
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
