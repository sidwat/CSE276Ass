import csv
import math
import os
import re
import shutil
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from robot_control.localization import (
    expected_measurement, load_landmarks, NoiseParams, pose_from_measurement, PoseEKF,
    tag_measurement)
from robot_control.waypoint_controller import (
    body_to_wheel_cmds, Gains, load_waypoints, WaypointController, wheel_cmds_to_body, wrap_angle)
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage


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
        self.declare_parameter('log_dir', '')          # write each run's CSVs and plots under here
        self.declare_parameter('run_name', '')         # run folder name; default run_<time>
        defaults = Gains()
        for name, value in vars(defaults).items():
            self.declare_parameter(name, value)
        noise_defaults = NoiseParams()
        for name, value in vars(noise_defaults).items():
            self.declare_parameter(name, value)

        path = self.get_parameter('waypoints_file').value
        if not path:
            raise RuntimeError('waypoints_file parameter is required')
        waypoints = load_waypoints(path)
        self.waypoints_file = path
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

        # Pose estimate in the start frame, and the body velocity actually being commanded
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.w = 0.0
        self.odom = [0.0, 0.0, 0.0]  # dead reckoning only, never corrected (for the plots)
        self.left_sent = self.right_sent = 0.0  # wheel commands last published, after trim
        self.tick_sightings = []  # (tag, implied pose, accepted) since the last control tick

        self.imu_yaw = None
        self.imu_yaw_offset = None
        self.imu_yaw_last = None
        if self.use_imu:
            self.create_subscription(Imu, 'imu_data', self.imu_callback, 10)

        self.ekf = None
        self.rejected_in_row = 0
        if self.use_vision:
            self.setup_vision(noise_defaults)

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

        self.waypoint_log = self.measurement_log = self.tick_log = None
        self.run_dir = None
        log_dir = self.get_parameter('log_dir').value
        if log_dir:
            self.setup_logs(os.path.expanduser(log_dir))

    def setup_vision(self, noise_defaults):
        path = self.get_parameter('landmarks_file').value
        if not path:
            raise RuntimeError('landmarks_file parameter is required with use_vision')
        self.landmarks = load_landmarks(path)
        self.landmarks_file = path
        noise = NoiseParams(
            **{name: float(self.get_parameter(name).value) for name in vars(noise_defaults)})
        self.ekf = PoseEKF(noise=noise)
        self.camera_frame = self.get_parameter('camera_frame').value
        self.camera_pose = tuple(float(self.get_parameter(name).value)
                                 for name in ('camera_x', 'camera_y', 'camera_yaw'))
        self.camera_upside_down = bool(self.get_parameter('camera_upside_down').value)
        self.max_tag_range = float(self.get_parameter('max_tag_range').value)
        self.relocalize_after = int(self.get_parameter('relocalize_after').value)
        self.create_subscription(TFMessage, '/tf', self.tf_callback, 100)
        self.get_logger().info(f'Localizing with {len(self.landmarks)} landmarks from {path}')
        for tag_id, (lx, ly, lyaw) in sorted(self.landmarks.items()):
            self.get_logger().info(f'  tag {tag_id}: x={lx:.2f} y={ly:.2f} yaw={lyaw:.2f}')

    def setup_logs(self, log_dir):
        run_name = self.get_parameter('run_name').value or time.strftime('run_%Y%m%d_%H%M%S')
        run_dir = os.path.join(log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        shutil.copy(self.waypoints_file, os.path.join(run_dir, 'waypoints.txt'))
        self.save_config(run_dir)
        # One row per control tick. corr_* is the EKF state after every sighting fused since the
        # previous tick; model_* is the motion model applied to it with the last command (the
        # pose the controller then acts on); vision_* is the pose the last of those sightings
        # implies by itself; odom_* is dead reckoning that is never corrected.
        self.tick_file = open(os.path.join(run_dir, 'ticks.csv'), 'w', newline='')
        self.tick_log = csv.writer(self.tick_file)
        self.tick_log.writerow(
            ['t', 'dt', 'waypoint', 'mode', 'corr_x', 'corr_y', 'corr_theta',
             'std_x', 'std_y', 'std_theta', 'model_x', 'model_y', 'model_theta',
             'model_std_x', 'model_std_y', 'model_std_theta',
             'odom_x', 'odom_y', 'odom_theta', 'n_accepted', 'n_rejected', 'vision_tag',
             'vision_x', 'vision_y', 'vision_theta', 'vision_accepted',
             'v_cmd', 'w_cmd', 'v', 'w', 'left', 'right'])
        self.waypoint_file = open(os.path.join(run_dir, 'waypoints.csv'), 'w', newline='')
        self.waypoint_log = csv.writer(self.waypoint_file)
        self.waypoint_log.writerow(
            ['index', 't', 'goal_x', 'goal_y', 'goal_theta', 'x', 'y', 'theta'])
        if self.ekf is not None:
            shutil.copy(self.landmarks_file, os.path.join(run_dir, 'landmarks.txt'))
            # Every tag sighting next to what the current estimate predicted, for debugging
            self.measurement_file = open(
                os.path.join(run_dir, 'measurements.csv'), 'w', newline='')
            self.measurement_log = csv.writer(self.measurement_file)
            # status: accepted / rejected / forced (re-localization) / out_of_range (ignored).
            # img_* is the estimate at the image time, which the innovation is computed
            # against; x/y/theta and post_* are the current estimate before and after.
            self.measurement_log.writerow(
                ['t', 'stamp', 'latency', 'tag', 'status', 'accepted', 'mahalanobis2',
                 'cam_x', 'cam_y', 'cam_z', 'normal_x', 'normal_y', 'normal_z',
                 'range', 'bearing', 'orientation', 'img_x', 'img_y', 'img_theta',
                 'exp_range', 'exp_bearing', 'exp_orientation',
                 'x', 'y', 'theta', 'std_x', 'std_y', 'std_theta',
                 'post_x', 'post_y', 'post_theta', 'post_std_x', 'post_std_y', 'post_std_theta',
                 'vision_x', 'vision_y', 'vision_theta'])
        self.log_start = time.monotonic()
        self.get_logger().info(f'Logging to {run_dir}')

    def save_config(self, run_dir):
        """Store every parameter and the config files this run used, next to its logs."""
        import yaml
        names = self.list_parameters([], 0).names
        params = {p.name: p.value for p in self.get_parameters(names)}
        with open(os.path.join(run_dir, 'params.yaml'), 'w') as f:
            yaml.safe_dump({'velocity_control_node': {'ros__parameters': params}}, f)
        try:
            from ament_index_python.packages import get_package_share_directory
            for package, path in (('apriltag_ros', 'cfg/tags_36h11.yaml'),
                                  ('robot_vision_camera', 'config/camera_parameter.yaml')):
                shutil.copy(os.path.join(get_package_share_directory(package), path), run_dir)
        except Exception as e:  # informative only: never stop a run for it
            self.get_logger().warn(f'Could not copy the detector/camera config: {e}')

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
            if tag_id not in self.landmarks:
                continue
            t, q = tf.transform.translation, tf.transform.rotation
            normal = (2.0 * (q.x * q.z + q.w * q.y),
                      2.0 * (q.y * q.z - q.w * q.x),
                      1.0 - 2.0 * (q.x * q.x + q.y * q.y))  # tag z axis in the camera frame
            measurement = tag_measurement(
                (t.x, t.y, t.z), normal, self.camera_pose, self.camera_upside_down)
            stamp = Time.from_msg(tf.header.stamp).nanoseconds * 1e-9
            landmark = self.landmarks[tag_id]
            vision = pose_from_measurement(measurement, landmark)
            at_image = tuple(float(v) for v in self.ekf.pose_at(stamp))
            pose, std = self.ekf.pose, self.ekf.std
            if measurement[0] > self.max_tag_range:
                status, accepted, distance = 'out_of_range', False, math.nan
            else:
                force = self.rejected_in_row >= self.relocalize_after
                if force:
                    self.get_logger().warn(f'{self.rejected_in_row} sightings rejected in a '
                                           f'row; re-localizing on tag {tag_id}')
                    self.ekf.inflate((0.3, 0.3, 0.3))
                accepted, distance = self.ekf.correct(measurement, landmark, stamp, force=force)
                status = ('forced' if force else 'accepted') if accepted else 'rejected'
                self.tick_sightings.append((tag_id, vision, accepted))
            if self.measurement_log:
                now = self.get_clock().now().nanoseconds * 1e-9
                expected = expected_measurement(at_image, landmark)
                self.measurement_log.writerow(
                    [f'{time.monotonic() - self.log_start:.3f}', f'{stamp:.3f}',
                     f'{now - stamp:.3f}', tag_id, status, int(accepted), f'{distance:.2f}']
                    + [f'{v:.4f}' for v in (t.x, t.y, t.z, *normal, *measurement, *at_image,
                                            *expected, *pose, *std, *self.ekf.pose,
                                            *self.ekf.std, *vision)])
                self.measurement_file.flush()
            if status == 'out_of_range':
                continue
            if accepted:
                self.rejected_in_row = 0
            else:
                self.rejected_in_row += 1
                self.get_logger().info(
                    f'Rejected tag {tag_id} sighting (Mahalanobis^2 {distance:.1f})',
                    throttle_duration_sec=1.0)

    def control_loop(self):
        now = time.monotonic()
        dt = now - self.last_time
        self.last_time = now

        # The state as corrected by the sightings since the last tick, before predicting
        if self.ekf is not None:
            corrected, corrected_std = self.ekf.pose, self.ekf.std
        else:
            corrected, corrected_std = (self.x, self.y, self.theta), (0.0, 0.0, 0.0)
        self.integrate_pose(dt)

        v = w = 0.0
        if now < self.hold_until or self.finished:
            mode = 'hold'
        elif self.use_imu and self.imu_yaw is None:
            mode = 'wait_imu'
            self.get_logger().info('Waiting for IMU data...', throttle_duration_sec=2.0)
        else:
            mode = self.controller.phase
            v, w, reached = self.controller.update(self.x, self.y, self.theta, dt)
            if reached is not None:
                wx, wy, wt = self.controller.waypoints[reached]
                self.get_logger().info(
                    f'Reached waypoint [{reached}] ({wx:.2f}, {wy:.2f}, {wt:.2f}); estimated '
                    f'pose ({self.x:.2f}, {self.y:.2f}, {self.theta:.2f})')
                if self.waypoint_log:
                    self.waypoint_log.writerow(
                        [reached, f'{now - self.log_start:.2f}', wx, wy, wt,
                         f'{self.x:.4f}', f'{self.y:.4f}', f'{self.theta:.4f}'])
                    self.waypoint_file.flush()
                self.hold_until = now + self.pause_at_waypoint
            if self.controller.done:
                self.get_logger().info('All waypoints reached')
                self.finished = True
        self.send(v, w)
        self.log_tick(now, dt, mode, corrected, corrected_std, v, w)

    def integrate_pose(self, dt):
        odom_heading = self.odom[2] + self.w * dt / 2.0
        self.odom[0] += self.v * math.cos(odom_heading) * dt
        self.odom[1] += self.v * math.sin(odom_heading) * dt
        self.odom[2] = wrap_angle(self.odom[2] + self.w * dt)
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

    def log_tick(self, now, dt, mode, corrected, corrected_std, v_cmd, w_cmd):
        sightings, self.tick_sightings = self.tick_sightings, []
        if not self.tick_log:
            return
        model_std = self.ekf.std if self.ekf is not None else (0.0, 0.0, 0.0)
        accepted = sum(1 for s in sightings if s[2])
        if sightings:
            tag_id, vision, ok = sightings[-1]
            vision_cols = [tag_id, *(f'{v:.4f}' for v in vision), int(ok)]
        else:
            vision_cols = ['', '', '', '', '']
        self.tick_log.writerow(
            [f'{now - self.log_start:.3f}', f'{dt:.4f}', self.controller.index, mode]
            + [f'{v:.4f}' for v in (*corrected, *corrected_std, self.x, self.y, self.theta,
                                    *model_std, *self.odom)]
            + [accepted, len(sightings) - accepted] + vision_cols
            + [f'{v:.4f}' for v in (v_cmd, w_cmd, self.v, self.w,
                                    self.left_sent, self.right_sent)])
        self.tick_file.flush()

    def send(self, v, w):
        left, right = body_to_wheel_cmds(
            v, w, self.speed_per_unit, self.wheel_base, self.max_wheel_cmd)
        # Dead-reckon from what was actually sent, after saturation
        self.v, self.w = wheel_cmds_to_body(left, right, self.speed_per_unit, self.wheel_base)
        # The trim compensates for mismatched motors so the robot actually does what was
        # commanded, so it is applied after (and not included in) the dead reckoning above.
        self.left_sent = left * (1.0 - self.wheel_trim)
        self.right_sent = right * (1.0 + self.wheel_trim)
        msg = Float32MultiArray()
        msg.data = [float(self.left_sent), float(self.right_sent)]
        self.publisher.publish(msg)

    def destroy_node(self):
        try:
            self.send(0.0, 0.0)
        except Exception:
            pass
        if self.waypoint_log:
            self.waypoint_file.close()
        if self.measurement_log:
            self.measurement_file.close()
        if self.tick_log:
            self.tick_file.close()
        super().destroy_node()
        if self.run_dir:
            # The motors are already stopped; a plotting failure must not hide the logs
            try:
                from robot_control.visualize_run import report
                report(self.run_dir)
            except (Exception, SystemExit) as e:
                print(f'Plotting failed ({e}); retry with: '
                      f'ros2 run robot_control visualize_run run {self.run_dir}')


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
