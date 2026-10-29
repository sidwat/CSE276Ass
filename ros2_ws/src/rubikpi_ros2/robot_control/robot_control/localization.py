"""EKF localization against AprilTag landmarks at known positions (no ROS dependencies)."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from robot_control.waypoint_controller import wrap_angle


def load_landmarks(path):
    """Read 'id, x, y, yaw' lines into {id: (x, y, yaw)}, skipping blanks and '#' comments."""
    landmarks = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) != 4:
                raise ValueError(f'{path}:{lineno}: expected "id, x, y, yaw", got {line!r}')
            tag_id = int(parts[0])
            if tag_id in landmarks:
                raise ValueError(f'{path}:{lineno}: tag {tag_id} is listed twice')
            landmarks[tag_id] = tuple(float(p) for p in parts[1:])
    return landmarks


def tag_measurement(translation, normal, camera_pose):
    """
    Turn a tag pose in the camera optical frame into (range, bearing, orientation) on the robot.

    translation is the tag centre and normal is the tag's z axis, both in the optical frame
    (x right, y down, z forward). camera_pose is (x, y, yaw) of the camera on the robot (x forward,
    y left). range and bearing locate the tag centre in the floor plane; orientation is the
    direction the printed face points, in the robot frame. The sign of the tag's z axis differs
    between pose estimators, so the normal is flipped to point back at the camera if needed.
    """
    tx, ty, tz = translation
    nx, ny, nz = normal
    if nx * tx + ny * ty + nz * tz > 0:
        nx, nz = -nx, -nz
    cam_x, cam_y, cam_yaw = camera_pose
    c, s = math.cos(cam_yaw), math.sin(cam_yaw)
    px = cam_x + c * tz + s * tx  # optical (z, -x) is (forward, left) on the robot
    py = cam_y + s * tz - c * tx
    face = cam_yaw + math.atan2(-nx, nz)
    return math.hypot(px, py), math.atan2(py, px), wrap_angle(face)


@dataclass
class NoiseParams:
    # Odometry errors as random walks: variance grows linearly with distance driven / angle
    # turned, so the result does not depend on the control rate.
    odom_dist_std: float = 0.1         # m of error after driving 1 m
    odom_turn_std: float = 0.1         # rad of error after turning 1 rad
    odom_drift_std: float = 0.05       # rad of heading drift after driving 1 m
    range_std: float = 0.02            # m
    range_std_per_m: float = 0.03      # added per m of range (PnP depth error grows with range)
    bearing_std: float = 0.02          # rad
    orientation_std: float = 0.15      # rad, PnP tag orientation is the noisiest part
    gate: float = 11.34                # Mahalanobis^2 cutoff, 99% for 3 DOF


class PoseEKF:
    """
    Extended Kalman filter on (x, y, theta).

    predict() integrates the commanded body velocity (dead reckoning, as in HW1) and grows the
    uncertainty; correct() fuses one tag sighting. Camera images arrive late, so the innovation
    is computed against the pose at the image timestamp (kept in a short history) and the
    correction is applied to the current pose.
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, initial_std=(0.02, 0.02, 0.02), noise=None):
        self.mu = np.array([x, y, theta], dtype=float)
        self.sigma = np.diag(np.square(initial_std))
        self.noise = noise or NoiseParams()
        self.history = deque(maxlen=200)

    @property
    def pose(self):
        return float(self.mu[0]), float(self.mu[1]), float(self.mu[2])

    @property
    def std(self):
        return tuple(float(v) for v in np.sqrt(np.diag(self.sigma)))

    def predict(self, v, w, dt, stamp=None):
        n = self.noise
        dist, turn = v * dt, w * dt
        heading = self.mu[2] + turn / 2.0  # midpoint heading over the step
        c, s = math.cos(heading), math.sin(heading)
        self.mu += [dist * c, dist * s, turn]
        self.mu[2] = wrap_angle(self.mu[2])

        G = np.array([[1.0, 0.0, -dist * s],
                      [0.0, 1.0, dist * c],
                      [0.0, 0.0, 1.0]])
        # Noise on (distance, rotation), mapped into (x, y, theta)
        V = np.array([[c, -dist * s / 2.0],
                      [s, dist * c / 2.0],
                      [0.0, 1.0]])
        M = np.diag([n.odom_dist_std ** 2 * abs(dist),
                     n.odom_turn_std ** 2 * abs(turn) + n.odom_drift_std ** 2 * abs(dist)])
        self.sigma = G @ self.sigma @ G.T + V @ M @ V.T
        if stamp is not None:
            self.history.append((stamp, self.mu.copy()))

    def _pose_at(self, stamp):
        if stamp is None:
            return self.mu
        for t, mu in reversed(self.history):
            if t <= stamp:
                return mu
        return self.mu

    def correct(self, measurement, landmark, stamp=None, force=False):
        """
        Fuse one (range, bearing, orientation) sighting of the landmark (x, y, yaw).

        Returns (accepted, mahalanobis^2). Sightings that disagree with the current estimate by
        more than the gate are rejected as outliers, unless force is set (re-localization).
        """
        n = self.noise
        x, y, theta = self._pose_at(stamp)
        lx, ly, lyaw = landmark
        dx, dy = lx - x, ly - y
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        expected = np.array([r, wrap_angle(math.atan2(dy, dx) - theta), wrap_angle(lyaw - theta)])
        H = np.array([[-dx / r, -dy / r, 0.0],
                      [dy / q, -dx / q, -1.0],
                      [0.0, 0.0, -1.0]])

        z = np.asarray(measurement, dtype=float)
        innovation = z - expected
        innovation[1] = wrap_angle(innovation[1])
        innovation[2] = wrap_angle(innovation[2])

        R = np.diag([(n.range_std + n.range_std_per_m * z[0]) ** 2,
                     n.bearing_std ** 2,
                     n.orientation_std ** 2])
        S = H @ self.sigma @ H.T + R
        S_inv = np.linalg.inv(S)
        distance = float(innovation @ S_inv @ innovation)
        if distance > n.gate and not force:
            return False, distance

        K = self.sigma @ H.T @ S_inv
        self.mu += K @ innovation
        self.mu[2] = wrap_angle(self.mu[2])
        self.sigma = (np.eye(3) - K @ H) @ self.sigma
        self.sigma = (self.sigma + self.sigma.T) / 2.0
        return True, distance

    def inflate(self, std):
        """Add uncertainty, e.g. before forcing a correction when the estimate is lost."""
        self.sigma += np.diag(np.square(std))
