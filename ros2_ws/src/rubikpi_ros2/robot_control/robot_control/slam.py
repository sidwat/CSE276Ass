"""
EKF-SLAM with AprilTag landmarks (no ROS dependencies).

State:       x = [x_r, y_r, theta, x_1, y_1, psi_1, ..., x_N, y_N, psi_N]  (3 + 3N entries)
Measurement: z = [range, bearing, orientation] of one tag, in the robot frame

(x_r, y_r, theta) is the robot pose in the start frame and (x_j, y_j, psi_j) the position and
facing direction of landmark j. N starts at 0 and grows: a tag id is added to the state the first
time it is confirmed, so the filter never knows how many landmarks exist. With landmark_dof = 2
the landmarks are (x_j, y_j) only and the orientation row of z is not used.

The start pose defines the map frame, so x_0|0 = 0 with (near) zero covariance and an empty map.
"""

from collections import deque, namedtuple
from dataclasses import dataclass
import math

import numpy as np

from robot_control.localization import NoiseParams
from robot_control.waypoint_controller import wrap_angle


@dataclass
class SLAMParams:
    landmark_dof: int = 3              # 3: landmarks are (x, y, psi); 2: (x, y)
    # A landmark is added only after min_sightings within confirm_window seconds whose implied
    # positions agree within confirm_distance: a bad initialisation can never be undone.
    min_sightings: int = 3
    confirm_window: float = 1.0        # s
    confirm_distance: float = 0.10     # m
    max_incidence: float = 1.047       # rad (60 deg) off the tag normal; PnP orientation flips
    latency: float = 0.15              # s, tau: camera delay/timestamp error, R grows by (w tau)^2
    initial_std: float = 1e-4          # m, rad: the start pose is the map frame, so ~0
    gate_2dof: float = 9.21            # Mahalanobis^2 cutoff, 99% for 2 DOF (NoiseParams.gate: 3)


Observation = namedtuple('Observation', 'event d2 expected')
"""
Result of EKFSLAM.observe().

event is one of 'skipped' (incidence too steep), 'candidate' (unknown id, not yet confirmed),
'added' (new landmark), 'updated' (accepted re-observation) or 'rejected' (failed the gate).
d2 is the Mahalanobis^2 innovation (nan unless updated/rejected) and expected the predicted
(range, bearing, orientation) (None for unmapped tags).
"""


def measurement_model(robot, landmark, dof=3):
    """
    Predicted measurement h and its Jacobian H_low for one landmark.

    robot is (x, y, theta), landmark (x, y[, psi]). H_low has dof rows and 3 + dof columns,
    ordered (x_r, y_r, theta, x_j, y_j[, psi_j]).
    """
    x, y, theta = robot
    dx, dy = landmark[0] - x, landmark[1] - y
    q = dx * dx + dy * dy
    r = math.sqrt(q)
    h = [r, wrap_angle(math.atan2(dy, dx) - theta)]
    H = [[-dx / r, -dy / r, 0.0, dx / r, dy / r],
         [dy / q, -dx / q, -1.0, -dy / q, dx / q]]
    if dof == 3:
        h.append(wrap_angle(landmark[2] - theta))
        H[0].append(0.0)
        H[1].append(0.0)
        H.append([0.0, 0.0, -1.0, 0.0, 0.0, 1.0])
    return np.array(h), np.array(H)


def inverse_measurement(robot, z, dof=3):
    """
    Landmark implied by one sighting z seen from robot, and the Jacobians J_x, J_z.

    J_x = dg/d(robot) is dof x 3 and J_z = dg/dz is dof x dof.
    """
    x, y, theta = robot
    r, phi = z[0], z[1]
    b = theta + phi
    cb, sb = math.cos(b), math.sin(b)
    m = [x + r * cb, y + r * sb]
    Jx = [[1.0, 0.0, -r * sb],
          [0.0, 1.0, r * cb]]
    Jz = [[cb, -r * sb],
          [sb, r * cb]]
    if dof == 3:
        m.append(wrap_angle(theta + z[2]))
        Jx.append([0.0, 0.0, 1.0])
        Jz = [row + [0.0] for row in Jz] + [[0.0, 0.0, 1.0]]
    return np.array(m), np.array(Jx), np.array(Jz)


def incidence(z):
    """Angle between the line of sight and the tag normal (0: seen face on)."""
    return wrap_angle(z[1] + math.pi - z[2])


class EKFSLAM:
    """
    EKF-SLAM over the robot pose and a growing list of landmarks.

    Same interface as PoseEKF (pose, std, predict) plus observe(), which buffers sightings of
    unknown ids until they are confirmed, adds them, and otherwise runs the gated EKF update.
    Camera images arrive late, so h and H are evaluated at the robot pose at the image time
    (kept in a short history) and the correction is applied to the current state.
    """

    def __init__(self, noise=None, params=None):
        self.noise = noise or NoiseParams()
        self.params = params or SLAMParams()
        if self.params.landmark_dof not in (2, 3):
            raise ValueError('landmark_dof must be 2 or 3')
        self.dof = self.params.landmark_dof
        self.mu = np.zeros(3)
        self.sigma = np.eye(3) * self.params.initial_std ** 2
        self.index = {}       # tag id -> offset of the landmark in mu
        self.candidates = {}  # tag id -> deque of (stamp, implied x, implied y)
        self.history = deque(maxlen=200)  # (stamp, robot pose) after each predict
        self.time = 0.0       # stand-in clock for callers that pass no stamps

    # --- state access -------------------------------------------------------------------------

    @property
    def pose(self):
        return float(self.mu[0]), float(self.mu[1]), float(self.mu[2])

    @property
    def std(self):
        return tuple(float(v) for v in np.sqrt(np.maximum(np.diag(self.sigma)[:3], 0.0)))

    @property
    def n_landmarks(self):
        return len(self.index)

    def landmarks(self):
        """{tag id: (mean, covariance)} with dof entries each, in the order they were added."""
        d = self.dof
        return {tag_id: (self.mu[j:j + d].copy(), self.sigma[j:j + d, j:j + d].copy())
                for tag_id, j in self.index.items()}

    def pose_at(self, stamp):
        """Robot pose at time stamp, from the history (the current pose if not covered)."""
        if stamp is None:
            return self.mu[:3].copy()
        for t, pose in reversed(self.history):
            if t <= stamp:
                return pose.copy()
        return self.mu[:3].copy()

    # --- motion -------------------------------------------------------------------------------

    def predict(self, v, w, dt, stamp=None):
        """Move the robot by the body velocity (v, w) for dt; landmarks are static. O(N)."""
        n = self.noise
        dist, turn = v * dt, w * dt
        heading = self.mu[2] + turn / 2.0  # midpoint heading over the step
        c, s = math.cos(heading), math.sin(heading)
        self.mu[:3] += [dist * c, dist * s, turn]
        self.mu[2] = wrap_angle(self.mu[2])

        Fr = np.array([[1.0, 0.0, -dist * s],
                       [0.0, 1.0, dist * c],
                       [0.0, 0.0, 1.0]])
        # Odometry error as random walks in (distance, rotation), mapped into (x, y, theta):
        # W = G_r / dt, so the variance per step does not depend on the loop rate.
        W = np.array([[c, -dist * s / 2.0],
                      [s, dist * c / 2.0],
                      [0.0, 1.0]])
        M = np.diag([n.odom_dist_std ** 2 * abs(dist),
                     n.odom_turn_std ** 2 * abs(turn) + n.odom_drift_std ** 2 * abs(dist)])
        S = self.sigma
        S[:3, :3] = Fr @ S[:3, :3] @ Fr.T + W @ M @ W.T
        S[:3, 3:] = Fr @ S[:3, 3:]
        S[3:, :3] = S[:3, 3:].T

        self.time = stamp if stamp is not None else self.time + dt
        self.history.append((self.time, self.mu[:3].copy()))

    def inflate(self, std):
        """Add uncertainty to the robot pose (x, y, theta)."""
        self.sigma[:3, :3] += np.diag(np.square(std))

    # --- measurements -------------------------------------------------------------------------

    def measurement_noise(self, r, w=0.0):
        """R for a sighting at range r while turning at w (rad/s)."""
        n = self.noise
        lag = (w * self.params.latency) ** 2
        R = np.diag([(n.range_std + n.range_std_per_m * r) ** 2,
                     n.bearing_std ** 2 + lag,
                     n.orientation_std ** 2 + lag])
        return R[:self.dof, :self.dof]

    def expected(self, tag_id, stamp=None):
        """Predicted (range, bearing, orientation) of a mapped tag, or None."""
        j = self.index.get(tag_id)
        if j is None:
            return None
        lm = self.mu[j:j + self.dof]
        pose = self.pose_at(stamp)
        h, _ = measurement_model(pose, lm, self.dof)
        if self.dof == 2:
            return float(h[0]), float(h[1]), math.nan
        return float(h[0]), float(h[1]), float(h[2])

    def observe(self, tag_id, z, stamp=None, w=0.0):
        """
        Process one (range, bearing, orientation) sighting of tag_id; return an Observation.

        A tag already in the map is a normal (gated) update; reappearing after being out of
        view is the same thing (a loop closure). An unknown tag is buffered until confirmed.
        """
        z = np.asarray(z, dtype=float)
        if abs(incidence(z)) > self.params.max_incidence:
            return Observation('skipped', math.nan, self.expected(tag_id, stamp))
        if tag_id in self.index:
            expected = self.expected(tag_id, stamp)
            accepted, d2 = self.update(tag_id, z, stamp, w)
            return Observation('updated' if accepted else 'rejected', d2, expected)

        if stamp is None:
            stamp = self.time
        pose = self.pose_at(stamp)
        b = pose[2] + z[1]
        buffer = self.candidates.setdefault(tag_id, deque(maxlen=20))
        buffer.append((stamp, pose[0] + z[0] * math.cos(b), pose[1] + z[0] * math.sin(b)))
        while buffer and buffer[0][0] < stamp - self.params.confirm_window:
            buffer.popleft()
        k = self.params.min_sightings
        if len(buffer) >= k:
            recent = np.array([(px, py) for _, px, py in list(buffer)[-k:]])
            spread = np.linalg.norm(recent - recent.mean(axis=0), axis=1).max()
            if spread <= self.params.confirm_distance:
                self.add_landmark(tag_id, z, stamp, w)
                del self.candidates[tag_id]
                return Observation('added', math.nan, None)
        return Observation('candidate', math.nan, None)

    def update(self, tag_id, z, stamp=None, w=0.0):
        """EKF update with a sighting of a mapped landmark; returns (accepted, Mahalanobis^2)."""
        j = self.index[tag_id]
        d = self.dof
        pose = self.pose_at(stamp)
        h, H_low = measurement_model(pose, self.mu[j:j + d], d)
        z = np.asarray(z, dtype=float)[:d]
        R = self.measurement_noise(z[0], w)

        # H = H_low F_x,j: only the robot and landmark j columns are non-zero
        cols = np.r_[0:3, j:j + d]
        nu = z - h
        nu[1:] = [wrap_angle(a) for a in nu[1:]]
        S = H_low @ self.sigma[np.ix_(cols, cols)] @ H_low.T + R
        S_inv = np.linalg.inv(S)
        d2 = float(nu @ S_inv @ nu)
        gate = self.noise.gate if d == 3 else self.params.gate_2dof
        if d2 > gate:
            return False, d2

        PHt = self.sigma[:, cols] @ H_low.T          # Sigma H^T
        K = PHt @ S_inv
        correction = K @ nu
        self.mu += correction
        self._wrap_angles()
        # Joseph form keeps Sigma symmetric positive semi-definite
        n = len(self.mu)
        IKH = np.eye(n)
        IKH[:, cols] -= K @ H_low
        self.sigma = IKH @ self.sigma @ IKH.T + K @ R @ K.T
        self.sigma = (self.sigma + self.sigma.T) / 2.0
        self._shift_history(correction[:3])
        return True, d2

    def add_landmark(self, tag_id, z, stamp=None, w=0.0):
        """Append a landmark from one sighting with the inverse sensor model (augmentation)."""
        d = self.dof
        pose = self.pose_at(stamp)
        z = np.asarray(z, dtype=float)
        m, Jx, Jz = inverse_measurement(pose, z, d)
        R = self.measurement_noise(z[0], w)
        S_nx = Jx @ self.sigma[:3, :]                     # cross-covariance with the whole state
        S_nn = Jx @ self.sigma[:3, :3] @ Jx.T + Jz @ R @ Jz.T
        self.index[tag_id] = len(self.mu)
        self.mu = np.concatenate([self.mu, m])
        self.sigma = np.block([[self.sigma, S_nx.T], [S_nx, S_nn]])

    def _wrap_angles(self):
        self.mu[2] = wrap_angle(self.mu[2])
        if self.dof == 3:
            psi = self.mu[5::3]
            self.mu[5::3] = np.arctan2(np.sin(psi), np.cos(psi))

    def _shift_history(self, delta):
        """
        Apply a correction to the stored poses too.

        Otherwise a second tag in the same (late) image would be compared with the uncorrected
        pose at that time and the first tag's correction would be applied twice.
        """
        for _, pose in self.history:
            pose += delta
            pose[2] = wrap_angle(pose[2])


def is_lap_end(index, goal, waypoints_per_lap=0):
    """Tell whether waypoint index ends a lap: every waypoints_per_lap, or (0) back at (0, 0)."""
    if waypoints_per_lap > 0:
        return (index + 1) % waypoints_per_lap == 0
    return index > 0 and math.hypot(goal[0], goal[1]) < 0.01


# --- log formats shared by velocity_control and simulate_slam -------------------------------

MAP_COLUMNS = ['tag', 'x', 'y', 'psi', 'var_x', 'cov_xy', 'var_y', 'var_psi']
SNAPSHOT_COLUMNS = ['lap', 't', 'robot_x', 'robot_y', 'robot_theta'] + MAP_COLUMNS
MEASUREMENT_COLUMNS = ['t', 'tag', 'range', 'bearing', 'orientation', 'exp_range', 'exp_bearing',
                       'exp_orientation', 'x', 'y', 'theta', 'accepted', 'mahalanobis2',
                       'event', 'n_landmarks', 'incidence']


def map_rows(slam):
    """Format the map as one MAP_COLUMNS row per landmark."""
    rows = []
    for tag_id, (m, cov) in slam.landmarks().items():
        psi = m[2] if slam.dof == 3 else math.nan
        var_psi = cov[2, 2] if slam.dof == 3 else math.nan
        rows.append([tag_id] + [f'{v:.5f}' for v in (m[0], m[1], psi)]
                    + [f'{v:.3e}' for v in (cov[0, 0], cov[0, 1], cov[1, 1], var_psi)])
    return rows


def measurement_row(t, tag_id, z, obs, pose, n_landmarks):
    """One MEASUREMENT_COLUMNS row; pose is the robot pose at the image time."""
    expected = obs.expected or (math.nan,) * 3
    return ([f'{t:.2f}', tag_id] + [f'{v:.4f}' for v in (*z, *expected, *pose)]
            + [int(obs.event in ('updated', 'added')), f'{obs.d2:.2f}', obs.event, n_landmarks,
               f'{incidence(z):.3f}'])
