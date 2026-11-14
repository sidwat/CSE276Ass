from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from robot_control.localization import NoiseParams
from robot_control.waypoint_controller import wrap_angle

# 99% chi-square cutoffs by measurement dimension
CHI2_99 = {2: 9.21, 3: 11.34}


@dataclass
class SLAMParams:
    landmark_dof: int = 3            # 3: (x, y, psi) per landmark, 2: (x, y) only
    min_sightings: int = 3           # sightings needed before a new tag enters the state
    candidate_window: float = 1.5    # s, those sightings must fall within this window
    candidate_spread: float = 0.15   # m, and their implied positions agree within this
    max_incidence: float = 1.05      # rad (60 deg), reject tags seen this far off their normal
    latency_std: float = 0.15        # s, tau in R: bearing/orientation noise of (w * tau)^2
    initial_std: float = 1e-4        # Sigma_0|0 = initial_std^2 * I (start pose defines frame)


def measurement_model(robot, landmark, dof=3):
    x, y, th = robot[0], robot[1], robot[2]
    dx, dy = landmark[0] - x, landmark[1] - y
    q = max(dx * dx + dy * dy, 1e-12)
    r = math.sqrt(q)
    h = [r, wrap_angle(math.atan2(dy, dx) - th)]
    H_r = [[-dx / r, -dy / r, 0.0],
           [dy / q, -dx / q, -1.0]]
    H_m = [[dx / r, dy / r],
           [-dy / q, dx / q]]
    if dof == 3:
        h.append(wrap_angle(landmark[2] - th))
        H_r.append([0.0, 0.0, -1.0])
        H_m = [row + [0.0] for row in H_m] + [[0.0, 0.0, 1.0]]
    return np.array(h), np.array(H_r), np.array(H_m)


def inverse_model(robot, z, dof=3):
    x, y, th = robot[0], robot[1], robot[2]
    r, phi = z[0], z[1]
    b = th + phi
    cb, sb = math.cos(b), math.sin(b)
    m = [x + r * cb, y + r * sb]
    J_x = [[1.0, 0.0, -r * sb],
           [0.0, 1.0, r * cb]]
    J_z = [[cb, -r * sb],
           [sb, r * cb]]
    if dof == 3:
        m.append(wrap_angle(th + z[2]))
        J_x.append([0.0, 0.0, 1.0])
        J_z = [row + [0.0] for row in J_z] + [[0.0, 0.0, 1.0]]
    return np.array(m), np.array(J_x), np.array(J_z)


def incidence_angle(z):
    return wrap_angle(z[1] + math.pi - z[2])


@dataclass
class Observation:
    kind: str                       # update | rejected | new | candidate | incidence
    mahalanobis2: float = float('nan')
    expected: tuple = (float('nan'),) * 3
    pose: tuple = (float('nan'),) * 3   # robot pose the sighting was interpreted from
    incidence: float = float('nan')


class EKFSLAM:

    def __init__(self, noise=None, params=None):
        self.noise = noise or NoiseParams()
        self.params = params or SLAMParams()
        if self.params.landmark_dof not in (2, 3):
            raise ValueError('landmark_dof must be 2 or 3')
        self.dof = self.params.landmark_dof
        self.mu = np.zeros(3)
        self.sigma = np.eye(3) * self.params.initial_std ** 2
        self.index = {}          # tag id -> offset of the landmark in mu
        self.sightings = {}      # tag id -> accepted updates (excluding the one that added it)
        self.candidates = {}     # tag id -> [(stamp, implied (x, y)), ...] not yet confirmed
        self.history = deque(maxlen=200)  # (stamp, robot pose) for late camera frames
        self.w = 0.0             # last turn rate, for the latency term of R
        self.nis = []            # Mahalanobis^2 of every accepted update (consistency check)
        self.counts = {'update': 0, 'rejected': 0, 'new': 0, 'candidate': 0, 'incidence': 0}

    @property
    def pose(self):
        return float(self.mu[0]), float(self.mu[1]), float(self.mu[2])

    @property
    def std(self):
        return tuple(float(v) for v in np.sqrt(np.maximum(np.diag(self.sigma)[:3], 0.0)))

    @property
    def n_landmarks(self):
        return len(self.index)

    def predict(self, v, w, dt, stamp=None):
        """Motion update: only the robot rows/columns of Sigma change, O(N)."""
        n = self.noise
        self.w = w
        dist, turn = v * dt, w * dt
        heading = self.mu[2] + turn / 2.0  # midpoint heading over the step
        c, s = math.cos(heading), math.sin(heading)
        self.mu[:3] += [dist * c, dist * s, turn]
        self.mu[2] = wrap_angle(self.mu[2])

        F_r = np.array([[1.0, 0.0, -dist * s],
                        [0.0, 1.0, dist * c],
                        [0.0, 0.0, 1.0]])

        W = np.array([[c, -dist * s / 2.0],
                      [s, dist * c / 2.0],
                      [0.0, 1.0]])
        M = np.diag([n.odom_dist_std ** 2 * abs(dist),
                     n.odom_turn_std ** 2 * abs(turn) + n.odom_drift_std ** 2 * abs(dist)])
        
        S = self.sigma
        S[:3, :3] = F_r @ S[:3, :3] @ F_r.T + W @ M @ W.T
        S[:3, 3:] = F_r @ S[:3, 3:]
        S[3:, :3] = S[:3, 3:].T
        
        if stamp is not None:
            self.history.append((stamp, self.mu[:3].copy()))

    def inflate(self, std):
        """Add process noise to the robot pose (e.g. when the heading is in doubt)."""
        self.sigma[:3, :3] += np.diag(np.square(std))

    def measurement_noise(self, z, w=None):
        
        n, p = self.noise, self.params
        lag = ((self.w if w is None else w) * p.latency_std) ** 2
        diag = [(n.range_std + n.range_std_per_m * z[0]) ** 2, n.bearing_std ** 2 + lag]
        if self.dof == 3:
            diag.append(n.orientation_std ** 2 + lag)
        return np.diag(diag)

    @property
    def gate(self):
        return self.noise.gate if self.dof == 3 else CHI2_99[2]

    def _pose_at(self, stamp):
        if stamp is None:
            return self.mu[:3]
        for t, pose in reversed(self.history):
            if t <= stamp:
                return pose
        return self.mu[:3]

    def observe(self, tag_id, z, stamp=None):
        z = np.asarray(z, dtype=float)[:self.dof]
        incidence = incidence_angle(z) if len(z) == 3 else float('nan')
        pose = tuple(float(v) for v in self._pose_at(stamp))
        if self.dof == 3 and abs(incidence) > self.params.max_incidence:
            self.counts['incidence'] += 1
            return Observation('incidence', pose=pose, incidence=incidence)
        if tag_id in self.index:
            accepted, d2, expected = self.update(tag_id, z, stamp)
            kind = 'update' if accepted else 'rejected'
            self.counts[kind] += 1
            return Observation(kind, d2, tuple(expected), pose, incidence)
        if self._candidate_confirmed(tag_id, z, stamp, pose):
            self.add_landmark(tag_id, z, stamp)
            self.counts['new'] += 1
            return Observation('new', pose=pose, incidence=incidence)
        self.counts['candidate'] += 1
        return Observation('candidate', pose=pose, incidence=incidence)

    def _candidate_confirmed(self, tag_id, z, stamp, pose):
        p = self.params
        implied, _, _ = inverse_model(pose, z, self.dof)
        t = stamp if stamp is not None else (self.history[-1][0] if self.history else 0.0)
        buf = [c for c in self.candidates.get(tag_id, []) if t - c[0] <= p.candidate_window]
        buf.append((t, implied[:2]))
        self.candidates[tag_id] = buf
        if len(buf) < p.min_sightings:
            return False
        pts = np.array([c[1] for c in buf[-p.min_sightings:]])
        spread = np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))
        if spread > p.candidate_spread:
            return False
        del self.candidates[tag_id]
        return True

    def update(self, tag_id, z, stamp=None, R=None):
        j, d = self.index[tag_id], self.dof
        pose = self._pose_at(stamp)
        h, H_r, H_m = measurement_model(pose, self.mu[j:j + d], d)
        H = np.zeros((d, len(self.mu)))
        H[:, :3] = H_r
        H[:, j:j + d] = H_m
        z = np.asarray(z, dtype=float)[:d]
        nu = z - h
        nu[1] = wrap_angle(nu[1])
        if d == 3:
            nu[2] = wrap_angle(nu[2])
        R = self.measurement_noise(z) if R is None else R
        S = H @ self.sigma @ H.T + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        if d2 > self.gate:
            return False, d2, h
        K = np.linalg.solve(S, H @ self.sigma).T  # Sigma H^T S^-1 (S, Sigma symmetric)
        self.mu += K @ nu
        self._wrap_angles()
        IKH = np.eye(len(self.mu)) - K @ H
        self.sigma = IKH @ self.sigma @ IKH.T + K @ R @ K.T  # Joseph form
        self.sigma = (self.sigma + self.sigma.T) / 2.0
        self.nis.append(d2)
        self.sightings[tag_id] = self.sightings.get(tag_id, 0) + 1
        return True, d2, h

    def add_landmark(self, tag_id, z, stamp=None, R=None):
        
        d = self.dof
        z = np.asarray(z, dtype=float)[:d]

        m, J_x, J_z = inverse_model(self._pose_at(stamp), z, d)
        R = self.measurement_noise(z) if R is None else R

        S_nx = J_x @ self.sigma[:3, :]
        S_nn = J_x @ self.sigma[:3, :3] @ J_x.T + J_z @ R @ J_z.T

        self.index[tag_id] = len(self.mu)
        self.sightings[tag_id] = 0

        self.mu = np.concatenate([self.mu, m])
        self.sigma = np.block([[self.sigma, S_nx.T], [S_nx, S_nn]])

        self.sigma = (self.sigma + self.sigma.T) / 2.0

    def _wrap_angles(self):
        self.mu[2] = wrap_angle(self.mu[2])
        if self.dof == 3 and len(self.mu) > 3:
            psi = self.mu[5::3]
            self.mu[5::3] = np.arctan2(np.sin(psi), np.cos(psi))

    # --- Map access -------------------------------------------------------------------------

    def landmarks(self):
        out = {}
        for tag_id, j in self.index.items():
            mean = list(self.mu[j:j + 2]) + [self.mu[j + 2] if self.dof == 3 else float('nan')]
            psi_var = self.sigma[j + 2, j + 2] if self.dof == 3 else float('nan')
            out[tag_id] = (tuple(float(v) for v in mean), self.sigma[j:j + 2, j:j + 2].copy(),
                           float(psi_var))
        return out

    def map_uncertainty(self):
        if not self.index:
            return float('nan')
        return float(np.mean([math.sqrt(max(np.trace(c), 0.0) / 2.0)
                              for _, c, _ in self.landmarks().values()]))
