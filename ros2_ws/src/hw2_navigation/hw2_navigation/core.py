"""ROS-free geometry and conservative command-model EKF; SI units throughout."""
import math
import numpy as np


def stamp_ns(stamp):
    """Convert a ROS-style sec/nanosec timestamp without float rounding."""
    return stamp.sec*1_000_000_000 + stamp.nanosec


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def transform(xyz, rpy):
    """T_parent_child: p_parent = T_parent_child @ p_child; Rz Ry Rx."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    t = np.eye(4)
    t[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                 [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    t[:3, 3] = xyz
    return t


def base_pose(world_tag, camera_tag, base_camera):
    t = world_tag @ np.linalg.inv(camera_tag) @ np.linalg.inv(base_camera)
    return np.array([t[0, 3], t[1, 3], math.atan2(t[1, 0], t[0, 0])])


# the Kalman filter
class PoseFilter:
    def __init__(self, process_std=(0.05, 0.05, 0.12)):
        self.x = np.zeros(3) # the state
        self.P = np.diag([0.03, 0.03, 0.08])**2 # the covariance of the state
        self.q = np.asarray(process_std)**2 # the process noise covariance 


    # motion model 
    def predict(self, v, w, dt):
        if not 0 < dt <= 0.2:
            return False
        h = self.x[2] + w*dt/2
        f = np.eye(3)
        f[0, 2], f[1, 2] = -v*dt*math.sin(h), v*dt*math.cos(h)
        self.x += [v*dt*math.cos(h), v*dt*math.sin(h), w*dt]
        self.x[2] = wrap(self.x[2])

        # this is the updating the covariance
        self.P = f @ self.P @ f.T + np.diag(self.q)*dt
        return True

    def correct(self, z, covariance, gate=11.34):
        z, covariance = np.asarray(z), np.asarray(covariance)
        if not np.isfinite(z).all() or not np.isfinite(covariance).all():
            return False, float('inf')
        innovation = z - self.x
        innovation[2] = wrap(innovation[2])
        s = self.P + covariance
        nis = float(innovation @ np.linalg.solve(s, innovation))
        if nis > gate:
            return False, nis
        k = np.linalg.solve(s, self.P.T).T
        self.x += k @ innovation
        self.x[2] = wrap(self.x[2])
        a = np.eye(3)-k
        self.P = a @ self.P @ a.T + k @ covariance @ k.T
        return True, nis
