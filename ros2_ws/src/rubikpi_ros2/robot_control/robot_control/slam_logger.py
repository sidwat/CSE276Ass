"""
Write everything the HW3 report needs from one SLAM run into log_dir/run_*/ (no ROS dependencies).

Shared by velocity_control (on the robot) and simulate_slam (offline) so both produce the same
files, which evaluate_slam reads:

  run_info.json        parameters, waypoint file, path type, laps; summary added at the end
  start_pose.txt       world-frame start pose: FILL IN with the tape-measured value after the run
  waypoints_input.txt  copy of the waypoint file that was driven
  trajectory.csv       filter pose + std, N, map uncertainty, dead reckoning, every log_period
  measurements.csv     every tag sighting: z, what happened to it, innovation size, pose used
  landmark_events.csv  each landmark the moment it entered the state
  waypoints.csv        estimate on arrival at each waypoint (with lap number)
  map_snapshots.csv    the whole map (means + covariances) at every waypoint, lap end and finish
  map_final.csv        the final map
  state_<label>.npz    full mu, Sigma and the id -> index map at each lap end and at the finish
  truth.csv            (simulation only) the true pose, for trajectory error
"""

import csv
import json
import math
import os
import shutil
import time

import numpy as np

START_POSE_TEMPLATE = """\
# World-frame pose of the robot's rotation centre at the start of this run, measured with a tape:
#   x0 [m], y0 [m], theta0 [rad]
# (world frame = the frame of landmarks_hw3_gt.txt: origin at the south-west corner, x east,
# y north). evaluate_slam uses it to move the map from the start frame into the world frame.
{pose}
"""

MAP_HEADER = ['label', 'lap', 't', 'tag', 'x', 'y', 'psi', 'cov_xx', 'cov_xy', 'cov_yy',
              'var_psi', 'sightings']


def fmt(v, digits=4):
    return f'{v:.{digits}f}' if isinstance(v, float) and math.isfinite(v) else str(v)


class SLAMRunLogger:
    """Create a run folder and append rows to its CSV files; flushes every write."""

    def __init__(self, log_dir, info, waypoints_file=None, start_pose=None, run_name=None):
        self.run_dir = os.path.join(
            os.path.expanduser(log_dir), run_name or time.strftime('run_%Y%m%d_%H%M%S'))
        os.makedirs(self.run_dir, exist_ok=True)
        self.t0 = time.monotonic()
        self.info = dict(info)
        self.info['started'] = time.strftime('%Y-%m-%d %H:%M:%S')
        if waypoints_file:
            shutil.copy(waypoints_file, self.path('waypoints_input.txt'))
            self.info['waypoints_file'] = os.path.abspath(waypoints_file)
        pose = ('{:.4f}, {:.4f}, {:.4f}'.format(*start_pose) if start_pose is not None
                else '# x0, y0, theta0   <- replace this line with the measured values')
        with open(self.path('start_pose.txt'), 'w') as f:
            f.write(START_POSE_TEMPLATE.format(pose=pose))
        self._write_info()

        self.files = {}
        self.writers = {}
        self._open('trajectory', [
            't', 'x', 'y', 'theta', 'std_x', 'std_y', 'std_theta', 'waypoint', 'lap', 'phase',
            'n_landmarks', 'map_std', 'sightings', 'updates', 'v', 'w',
            'odom_x', 'odom_y', 'odom_theta', 'pred_x', 'pred_y', 'pred_theta', 'imu_yaw'])
        self._open('measurements', [
            't', 'stamp', 'tag', 'range', 'bearing', 'orientation', 'incidence', 'kind',
            'mahalanobis2', 'exp_range', 'exp_bearing', 'exp_orientation', 'x', 'y', 'theta',
            'n_landmarks'])
        self._open('landmark_events', [
            't', 'tag', 'x', 'y', 'psi', 'std_x', 'std_y', 'robot_x', 'robot_y', 'robot_theta',
            'robot_std_theta', 'n_landmarks'])
        self._open('waypoints', [
            'index', 'lap', 't', 'goal_x', 'goal_y', 'goal_theta', 'x', 'y', 'theta',
            'std_x', 'std_y', 'std_theta', 'n_landmarks'])
        self._open('map_snapshots', MAP_HEADER)
        self.truth = None

    def path(self, name):
        return os.path.join(self.run_dir, name)

    def now(self):
        return time.monotonic() - self.t0

    def _open(self, name, header):
        f = open(self.path(name + '.csv'), 'w', newline='')
        self.files[name] = f
        self.writers[name] = csv.writer(f)
        self.writers[name].writerow(header)

    def _row(self, name, values):
        self.writers[name].writerow([fmt(v) for v in values])
        self.files[name].flush()

    def _write_info(self):
        with open(self.path('run_info.json'), 'w') as f:
            json.dump(self.info, f, indent=2, sort_keys=True, default=str)

    # --- Rows ---------------------------------------------------------------------------------

    def trajectory(self, t, slam, waypoint, lap, phase, sightings, updates, v, w, odom, pred,
                   imu_yaw=float('nan')):
        std = slam.std
        self._row('trajectory', [
            t, *slam.pose, *std, waypoint, lap, phase, slam.n_landmarks, slam.map_uncertainty(),
            sightings, updates, v, w, *odom, *pred, imu_yaw])

    def measurement(self, t, stamp, tag, z, obs, n_landmarks):
        z = list(z) + [float('nan')] * (3 - len(z))
        self._row('measurements', [
            t, stamp, tag, *z, obs.incidence, obs.kind, obs.mahalanobis2,
            *(list(obs.expected) + [float('nan')] * (3 - len(obs.expected))), *obs.pose,
            n_landmarks])

    def landmark_added(self, t, tag, slam):
        mean, cov, _ = slam.landmarks()[tag]
        self._row('landmark_events', [
            t, tag, *mean, math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1]), *slam.pose,
            slam.std[2], slam.n_landmarks])

    def waypoint(self, index, lap, t, goal, slam):
        self._row('waypoints', [index, lap, t, *goal, *slam.pose, *slam.std, slam.n_landmarks])

    def snapshot(self, label, lap, t, slam, save_state=False):
        """Log the whole map; with save_state, also the full mu/Sigma as state_<label>.npz."""
        for tag, (mean, cov, var_psi) in sorted(slam.landmarks().items()):
            self._row('map_snapshots', [
                label, lap, t, tag, *mean, cov[0, 0], cov[0, 1], cov[1, 1], var_psi,
                slam.sightings.get(tag, 0)])
        if save_state:
            ids = sorted(slam.index, key=slam.index.get)
            np.savez(self.path(f'state_{label}.npz'), mu=slam.mu, sigma=slam.sigma,
                     ids=np.array(ids, dtype=int),
                     offsets=np.array([slam.index[i] for i in ids], dtype=int),
                     landmark_dof=slam.dof)

    def truth_pose(self, t, pose):
        if self.truth is None:
            self._open('truth', ['t', 'x', 'y', 'theta'])
            self.truth = True
        self._row('truth', [t, *pose])

    # --- End of run ---------------------------------------------------------------------------

    def finish(self, slam, t, completed, extra=None):
        with open(self.path('map_final.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(MAP_HEADER)
            for tag, (mean, cov, var_psi) in sorted(slam.landmarks().items()):
                writer.writerow([fmt(v) for v in (
                    'final', -1, t, tag, *mean, cov[0, 0], cov[0, 1], cov[1, 1], var_psi,
                    slam.sightings.get(tag, 0))])
        self.snapshot('final', -1, t, slam, save_state=True)
        nis = np.array(slam.nis)
        self.info['summary'] = {
            'duration_s': round(t, 2),
            'completed_all_waypoints': bool(completed),
            'n_landmarks': slam.n_landmarks,
            'landmark_ids': sorted(slam.index),
            'counts': dict(slam.counts),
            'mean_nis': float(nis.mean()) if len(nis) else None,
            'nis_dof': slam.dof,
            'final_pose_start_frame': list(slam.pose),
            'final_pose_std': list(slam.std),
        }
        if extra:
            self.info['summary'].update(extra)
        self._write_info()
        for f in self.files.values():
            f.close()
        self.files = {}
