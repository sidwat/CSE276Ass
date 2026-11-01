"""Associate detections with capture-time TF; validate and fuse world-pose measurements.

No ROS node/executor dependencies: lookup is injected, making delayed TF testable.
All image geometry, calibration and PnP belong to apriltag_ros.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np

from hw2_navigation.core import base_pose, stamp_ns, transform


def transform_matrix(msg):
    """Convert geometry_msgs/Transform to T_parent_child, validating its quaternion."""
    t, q = msg.translation, msg.rotation
    xyz = np.array([t.x, t.y, t.z], dtype=float)
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(quat)
    if not np.isfinite(xyz).all() or not np.isfinite(quat).all() or abs(norm-1) > 1e-3:
        raise ValueError('Invalid TF translation or quaternion')
    x, y, z, w = quat/norm
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = xyz
    return matrix


@dataclass
class VisualResult:
    reason: str
    accepted: bool = False
    rejected: int = 0
    age: float = 0.0


class VisualLocalizer:
    def __init__(self, cfg, tags, base_camera):
        self.cfg, self.tags, self.base_camera = cfg, tags, base_camera
        self.pending = deque(maxlen=10)
        self.last_received = -1
        self.reason = 'waiting_for_measurements'

    def enqueue(self, msg, now_ns):
        stamp = stamp_ns(msg.header.stamp)
        age = (now_ns-stamp)*1e-9
        if age < -.02 or age > self.cfg['vision']['max_age'] or stamp <= self.last_received:
            self.reason = 'stale_or_out_of_order_image'
            return self.reason
        self.last_received = stamp
        self.pending.append(msg)
        self.reason = 'waiting_for_tag_tf'
        return self.reason

    def process(self, now_ns, lookup, pose_filter):
        """Process at most one image per tick; no waits and no use of latest-time TF."""
        if not self.pending:
            return VisualResult(self.reason)
        msg = self.pending[0]
        age = (now_ns-stamp_ns(msg.header.stamp))*1e-9
        if age < -.02 or age > self.cfg['vision']['max_age']:
            self.pending.popleft()
            self.reason = 'expired_waiting_for_tag_tf'
            return VisualResult(self.reason)
        if self.base_camera is None:
            self.pending.popleft()
            self.reason = 'missing_camera_mount'
            return VisualResult(self.reason)
        candidates, missing = [], False
        for d in msg.detections:
            tag = self.tags.get(d.id)
            if (tag is None or d.family != self.cfg['_detector']['family'] or
                    d.hamming > self.cfg['_detector']['max_hamming'] or not np.isfinite(d.decision_margin) or
                    d.decision_margin < self.cfg['vision']['min_margin']):
                continue
            frame = self.cfg['_detector']['frames'].get(d.id, f'{d.family}:{d.id}')
            tf = lookup(msg.header.frame_id, frame, msg.header.stamp)
            if tf is None:
                missing = True
                continue
            # Reject a cached/latest transform presented as a capture-time observation.
            if (stamp_ns(tf.header.stamp) != stamp_ns(msg.header.stamp) or
                    tf.header.frame_id != msg.header.frame_id or tf.child_frame_id != frame):
                continue
            try:
                ct = transform_matrix(tf.transform)
            except ValueError:
                continue
            distance = float(np.linalg.norm(ct[:3, 3]))
            if ct[2, 3] <= 0 or distance > self.cfg['vision']['max_range']:
                continue
            z = base_pose(transform(tag['xyz'], tag['rpy']), ct, self.base_camera)
            std = np.asarray(self.cfg['filter']['visual_std'])*max(1.0, distance)
            candidates.append((distance, -float(d.decision_margin), z, np.diag(std**2)))
        # Allow bounded time for same-image TFs; a missing tag must not block a valid one.
        if missing and (not candidates or age < self.cfg['vision'].get('tf_wait', .05)):
            self.reason = 'waiting_for_tag_tf'
            return VisualResult(self.reason)
        self.pending.popleft()
        rejected = 0
        for _, _, z, covariance in sorted(candidates, key=lambda item: item[:2]):
            ok, nis = pose_filter.correct(z, covariance, self.cfg['filter']['nis_gate'])
            if ok:
                self.reason = f'visual_accepted_nis={nis:.2f}'
                return VisualResult(self.reason, True, rejected, max(0.0, age))
            rejected += 1
        self.reason = 'no_valid_visual_update'
        return VisualResult(self.reason, rejected=rejected)
