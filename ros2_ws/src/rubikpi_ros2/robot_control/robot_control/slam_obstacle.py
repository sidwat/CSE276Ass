"""
HW4 extra credit: map the obstacle with EKF-SLAM and plan on the map as it is built (no ROS).

What the robot knows in advance: the arena walls (geometry), its start pose, the goal, and only a
PRIOR on the obstacle: an axis-aligned box near slam.prior.center (within center_tolerance)
whose sides are at most max_size. Where the box really is comes from the tags on its faces,
which EKF-SLAM (robot_control/slam.py) maps while the robot drives.

Obstacle estimate. Tags are split into wall tags (within wall_band of a wall) and obstacle tags.
Each obstacle tag is assigned to a box face by its mapped orientation psi (the direction its
printed face points: east 0, north pi/2, west pi, south -pi/2). The only assumptions are that
each tag (half-width h) lies entirely on its face and that no side is longer than s_max. A face
with tags at lateral positions u_1..u_n then gives

  * its own plane:  e.g. east face  x_E  ->  x_max = x_E   and   x_min >= x_E - s_max
  * the face spans every tag:  y_min >= max(u) + h - s_max   and   y_max <= min(u) - h + s_max

Every such statement bounds one side of the box; the estimate is their intersection with the
prior region, each bound widened by k_sigma SLAM standard deviations plus bound_margin
(systematic map error). It is therefore never larger than the prior region: a tag that is still
uncertain only adds a weak constraint. It contains the box if the SLAM errors stay within that
widening, and it shrinks as more faces and more tags are mapped: two tags mounted flush with the
two ends of a face pin that side to within the widening.

Replanning. SlamPlanner plans with the HW4 planners (fast: visibility graph + A*, safe: GVD +
Dijkstra) on a copy of the world whose obstacle is the current estimate, starting from the
current SLAM pose. It replans when the estimate has changed by more than replan_threshold (at
most every replan_interval seconds) and immediately if the remaining path would bring the robot
body into the new estimate.
"""

from dataclasses import dataclass, field, replace
import math

import numpy as np

from robot_control.gvd_planner import plan_safe
from robot_control.path_tools import path_length, PlanningError, Timing, to_waypoints
from robot_control.visibility_planner import plan_fast
from robot_control.waypoint_controller import wrap_angle
from robot_control.world_model import Box, WorldError

FACE_YAW = {'east': 0.0, 'north': math.pi / 2, 'west': math.pi, 'south': -math.pi / 2}


@dataclass
class SlamConfig:
    prior_center: tuple = (1.5, 1.5)   # m, where the obstacle is expected
    center_tolerance: float = 0.25     # m, how far its centre may be from prior_center
    max_size: tuple = (0.61, 0.61)     # m, largest possible side lengths (2 ft)
    k_sigma: float = 2.0               # widen every face bound by this many SLAM stds
    wall_band: float = 0.30            # m, tags this close to a wall are wall tags
    face_angle_tolerance: float = 0.6  # rad, |psi - face yaw| allowed when assigning a face
    tag_half_width: float = 0.0865     # m, half the printed tag (0.173 m): it lies on the face
    bound_margin: float = 0.03         # m added to every bound (map error SLAM's sigma misses)
    min_size: float = 0.05             # m: bounds closer than this conflict -> use the prior
    max_psi_std: float = 0.25          # rad: a tag is used once its face is this certain
    replan_threshold: float = 0.03     # m change of any box side that triggers a replan
    replan_interval: float = 2.0       # s between map-triggered replans
    freeze_radius: float = 0.30        # m from the goal: no more replanning
    min_leg: float = 0.12              # m: merge shorter legs (controller replan_distance 0.10)
    retry_interval: float = 0.5        # s between attempts after a failed (re)plan
    prior_landmarks: str = 'none'      # 'none': map every tag; 'walls': wall tags known
    prior_std: tuple = (0.02, 0.02, 0.05)  # std of known wall tags (m, m, rad)

    @classmethod
    def from_world(cls, world):
        s = dict(world.slam or {})
        prior = dict(s.get('prior') or {})
        cfg = cls()
        if 'center' in prior:
            cfg.prior_center = tuple(float(v) for v in prior['center'])
        if 'center_tolerance' in prior:
            cfg.center_tolerance = float(prior['center_tolerance'])
        if 'max_size' in prior:
            m = prior['max_size']
            cfg.max_size = (float(m), float(m)) if not isinstance(m, (list, tuple)) else \
                tuple(float(v) for v in m)
        for name in ('k_sigma', 'wall_band', 'face_angle_tolerance', 'min_size', 'max_psi_std',
                     'tag_half_width', 'bound_margin',
                     'replan_threshold', 'replan_interval', 'freeze_radius', 'min_leg',
                     'retry_interval'):
            if name in s:
                setattr(cfg, name, float(s[name]))
        if 'prior_landmarks' in s:
            cfg.prior_landmarks = str(s['prior_landmarks'])
        if 'prior_std' in s:
            cfg.prior_std = tuple(float(v) for v in s['prior_std'])
        if cfg.prior_landmarks not in ('none', 'walls'):
            raise WorldError(f"slam.prior_landmarks must be 'none' or 'walls', not "
                             f'{cfg.prior_landmarks!r}')
        return cfg


@dataclass
class FaceFit:
    face: str
    coord: float          # position of the face plane (x for east/west, y for north/south)
    std: float
    lateral_min: float    # extreme tag positions along the face (y for east/west, x for N/S)
    lateral_max: float
    lateral_std: float    # largest std of a tag along the face
    tags: list = field(default_factory=list)


@dataclass
class ObstacleEstimate:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    faces: dict = field(default_factory=dict)       # face name -> FaceFit
    tags: dict = field(default_factory=dict)        # obstacle tag id -> face name (or None)
    yaw_offset: float = float('nan')                # mean tag yaw residual (box rotation)
    source: str = 'prior'                           # 'prior' or 'slam'
    conflicts: list = field(default_factory=list)   # axes whose tag bounds contradicted

    @property
    def box(self):
        return Box((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0,
                   (self.x_max - self.x_min) / 2.0, (self.y_max - self.y_min) / 2.0)

    def bounds(self):
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def change(self, other):
        """Largest move of any side (m) relative to another estimate (inf if None)."""
        if other is None:
            return math.inf
        return max(abs(a - b) for a, b in zip(self.bounds(), other.bounds()))

    def to_dict(self):
        return {
            'bounds': [self.x_min, self.x_max, self.y_min, self.y_max],
            'faces': {f: {'coord': ff.coord, 'std': ff.std, 'lateral_min': ff.lateral_min,
                          'lateral_max': ff.lateral_max, 'lateral_std': ff.lateral_std,
                          'tags': ff.tags}
                      for f, ff in self.faces.items()},
            'tags': {str(k): v for k, v in self.tags.items()},
            'yaw_offset': None if math.isnan(self.yaw_offset) else self.yaw_offset,
            'source': self.source,
            'conflicts': list(self.conflicts),
        }


def prior_estimate(cfg):
    cx, cy = cfg.prior_center
    hx = cfg.max_size[0] / 2.0 + cfg.center_tolerance
    hy = cfg.max_size[1] / 2.0 + cfg.center_tolerance
    return ObstacleEstimate(cx - hx, cx + hx, cy - hy, cy + hy, source='prior')


def is_wall_tag(world, x, y, band):
    return min(x, world.size - x, y, world.size - y) < band


def split_landmarks(landmarks, world, band):
    """
    Split SLAM landmarks into (wall, obstacle) dicts.

    landmarks is EKFSLAM.landmarks(): id -> ((x, y, psi), 2x2 position covariance, var psi).
    """
    wall, obstacle = {}, {}
    for tag, lm in landmarks.items():
        (x, y, _), _, _ = lm
        (wall if is_wall_tag(world, x, y, band) else obstacle)[tag] = lm
    return wall, obstacle


def assign_face(psi, x, y, cfg):
    """Face a tag is on, from its orientation (or its position if psi is unknown)."""
    if math.isfinite(psi):
        face, err = min(((f, abs(wrap_angle(psi - yaw))) for f, yaw in FACE_YAW.items()),
                        key=lambda t: t[1])
        return face if err <= cfg.face_angle_tolerance else None
    dx, dy = x - cfg.prior_center[0], y - cfg.prior_center[1]
    if abs(dx) >= abs(dy):
        return 'east' if dx > 0 else 'west'
    return 'north' if dy > 0 else 'south'


def _weighted(values, variances):
    w = 1.0 / np.maximum(np.asarray(variances, dtype=float), 1e-8)
    return float(np.sum(w * np.asarray(values)) / np.sum(w)), float(math.sqrt(1.0 / np.sum(w)))


def fit_faces(obstacle, cfg):
    """Group obstacle tags by face; returns ({face: FaceFit}, {tag: face}, yaw offset)."""
    groups, tag_face, residuals = {}, {}, []
    for tag, ((x, y, psi), cov, var_psi) in obstacle.items():
        if math.isfinite(psi) and var_psi > cfg.max_psi_std ** 2:
            tag_face[tag] = None  # its orientation (hence its face) is not yet certain
            continue
        face = assign_face(psi, x, y, cfg)
        tag_face[tag] = face
        if face is None:
            continue
        groups.setdefault(face, []).append((tag, x, y, cov))
        if math.isfinite(psi):
            residuals.append(wrap_angle(psi - FACE_YAW[face]))
    faces = {}
    for face, items in groups.items():
        xs = [it[1] for it in items]
        ys = [it[2] for it in items]
        vx = [it[3][0, 0] for it in items]
        vy = [it[3][1, 1] for it in items]
        if face in ('east', 'west'):
            coord, std = _weighted(xs, vx)
            lat, lat_var = ys, vy
        else:
            coord, std = _weighted(ys, vy)
            lat, lat_var = xs, vx
        faces[face] = FaceFit(face, coord, std, float(min(lat)), float(max(lat)),
                              float(math.sqrt(max(lat_var))), [it[0] for it in items])
    yaw = float(np.mean(residuals)) if residuals else float('nan')
    return faces, tag_face, yaw


def estimate_obstacle(landmarks, world, cfg):
    """Conservative axis-aligned box from the mapped obstacle tags (the prior if none)."""
    _, obstacle = split_landmarks(landmarks, world, cfg.wall_band)
    faces, tag_face, yaw = fit_faces(obstacle, cfg)
    prior = prior_estimate(cfg)
    if not faces:
        prior.tags = tag_face
        return prior
    k, h, m = cfg.k_sigma, cfg.tag_half_width, cfg.bound_margin
    sx, sy = cfg.max_size
    up = {'x': [], 'y': []}
    lo = {'x': [], 'y': []}
    for face, f in faces.items():
        e, el = k * f.std + m, k * f.lateral_std + m
        if face == 'east':
            up['x'].append(f.coord + e)
            lo['x'].append(f.coord - sx - e)
        elif face == 'west':
            lo['x'].append(f.coord - e)
            up['x'].append(f.coord + sx + e)
        elif face == 'north':
            up['y'].append(f.coord + e)
            lo['y'].append(f.coord - sy - e)
        else:
            lo['y'].append(f.coord - e)
            up['y'].append(f.coord + sy + e)
        # every tag lies on the face, and the face is at most `size` long
        axis, size = ('y', sy) if face in ('east', 'west') else ('x', sx)
        up[axis].append(f.lateral_min - h + size + el)
        lo[axis].append(f.lateral_max + h - size - el)
    bounds, conflicts = [], []
    for axis, (p_lo, p_hi) in (('x', (prior.x_min, prior.x_max)),
                               ('y', (prior.y_min, prior.y_max))):
        a = max(lo[axis] + [p_lo])  # the prior region is a constraint too
        b = min(up[axis] + [p_hi])
        if b - a < cfg.min_size:
            # contradictory tags (e.g. a flipped orientation put a tag on the wrong face):
            # trust none of them on this axis and fall back to the prior (conservative)
            a, b = p_lo, p_hi
            conflicts.append(axis)
        bounds += [a, b]
    return ObstacleEstimate(*bounds, faces=faces, tags=tag_face, yaw_offset=yaw, source='slam',
                            conflicts=conflicts)


class SlamPlanner:
    """Plan on the SLAM obstacle estimate and decide when to replan (see the module doc)."""

    def __init__(self, world, mode, cfg=None, timing=None):
        if mode not in ('fast', 'safe'):
            raise ValueError("mode must be 'fast' or 'safe'")
        self.world = world          # world.boxes (ground truth) is never used for planning
        self.mode = mode
        self.cfg = cfg or SlamConfig.from_world(world)
        self.timing = timing or Timing.from_world(world)
        self.pass_through = bool(world.planner.get('pass_through', True))
        self.margin = float(dict(world.planner.get('fast') or {}).get('margin', 0.08))
        self.estimate = None        # estimate the current plan was made on
        self.latest = None          # most recent estimate (every update)
        self.plan = None
        self.waypoints = None
        self.t_last = -math.inf
        self.t_failed = -math.inf   # last failed attempt (retries are rate-limited)
        self.history = []           # one dict per (re)plan or failed attempt

    # --- worlds --------------------------------------------------------------------------------

    def planning_world(self, estimate, start=None):
        """Return the world as the robot believes it: the estimated box, not the true one."""
        w = replace(self.world, boxes=[estimate.box])
        return replace(w, start=tuple(start)) if start is not None else w

    def guard_world(self):
        """World for the clearance guard: the latest estimate (the prior before any)."""
        return self.planning_world(self.latest or prior_estimate(self.cfg))

    # --- replanning ----------------------------------------------------------------------------

    def path_clear(self, estimate, pose, remaining):
        """Check the rest of the current path keeps the robot body out of the estimate."""
        pts = [tuple(pose[:2])] + [tuple(w[:2]) for w in remaining]
        box = estimate.box
        for p, q in zip(pts[:-1], pts[1:]):
            n = max(2, int(math.ceil(math.dist(p, q) / 0.01)) + 1)
            t = np.linspace(0.0, 1.0, n)
            d = box.distance(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
            if float(np.min(d)) < self.world.robot_radius:
                return False
        return True

    def update(self, t, pose, landmarks, remaining=None, force=False):
        """
        Re-estimate the obstacle; return new waypoints if a (re)plan was made, else None.

        t is the run time (s), pose the SLAM pose (x, y, theta), landmarks
        EKFSLAM.landmarks(), remaining the waypoints the controller still has to reach.
        """
        est = estimate_obstacle(landmarks, self.world, self.cfg)
        self.latest = est
        if self.plan is None:
            reason = 'initial'
        elif force:
            reason = 'forced'
        elif math.dist(pose[:2], self.world.goal[:2]) < self.cfg.freeze_radius:
            return None
        elif remaining is not None and not self.path_clear(est, pose, remaining):
            reason = 'blocked'
        elif (est.change(self.estimate) > self.cfg.replan_threshold
              and t - self.t_last >= self.cfg.replan_interval):
            reason = 'map'
        else:
            return None
        if t - self.t_failed < self.cfg.retry_interval:
            return None
        plan, note = self._plan(est, pose)
        entry = {'t': t, 'reason': reason, 'pose': list(pose), 'estimate': est.to_dict(),
                 'n_landmarks': len(landmarks), 'ok': plan is not None, 'note': note}
        if plan is None:
            self.t_failed = t
            self.history.append(entry)
            return None
        w = self.planning_world(est)
        pruned = prune_short_legs(w, plan.path, self.cfg.min_leg)
        if len(pruned) != len(plan.path):
            plan = replace(plan, path=pruned)
        self.estimate, self.plan, self.t_last = est, plan, t
        self.waypoints = to_waypoints(plan.path, self.world.goal[2], self.pass_through)
        dmin, dmean, _ = w.path_clearance(plan.path)
        entry.update({'path': [list(map(float, p[:2])) for p in plan.path],
                      'waypoints': [[x, y, th if math.isfinite(th) else None]
                                    for x, y, th in self.waypoints],
                      'length': path_length(plan.path),
                      'min_body_gap_est': dmin - self.world.robot_radius,
                      'mean_clearance_est': dmean})
        self.history.append(entry)
        return self.waypoints

    def _plan(self, est, pose):
        """Plan from pose on the estimate; the fast planner relaxes its margin if it must."""
        world = self.planning_world(est, pose)
        if self.mode == 'safe':
            try:
                return plan_safe(world, self.timing), None
            except (WorldError, PlanningError) as e:
                return None, f'safe planner failed: {e}'
        last = None
        for margin in (self.margin, self.margin / 2.0, 0.0):
            try:
                note = None if margin == self.margin else f'margin relaxed to {margin:.3f} m'
                return plan_fast(world, self.timing, margin=margin), note
            except (WorldError, PlanningError) as e:
                last = e
        return None, f'fast planner failed: {last}'


def prune_short_legs(world, path, min_leg):
    """
    Drop interior vertices that leave a leg shorter than min_leg.

    The controller cannot drive a leg shorter than its replan_distance (it only turns), so a
    vertex a few cm from its neighbour (typically from the GVD retraction next to the start or
    the goal) would leave the robot short of it. A vertex is dropped only if the merged segment
    keeps the clearance of the two legs it replaces (minus 1 cm).
    """
    path = [tuple(p[:2]) for p in path]
    changed = True
    while changed and len(path) > 2:
        changed = False
        for i in range(1, len(path) - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            if min(math.dist(a, b), math.dist(b, c)) >= min_leg:
                continue
            need = min(world.segment_clearance(a, b), world.segment_clearance(b, c)) - 0.01
            if world.segment_clearance(a, c) >= need:
                del path[i]
                changed = True
                break
    return path


def seed_known_walls(slam, world, cfg):
    """With slam.prior_landmarks 'walls', add the wall tags of the world file as known."""
    added = []
    if cfg.prior_landmarks != 'walls':
        return added
    for tag, (x, y, yaw) in sorted(world.landmarks.items(), key=lambda kv: str(kv[0])):
        if is_wall_tag(world, x, y, cfg.wall_band):
            slam.add_known_landmark(tag, (x, y, yaw), cfg.prior_std)
            added.append(tag)
    return added
