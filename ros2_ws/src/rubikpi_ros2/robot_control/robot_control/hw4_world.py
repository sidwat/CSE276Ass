"""
HW4 world model: arena walls, one box obstacle, a disk robot and the tag map (no ROS).

Everything is in the world frame of config/hw4_world.yaml (the HW3 ground-truth frame: x east,
y north, metres and radians, yaw counter-clockwise from +x). The robot is modelled as a disk of
radius robot_radius about its rotation centre, so a pose is collision-free iff the clearance of
its (x, y) is at least robot_radius, at every heading (an in-place turn sweeps exactly the disk).

The one function both planners use is World.clearance(x, y): the distance from a point to the
nearest of the five obstacles (four walls and the box). For an axis-aligned box of half-sizes
(hx, hy) centred at c it is  || max(|p - c| - h, 0) ||  (component-wise max): zero inside.
"""

import ast
from dataclasses import dataclass, field
import math
import operator

import numpy as np

from robot_control.waypoint_controller import wrap_angle

WALL_W, WALL_E, WALL_S, WALL_N, BOX = range(5)
LABELS = ('west wall', 'east wall', 'south wall', 'north wall', 'box')

# box face -> (outward normal x, outward normal y, yaw of a tag printed on it)
FACES = {
    'east': (1.0, 0.0, 0.0),
    'north': (0.0, 1.0, math.pi / 2.0),
    'west': (-1.0, 0.0, math.pi),
    'south': (0.0, -1.0, -math.pi / 2.0),
}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos}


def number(value):
    """
    Parse a YAML value as a float: a number, null (nan) or arithmetic text such as '3*pi/4'.

    Only + - * / ** and the names pi / deg (pi/180) are allowed, so '90*deg' also works.
    """
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    names = {'pi': math.pi, 'deg': math.pi / 180.0, 'nan': math.nan}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f'not a number: {value!r}')

    return float(ev(ast.parse(str(value).strip(), mode='eval')))


@dataclass
class Box:
    """Axis-aligned rectangular obstacle: centre (cx, cy), side lengths (sx, sy)."""

    cx: float
    cy: float
    sx: float
    sy: float

    @property
    def hx(self):
        return self.sx / 2.0

    @property
    def hy(self):
        return self.sy / 2.0

    def distance(self, x, y):
        """Euclidean distance to the box (0 inside); works on numpy arrays."""
        dx = np.maximum(np.abs(np.asarray(x, dtype=float) - self.cx) - self.hx, 0.0)
        dy = np.maximum(np.abs(np.asarray(y, dtype=float) - self.cy) - self.hy, 0.0)
        return np.hypot(dx, dy)

    def corners(self):
        """Corners counter-clockwise from north-east."""
        return [(self.cx + self.hx, self.cy + self.hy), (self.cx - self.hx, self.cy + self.hy),
                (self.cx - self.hx, self.cy - self.hy), (self.cx + self.hx, self.cy - self.hy)]

    def face_tag(self, face):
        """World pose (x, y, yaw) of a tag centred on a face, printed side outward."""
        nx, ny, yaw = FACES[face]
        return self.cx + nx * self.hx, self.cy + ny * self.hy, yaw


@dataclass
class World:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    box: Box
    robot_radius: float
    start: tuple                 # (x, y, theta)
    goal: tuple                  # (x, y, theta); theta nan = arrive along the last leg
    landmarks: dict = field(default_factory=dict)  # id -> (x, y, yaw)
    fast: dict = field(default_factory=dict)
    safe: dict = field(default_factory=dict)
    weighted: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    camera: dict = field(default_factory=dict)
    source: str = ''

    # ---- distances --------------------------------------------------------------------
    def distances(self, x, y):
        """Distance to each obstacle, stacked as (5, *shape) in LABELS order."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.stack([x - self.x_min, self.x_max - x, y - self.y_min, self.y_max - y,
                         self.box.distance(x, y)])

    def clearance(self, x, y):
        """Distance from (x, y) to the nearest obstacle (negative outside the walls)."""
        return np.min(self.distances(x, y), axis=0)

    def clearance_and_label(self, x, y):
        d = self.distances(x, y)
        return np.min(d, axis=0), np.argmin(d, axis=0)

    # ---- camera model -----------------------------------------------------------------
    def tags_in_view(self, pose):
        """Ids of the tags the camera model (self.camera) sees from pose (x, y, theta)."""
        c = self.camera
        return [t for t, lm in sorted(self.landmarks.items())
                if visible(self, pose, lm, c['half_fov'], c['max_range'], c['max_facing'])]

    def confirm_heading(self, x, y, preferred, step_deg=5.0):
        """
        Return the heading nearest `preferred` with a tag in view (preferred if none is).

        At a stop the controller settles and needs a tag sighting to confirm its heading; with
        none in view it SEARCHes (turns step by step), which costs tens of seconds.
        """
        n = int(round(180.0 / step_deg))
        for k in range(n + 1):
            for sign in ((1,) if k == 0 else (1, -1)):
                th = wrap_angle(preferred + sign * math.radians(k * step_deg))
                if self.tags_in_view((x, y, th)):
                    return th
        return preferred

    def segment_points(self, p, q, step=0.005):
        n = max(2, int(math.ceil(math.dist(p[:2], q[:2]) / step)) + 1)
        t = np.linspace(0.0, 1.0, n)
        return p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])

    def segment_clearance(self, p, q, step=0.005):
        """Smallest clearance along the straight segment p -> q (sampled every step metres)."""
        return float(np.min(self.clearance(*self.segment_points(p, q, step))))

    def segment_free(self, p, q, rho, step=0.005, tol=1e-6):
        """Check that every point of p -> q is at least rho from every obstacle."""
        return self.segment_clearance(p, q, step) >= rho - tol

    # ---- checks -----------------------------------------------------------------------
    def check(self):
        """Human-readable problems with the configuration (empty list if none)."""
        problems = []
        r = self.robot_radius
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            problems.append('arena: x_min < x_max and y_min < y_max are required')
        for name, pose in (('start', self.start), ('goal', self.goal)):
            c = float(self.clearance(pose[0], pose[1]))
            if c < r:
                problems.append(f'{name} ({pose[0]:.3f}, {pose[1]:.3f}) is only {c:.3f} m from '
                                f'the nearest obstacle: the robot body (radius {r:.3f}) '
                                f'would touch it')
            elif c < r + self.fast.get('margin', 0.0):
                problems.append(f'{name} clearance {c:.3f} m is inside robot_radius + fast '
                                f'margin; the fast plan will start/end with less margin')
        if self.box.sx <= 0 or self.box.sy <= 0:
            problems.append('obstacle size must be positive')
        if not self.landmarks:
            problems.append('no landmarks: the EKF cannot correct anything')
        for tag, (lx, ly, _) in sorted(self.landmarks.items()):
            if self.box.distance(lx, ly) > 0.02 and not (
                    self.x_min - 0.1 <= lx <= self.x_max + 0.1
                    and self.y_min - 0.1 <= ly <= self.y_max + 0.1):
                problems.append(f'tag {tag} at ({lx:.2f}, {ly:.2f}) is outside the arena')
        return problems

    def describe(self):
        b = self.box
        lines = [
            f'world file: {self.source or "(in memory)"}',
            f'arena x [{self.x_min:.3f}, {self.x_max:.3f}]  y [{self.y_min:.3f}, '
            f'{self.y_max:.3f}] m  ({self.x_max - self.x_min:.3f} x '
            f'{self.y_max - self.y_min:.3f})',
            f'box centre ({b.cx:.3f}, {b.cy:.3f}), size {b.sx:.3f} x {b.sy:.3f} m',
            f'robot radius {self.robot_radius:.3f} m',
            'start ({:.3f}, {:.3f}, {:.3f})   goal ({:.3f}, {:.3f}, {})'.format(
                *self.start, self.goal[0], self.goal[1],
                'free' if math.isnan(self.goal[2]) else f'{self.goal[2]:.3f}'),
            f'{len(self.landmarks)} tags: {sorted(self.landmarks)}',
        ]
        return '\n'.join(lines)


def visible(world, pose, landmark, half_fov, max_range, max_facing):
    """
    Return the exact (range, bearing, orientation) of a tag in view from pose, else None.

    The tag must be within range and the field of view, face the camera within max_facing, and
    the box must not block the line of sight (a tag on the box ends at its face).
    """
    x, y, th = pose
    lx, ly, lyaw = landmark
    r = math.hypot(lx - x, ly - y)
    bearing = wrap_angle(math.atan2(ly - y, lx - x) - th)
    orientation = wrap_angle(lyaw - th)
    facing = abs(wrap_angle(orientation - (bearing + math.pi)))
    if r > max_range or abs(bearing) > half_fov or facing > max_facing:
        return None
    n = max(2, int(r / 0.02))
    for k in range(1, n):
        t = k / n
        if r * (1.0 - t) < 0.03:
            break
        if world.box.distance(x + t * (lx - x), y + t * (ly - y)) <= 0.0:
            return None
    return r, bearing, orientation


DEFAULTS = {
    'fast': {'margin': 0.10, 'polygon_k': 2, 'pass_through': True, 'merge_corners': True},
    'safe': {'resolution': 0.02, 'min_margin': 0.05, 'rdp_epsilon': 0.03, 'min_leg': 0.15},
    'weighted': {'alpha': 0.3, 'resolution': 0.02, 'rdp_epsilon': 0.03, 'min_leg': 0.15,
                 'tradeoff_alphas': [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]},
    'timing': {'v': 0.25, 'w': 1.3, 't_stop': 1.5, 't_pass': 0.3},
    'camera': {'half_fov': 0.6, 'max_range': 2.5, 'max_facing': 1.05, 'confirm_heading': True},
}


def _section(data, name):
    merged = dict(DEFAULTS[name])
    merged.update(data.get(name) or {})
    for key, value in merged.items():
        if key == 'tradeoff_alphas':
            merged[key] = [number(v) for v in value]
        elif isinstance(value, bool):
            continue
        elif key == 'polygon_k':
            merged[key] = int(value)
        else:
            merged[key] = number(value)
    return merged


def world_from_dict(data, source=''):
    arena = data['arena']
    obstacle = data['obstacle']
    centre = [number(v) for v in obstacle['center']]
    size = obstacle['size']
    size = [number(size), number(size)] if not isinstance(size, (list, tuple)) else [
        number(v) for v in size]
    box = Box(centre[0], centre[1], size[0], size[1])

    landmarks = {}
    for face, tag in (obstacle.get('face_tags') or {}).items():
        if face not in FACES:
            raise ValueError(f'obstacle.face_tags: unknown face {face!r} (use {list(FACES)})')
        if tag is not None:
            landmarks[int(tag)] = box.face_tag(face)
    for row in data.get('landmarks') or []:
        if len(row) != 4:
            raise ValueError(f'landmarks: expected [id, x, y, yaw], got {row!r}')
        landmarks[int(row[0])] = tuple(number(v) for v in row[1:])  # measured values win

    start = tuple(number(v) for v in data['start'])
    goal = list(data['goal'])
    if len(goal) == 2:
        goal.append(None)
    goal = tuple(number(v) for v in goal)
    if len(start) != 3 or math.isnan(start[2]):
        raise ValueError('start must be [x, y, theta] with a heading')
    return World(
        x_min=number(arena['x_min']), x_max=number(arena['x_max']),
        y_min=number(arena['y_min']), y_max=number(arena['y_max']),
        box=box, robot_radius=number(data['robot']['radius']),
        start=start, goal=goal, landmarks=landmarks,
        fast=_section(data, 'fast'), safe=_section(data, 'safe'),
        weighted=_section(data, 'weighted'), timing=_section(data, 'timing'),
        camera=_section(data, 'camera'), source=source)


def load_world(path):
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return world_from_dict(data, source=str(path))


def write_landmarks(world, path):
    """Write the tag map in the 'id, x, y, yaw' format velocity_control reads."""
    with open(path, 'w') as f:
        f.write(f'# HW4 tag map, world frame, generated from {world.source}\n'
                '#   id, x [m], y [m], yaw [rad]\n')
        for tag, (x, y, yaw) in sorted(world.landmarks.items()):
            f.write(f'{tag}, {x:.4f}, {y:.4f}, {yaw:.4f}\n')
