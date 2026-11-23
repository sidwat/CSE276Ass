from dataclasses import dataclass, field
import math

import numpy as np

# Obstacle labels returned by World.clearance: the four walls, then the boxes (4, 5, ...)
WEST, EAST, SOUTH, NORTH = 0, 1, 2, 3
WALL_NAMES = ('west wall', 'east wall', 'south wall', 'north wall')


class WorldError(ValueError):
    """The world file is invalid or the start/goal is not collision-free."""


@dataclass
class Box:
    """Axis-aligned box: centre (cx, cy) and half-sizes (hx, hy), metres."""

    cx: float
    cy: float
    hx: float
    hy: float

    def distance(self, x, y):
        """Euclidean distance from point(s) to the box (0 inside); works on numpy arrays."""
        dx = np.maximum(np.abs(np.asarray(x, dtype=float) - self.cx) - self.hx, 0.0)
        dy = np.maximum(np.abs(np.asarray(y, dtype=float) - self.cy) - self.hy, 0.0)
        return np.hypot(dx, dy)

    def corners(self):
        """Corners counter-clockwise from north-east."""
        return [(self.cx + self.hx, self.cy + self.hy), (self.cx - self.hx, self.cy + self.hy),
                (self.cx - self.hx, self.cy - self.hy), (self.cx + self.hx, self.cy - self.hy)]


@dataclass
class World:
    size: float
    boxes: list
    robot_radius: float
    start: tuple                     # (x, y, theta)
    goal: tuple                      # (x, y, theta or nan: keep the arrival heading)
    landmarks: dict = field(default_factory=dict)   # id or family:id -> (x, y, yaw)
    planner: dict = field(default_factory=dict)     # raw 'planner' section of the yaml
    source: str = ''

    # --- clearance --------------------------------------------------------------------------

    def clearance(self, x, y):
        """
        Return (distance to the nearest obstacle, label of that obstacle) for point(s).

        Labels: 0-3 the west/east/south/north walls, 4 + k box k. Inside a box the distance
        is 0; outside the arena it is negative.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = [x, self.size - x, y, self.size - y] + [b.distance(x, y) for b in self.boxes]
        d = np.stack(np.broadcast_arrays(*d))
        return d.min(axis=0), d.argmin(axis=0)

    def clearance_at(self, x, y):
        """Clearance of one point, as a float."""
        d, _ = self.clearance(x, y)
        return float(d)

    def label_name(self, label):
        return WALL_NAMES[label] if label < 4 else f'box {label - 4}'

    def segment_clearance(self, p, q, step=0.005):
        """Smallest clearance along the straight segment p -> q (sampled every step metres)."""
        n = max(2, int(math.ceil(math.dist(p[:2], q[:2]) / step)) + 1)
        t = np.linspace(0.0, 1.0, n)
        d, _ = self.clearance(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
        return float(d.min())

    def segment_free(self, p, q, rho, step=0.005):
        """Check that the robot centre can move along p -> q keeping clearance >= rho."""
        return self.segment_clearance(p, q, step) >= rho - 1e-9

    def path_clearance(self, path, step=0.005):
        """Clearance sampled along a polyline: (min, mean, samples)."""
        samples = []
        for p, q in zip(path[:-1], path[1:]):
            n = max(2, int(math.ceil(math.dist(p[:2], q[:2]) / step)) + 1)
            t = np.linspace(0.0, 1.0, n)
            d, _ = self.clearance(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
            samples.append(d)
        d = np.concatenate(samples) if samples else np.array([self.clearance_at(*path[0][:2])])
        return float(d.min()), float(d.mean()), d

    # --- configuration-space obstacles ------------------------------------------------------

    def c_obstacle(self, box, rho, segments=2):
        """
        Polygon (counter-clockwise) containing the box grown by a disk of radius rho.

        The Minkowski sum of a box and a disk is a rounded box. Each quarter-circle corner is
        replaced by its circumscribed polygon with `segments` sides: tangent points every
        pi/2/segments, vertices half-way between them at radius rho / cos(pi/4/segments). The
        polygon therefore contains the true C-obstacle and bulges at most
        rho * (1 / cos(pi/4/segments) - 1) beyond it.
        """
        step = (math.pi / 2.0) / segments
        r = rho / math.cos(step / 2.0)
        pts = []
        for (cx, cy), a0 in zip(box.corners(), (0.0, math.pi / 2, math.pi, 1.5 * math.pi)):
            for i in range(segments):
                a = a0 + (i + 0.5) * step
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts

    def check_endpoints(self, rho):
        """Raise WorldError if the start or goal is closer than rho to an obstacle."""
        for name, p in (('start', self.start), ('goal', self.goal)):
            d, lab = self.clearance(p[0], p[1])
            if float(d) < rho:
                raise WorldError(
                    f'{name} ({p[0]:.3f}, {p[1]:.3f}) is {float(d):.3f} m from the '
                    f'{self.label_name(int(lab))}, closer than the required {rho:.3f} m '
                    f'(robot radius + margin). Move it or reduce the margin in {self.source}.')

    def planner_section(self, name):
        return dict(self.planner.get(name) or {})


def _pose(value, name, allow_null_theta=False):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise WorldError(f'{name} must be [x, y, theta], got {value!r}')
    x, y, th = value
    if th is None:
        if not allow_null_theta:
            raise WorldError(f'{name} theta must be a number')
        th = float('nan')
    return float(x), float(y), float(th)


def world_from_dict(data, source=''):
    """Build a World from the parsed yaml (see config/hw4_world.yaml)."""
    try:
        size = float(data['arena']['size'])
        boxes = []
        for i, ob in enumerate(data.get('obstacles') or []):
            (cx, cy), (sx, sy) = ob['center'], ob['size']
            if sx <= 0 or sy <= 0:
                raise WorldError(f'obstacle {i}: size must be positive')
            boxes.append(Box(float(cx), float(cy), float(sx) / 2.0, float(sy) / 2.0))
        radius = float(data['robot']['radius'])
        start = _pose(data['start'], 'start')
        goal = _pose(data['goal'], 'goal', allow_null_theta=True)
        landmarks = {}
        for lm in data.get('landmarks') or []:
            tag = int(lm['id'])
            family = lm.get('family', '36h11')
            if family != '36h11':
                tag = f'{family}:{tag}'
            if tag in landmarks:
                raise WorldError(f'landmark id {tag} is listed twice')
            landmarks[tag] = (float(lm['x']), float(lm['y']), float(lm['yaw']))
    except (KeyError, TypeError, ValueError) as e:
        if isinstance(e, WorldError):
            raise
        raise WorldError(f'{source or "world"}: malformed entry ({e!r})') from e
    world = World(size, boxes, radius, start, goal, landmarks, dict(data.get('planner') or {}),
                  source)
    for name, p in (('start', start), ('goal', goal)):
        if not (0.0 < p[0] < size and 0.0 < p[1] < size):
            raise WorldError(f'{name} ({p[0]}, {p[1]}) is outside the {size} m arena')
        for k, b in enumerate(boxes):
            if float(b.distance(p[0], p[1])) == 0.0:
                raise WorldError(f'{name} is inside obstacle {k}')
    return world


def load_world(path):
    """Read a world yaml file (config/hw4_world.yaml format)."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise WorldError(f'{path}: not a yaml mapping')
    return world_from_dict(data, source=str(path))
