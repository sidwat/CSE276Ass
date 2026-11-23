"""Path post-processing shared by the HW4 planners: simplify, waypoints, time, metrics (no ROS)."""

from dataclasses import dataclass, field
import json
import math

import numpy as np

from robot_control.waypoint_controller import wrap_angle


class PlanningError(RuntimeError):
    """No collision-free path exists for the requested planner and settings."""


@dataclass
class Timing:
    v: float = 0.25        # m/s cruise speed
    w: float = 1.3         # rad/s turn-in-place rate
    t_stop: float = 1.5    # s per settling waypoint (settle_time + pause_at_waypoint)
    t_pass: float = 0.3    # s per pass-through waypoint

    @classmethod
    def from_world(cls, world):
        sec = dict(world.planner.get('timing') or {})
        return cls(**{k: float(v) for k, v in sec.items() if k in vars(cls())})


@dataclass
class Plan:
    """Output of a planner. path[0] is the start, path[-1] the goal (x, y)."""

    mode: str
    path: list
    rho: float                       # clearance the plan guarantees for the robot centre
    raw_path: list = field(default_factory=list)   # before merging / simplification
    extras: dict = field(default_factory=dict)      # representation, for plots


def path_length(path):
    return float(sum(math.dist(p[:2], q[:2]) for p, q in zip(path[:-1], path[1:])))


def segment_headings(path):
    return [math.atan2(q[1] - p[1], q[0] - p[0]) for p, q in zip(path[:-1], path[1:])]


def total_turn(path, start_heading, goal_heading=float('nan')):
    """Sum of |heading changes| (rad): start heading -> each leg -> final heading."""
    heading, turn = start_heading, 0.0
    for h in segment_headings(path):
        turn += abs(wrap_angle(h - heading))
        heading = h
    if math.isfinite(goal_heading):
        turn += abs(wrap_angle(goal_heading - heading))
    return turn


def perpendicular_distances(points, a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    P = np.asarray(points, dtype=float)
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-12:
        return np.linalg.norm(P - a, axis=1)
    return np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / n


def rdp(points, eps, segment_ok=None):
    """
    Ramer-Douglas-Peucker simplification of a polyline.

    Keeps the point farthest from the chord while it is more than eps away, recursively, so
    every dropped point lies within eps of the result. If segment_ok(p, q) is given, a chord
    that fails it is split too (e.g. one that would cut too close to an obstacle).
    """
    points = [tuple(p) for p in points]
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    d = perpendicular_distances(points[1:-1], a, b)
    i = int(np.argmax(d)) + 1
    if d[i - 1] > eps or (segment_ok is not None and not segment_ok(a, b)):
        left = rdp(points[:i + 1], eps, segment_ok)
        right = rdp(points[i:], eps, segment_ok)
        return left[:-1] + right
    return [a, b]


def to_waypoints(path, goal_theta=float('nan'), pass_through=True):
    """
    Turn a path (start first) into controller waypoints (x, y, theta), start excluded.

    Each intermediate vertex gets the heading of the leg that arrives there, or nan when
    pass_through is set (the controller then drives through it without a final turn or settle).
    The goal gets goal_theta, or the arrival heading if goal_theta is nan.
    """
    heads = segment_headings(path)
    out = []
    for i in range(1, len(path)):
        last = i == len(path) - 1
        if last:
            theta = goal_theta if math.isfinite(goal_theta) else heads[-1]
        else:
            theta = float('nan') if pass_through else heads[i - 1]
        out.append((float(path[i][0]), float(path[i][1]), float(theta)))
    return out


def estimate_time(path, start_heading, timing, goal_theta=float('nan'), pass_through=True):
    """Travel-time model: sum(l / v) + sum(|turn| / w) + stops * t_stop + passes * t_pass."""
    n_wp = len(path) - 1
    stops = 1 if pass_through else n_wp
    passes = n_wp - stops
    return (path_length(path) / timing.v
            + total_turn(path, start_heading, goal_theta) / timing.w
            + stops * timing.t_stop + passes * timing.t_pass)


def metrics(world, path, timing, pass_through=True):
    """Length, clearance of the robot centre and of its body, turning, waypoints, time."""
    dmin, dmean, _ = world.path_clearance(path)
    return {
        'length': path_length(path),
        'min_clearance': dmin,
        'mean_clearance': dmean,
        'min_body_gap': dmin - world.robot_radius,
        'turn': total_turn(path, world.start[2], world.goal[2]),
        'waypoints': len(path) - 1,
        'time_stop': estimate_time(path, world.start[2], timing, world.goal[2], False),
        'time_pass': estimate_time(path, world.start[2], timing, world.goal[2], True),
        'time': estimate_time(path, world.start[2], timing, world.goal[2], pass_through),
    }


def write_waypoints(path_out, waypoints, header_lines=()):
    """Write 'x, y, theta' lines (theta 'nan' = pass-through) for velocity_control."""
    with open(path_out, 'w') as f:
        for line in header_lines:
            f.write(f'# {line}\n')
        for x, y, th in waypoints:
            theta = 'nan' if not math.isfinite(th) else f'{th:.4f}'
            f.write(f'{x:.4f}, {y:.4f}, {theta}\n')


def save_plan_json(path_out, plan, world, timing, pass_through=True):
    """Plan, waypoints and metrics as JSON (hw4 eval reads it back)."""
    data = {
        'mode': plan.mode,
        'world': world.source,
        'start': list(world.start),
        'goal': [world.goal[0], world.goal[1],
                 world.goal[2] if math.isfinite(world.goal[2]) else None],
        'robot_radius': world.robot_radius,
        'rho': plan.rho,
        'path': [list(map(float, p[:2])) for p in plan.path],
        'waypoints': [[x, y, th if math.isfinite(th) else None]
                      for x, y, th in to_waypoints(plan.path, world.goal[2], pass_through)],
        'metrics': metrics(world, plan.path, timing, pass_through),
    }
    with open(path_out, 'w') as f:
        json.dump(data, f, indent=1)
    return data
