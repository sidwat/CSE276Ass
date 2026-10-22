"""Smooth constant-speed trajectory that meets every waypoint position and heading exactly."""

from dataclasses import dataclass, field
import itertools
import math

import numpy as np

FOOT = 0.3048

# Quintic Hermite basis in power form; rows weight [p0, d0, dd0, dd1, d1, p1], columns s^0..s^5
_HERMITE = np.array([
    [1.0, 0.0, 0.0, -10.0, 15.0, -6.0],
    [0.0, 1.0, 0.0, -6.0, 8.0, -3.0],
    [0.0, 0.0, 0.5, -1.5, 1.5, -0.5],
    [0.0, 0.0, 0.0, 0.5, -1.0, 0.5],
    [0.0, 0.0, 0.0, -4.0, 7.0, -3.0],
    [0.0, 0.0, 0.0, 10.0, -15.0, 6.0],
])

_STOP_COST = 1.0      # extra cost of a direction change (the robot touches zero speed)
_INFEASIBLE = 1e6


def _basis(s, deriv=0):
    coeffs = _HERMITE.copy()
    for _ in range(deriv):
        coeffs = coeffs[:, 1:] * np.arange(1, coeffs.shape[1])
    powers = np.vstack([s ** k for k in range(coeffs.shape[1])])
    return coeffs @ powers


def _unit(angle):
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def _normal(angle):
    return np.stack([-np.sin(angle), np.cos(angle)], axis=-1)


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _distance_to_segment(points, a, b):
    ab = b - a
    t = np.clip(np.sum((points - a) * ab, axis=-1) / np.sum(ab * ab, axis=-1), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[..., None] * ab), axis=-1)


def _smoothstep(u):
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)


def _smoothstep_integral(u):
    return u ** 4 * (2.5 - 3.0 * u + u ** 2)


@dataclass
class PlannerConfig:
    cruise_speed: float = 0.25      # m/s, held constant between the start and stop ramps
    ramp_time: float = 1.5          # s, smooth acceleration from rest (and deceleration to rest)
    max_curvature: float = 4.0      # 1/m, tightest turn allowed (radius 0.25 m)
    corridor_width: float = FOOT    # m, max distance from the straight line between waypoints
    arena_min_x: float = -0.0668    # m, lower-left arena corner in the waypoint frame
    arena_min_y: float = -0.0668
    arena_size: float = 2.1336      # m, 7 ft square
    speed_per_unit: float = 1.4     # calibration, same meaning as in velocity_control.yaml
    wheel_base: float = 0.8
    wheel_trim: float = 0.0         # >0 if it veers right going straight, same as velocity_control
    max_wheel_cmd: float = 0.5

    def curvature_limit(self):
        """Tightest curvature the wheels can follow at cruise speed without saturating."""
        wheel_limit = (self.max_wheel_cmd * self.speed_per_unit / self.cruise_speed - 1.0) \
            * 2.0 / self.wheel_base
        return min(self.max_curvature, wheel_limit)


@dataclass
class Sample:
    x: float
    y: float
    heading: float
    v: float          # body forward velocity, negative when reversing
    w: float          # yaw rate
    curvature: float


@dataclass
class _Leg:
    """A stretch driven in one direction, from rest to rest."""

    direction: int
    s: np.ndarray
    xy: np.ndarray
    phi: np.ndarray
    kappa: np.ndarray
    peak_speed: float
    ramp_time: float
    start_time: float = 0.0

    @property
    def length(self):
        return float(self.s[-1])

    @property
    def duration(self):
        ramp_dist = self.peak_speed * self.ramp_time
        return 2.0 * self.ramp_time + max(0.0, self.length - ramp_dist) / self.peak_speed

    def progress(self, t):
        """Arc length and speed t seconds into the leg."""
        v, tr, dur = self.peak_speed, self.ramp_time, self.duration
        t = min(max(t, 0.0), dur)
        if t < tr:
            return v * tr * _smoothstep_integral(t / tr), v * _smoothstep(t / tr)
        if t > dur - tr:
            u = (dur - t) / tr
            return self.length - v * tr * _smoothstep_integral(u), v * _smoothstep(u)
        return v * tr / 2.0 + v * (t - tr), v

    def time_at(self, s):
        ts = np.linspace(0.0, self.duration, 2000)
        return float(np.interp(s, [self.progress(t)[0] for t in ts], ts))


@dataclass
class PlanReport:
    directions: list
    straight: list
    stops: list           # waypoint indices where the robot touches zero speed to change direction
    max_deviation: list
    max_curvature: float
    curvature_limit: float
    arena_bounds: tuple
    path_bounds: tuple
    violations: list = field(default_factory=list)


class Trajectory:
    """Time-parameterised plan; sample(t) gives the pose and body velocity to command at t."""

    def __init__(self, legs, waypoint_times, segments, report):
        self.legs = legs
        self.waypoint_times = waypoint_times
        self.segments = segments
        self.report = report
        t = 0.0
        for leg in legs:
            leg.start_time = t
            t += leg.duration
        self.duration = t

    def sample(self, t):
        leg = self.legs[-1]
        for candidate in self.legs:
            if t < candidate.start_time + candidate.duration:
                leg = candidate
                break
        s, speed = leg.progress(t - leg.start_time)
        x = np.interp(s, leg.s, leg.xy[:, 0])
        y = np.interp(s, leg.s, leg.xy[:, 1])
        phi = np.interp(s, leg.s, leg.phi)
        kappa = np.interp(s, leg.s, leg.kappa)
        heading = _wrap(phi if leg.direction > 0 else phi + math.pi)
        return Sample(float(x), float(y), heading, leg.direction * speed, kappa * speed, kappa)


class _Problem:
    """
    Shape of the path for a fixed driving direction on every segment.

    Each segment is a quintic Hermite curve whose end tangents follow the waypoint headings (in
    the direction of travel), so headings are met exactly. A segment whose headings both point
    along its chord is a fixed straight line. The free variables are the end curvatures and the
    two tangent lengths of every curved segment. Where the direction does not change, the
    curvature is shared by both segments, so heading and curvature are continuous; where it
    changes, the robot is momentarily at rest and the two sides are independent.
    """

    SAMPLES = 60
    MARGIN = 0.005  # m
    RATE_WEIGHT = 1e-3  # smooth turn entry: penalise how fast curvature changes

    def __init__(self, waypoints, config, directions):
        self.config = config
        self.wp = np.array([w[:2] for w in waypoints], dtype=float)
        theta = np.array([w[2] for w in waypoints], dtype=float)
        self.direction = np.asarray(directions, dtype=int)
        self.n_seg = len(self.direction)
        flip = np.where(self.direction < 0, math.pi, 0.0)
        self.phi0 = theta[:-1] + flip
        self.phi1 = theta[1:] + flip
        self.chord = np.linalg.norm(self.wp[1:] - self.wp[:-1], axis=1)
        self.straight = [self._is_straight(j) for j in range(self.n_seg)]
        self.k_lim = config.curvature_limit()
        lo = np.array([config.arena_min_x, config.arena_min_y])
        self.arena = (lo, lo + config.arena_size)

        count = 0
        self.k0_idx = [-1] * self.n_seg
        self.k1_idx = [-1] * self.n_seg
        for knot in range(self.n_seg + 1):
            before, after = knot - 1, knot
            has_before, has_after = before >= 0, after < self.n_seg
            if has_before and has_after and self.direction[before] == self.direction[after]:
                if not (self.straight[before] or self.straight[after]):
                    self.k1_idx[before] = self.k0_idx[after] = count
                    count += 1
                continue
            if has_before and not self.straight[before]:
                self.k1_idx[before] = count
                count += 1
            if has_after and not self.straight[after]:
                self.k0_idx[after] = count
                count += 1
        self.len_idx = {}
        for j in range(self.n_seg):
            if not self.straight[j]:
                self.len_idx[j] = count
                count += 2
        self.n_vars = count

        s = np.linspace(0.0, 1.0, self.SAMPLES)
        self.B = [_basis(s, d) for d in range(3)]
        self.ds = 1.0 / (self.SAMPLES - 1)

    def _is_straight(self, j):
        chord = self.wp[j + 1] - self.wp[j]
        along = math.atan2(chord[1], chord[0])
        return abs(_wrap(self.phi0[j] - along)) < 1e-3 and abs(_wrap(self.phi1[j] - along)) < 1e-3

    def steps(self):
        steps = np.ones(self.n_vars)
        for j in self.len_idx.values():
            steps[j:j + 2] = 0.3
        return steps

    def guess(self, lengths=(1.0, 1.0), curvatures=(0.0, 0.0)):
        x = np.zeros(self.n_vars)
        for j, i in self.len_idx.items():
            x[i:i + 2] = np.log(self.chord[j] * np.array(lengths))
        for j in range(self.n_seg):
            if self.k0_idx[j] >= 0:
                x[self.k0_idx[j]] = curvatures[0]
            if self.k1_idx[j] >= 0:
                x[self.k1_idx[j]] = curvatures[1]
        return x

    def segments(self, x):
        """Hermite control data (n_seg, 6, 2): [p0, d0, dd0, dd1, d1, p1]."""
        k0 = np.array([x[i] if i >= 0 else 0.0 for i in self.k0_idx])[:, None]
        k1 = np.array([x[i] if i >= 0 else 0.0 for i in self.k1_idx])[:, None]
        a, b = self.chord.copy(), self.chord.copy()
        for j, i in self.len_idx.items():
            a[j], b[j] = math.exp(x[i]), math.exp(x[i + 1])
        a, b = a[:, None], b[:, None]
        return np.stack([
            self.wp[:-1],
            a * _unit(self.phi0),
            a ** 2 * k0 * _normal(self.phi0),
            b ** 2 * k1 * _normal(self.phi1),
            b * _unit(self.phi1),
            self.wp[1:],
        ], axis=1)

    def evaluate(self, ctrl, basis=None):
        """Points, signed curvature and parametric speed for every segment."""
        b0, b1, b2 = basis or self.B
        pts = np.einsum('ks,nkd->nsd', b0, ctrl)
        d1 = np.einsum('ks,nkd->nsd', b1, ctrl)
        d2 = np.einsum('ks,nkd->nsd', b2, ctrl)
        speed = np.linalg.norm(d1, axis=-1)
        kappa = (d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]) / np.maximum(speed, 1e-9) ** 3
        return pts, kappa, speed

    def _deviation(self, pts):
        return _distance_to_segment(pts, self.wp[:-1, None], self.wp[1:, None])

    def cost(self, x):
        cfg = self.config
        pts, kappa, speed = self.evaluate(self.segments(x))
        ds = speed * self.ds
        dev = self._deviation(pts)
        dk_ds = np.diff(kappa, axis=1) / np.maximum(0.5 * (ds[:, 1:] + ds[:, :-1]), 1e-9)
        # Plan slightly inside the limits so the densely re-sampled path still meets them
        lo, hi = self.arena[0] + self.MARGIN, self.arena[1] - self.MARGIN
        violation = (
            np.sum(np.maximum(0.0, dev - (cfg.corridor_width - self.MARGIN)) ** 2)
            + np.sum(np.maximum(0.0, lo - pts) ** 2) + np.sum(np.maximum(0.0, pts - hi) ** 2)
            + 0.01 * np.sum(np.maximum(0.0, np.abs(kappa) - 0.97 * self.k_lim) ** 2)
            + np.sum(np.maximum(0.0, 0.05 - speed) ** 2)
        )
        rate = np.sum(dk_ds ** 2 * 0.5 * (ds[:, 1:] + ds[:, :-1]))
        return (0.02 * np.sum(kappa ** 2 * ds) + self.RATE_WEIGHT * rate
                + 2.0 * np.sum(dev ** 2 * ds) + 0.05 * np.sum(ds) + 1e4 * violation)

    def check(self, x):
        """Per-segment deviation, overall curvature/bounds, and violated constraints."""
        cfg = self.config
        s = np.linspace(0.0, 1.0, 400)
        pts, kappa, _ = self.evaluate(self.segments(x), [_basis(s, d) for d in range(3)])
        dev = self._deviation(pts).max(axis=1)
        flat = pts.reshape(-1, 2)
        lo, hi = self.arena
        violations = []
        for j, d in enumerate(dev):
            if d > cfg.corridor_width + 2e-3:
                violations.append(
                    f'segment {j} leaves the 1 ft corridor by {d - cfg.corridor_width:.3f} m')
        k_max = float(np.abs(kappa).max())
        if k_max > self.k_lim * 1.03:
            violations.append(f'curvature {k_max:.2f} exceeds the limit {self.k_lim:.2f} 1/m')
        out = max(float(np.max(lo - flat)), float(np.max(flat - hi)))
        if out > 2e-3:
            violations.append(f'path leaves the arena by {out:.3f} m')
        return dev, k_max, (tuple(flat.min(axis=0)), tuple(flat.max(axis=0))), violations


def _pattern_search(f, x, steps, min_scale=1e-3, max_evals=20000):
    """Derivative-free coordinate search; robust with the hinge penalties in the cost."""
    fx, scale, evals = f(x), 1.0, 1
    while scale > min_scale and evals < max_evals:
        improved = False
        for i in range(len(x)):
            for sign in (1.0, -1.0):
                trial = x.copy()
                trial[i] += sign * scale * steps[i]
                ft = f(trial)
                evals += 1
                if ft < fx:
                    x, fx, improved = trial, ft, True
                    break
        if not improved:
            scale *= 0.5
    return x, fx


def _fit_segment(waypoints, config, direction):
    """Best shape for one segment on its own, from several starting shapes."""
    problem = _Problem(waypoints, config, [direction])
    starts = itertools.product(
        [(0.7, 0.7), (1.4, 1.4)], [(0.0, 0.0), (3.0, -3.0), (-3.0, 3.0), (3.0, 3.0), (-3.0, -3.0)])
    fits = [_pattern_search(problem.cost, problem.guess(lengths, curvatures), problem.steps(),
                            min_scale=0.02)
            for lengths, curvatures in starts]
    x, f = min(fits, key=lambda fit: fit[1])
    x, f = _pattern_search(problem.cost, x, problem.steps())
    feasible = not problem.check(x)[3]
    return problem, x, f + (0.0 if feasible else _INFEASIBLE)


def _choose_directions(waypoints, config):
    """Pick forward/reverse per segment: feasible first, then fewest direction changes."""
    fits = [{d: _fit_segment(waypoints[j:j + 2], config, d) for d in (1, -1)}
            for j in range(len(waypoints) - 1)]
    best = {d: (fits[0][d][2], [d]) for d in (1, -1)}
    for seg in fits[1:]:
        best = {
            d: min(((cost + (_STOP_COST if prev != d else 0.0) + seg[d][2], path + [d])
                    for prev, (cost, path) in best.items()), key=lambda c: c[0])
            for d in (1, -1)
        }
    directions = min(best.values(), key=lambda c: c[0])[1]
    return directions, [fits[j][d] for j, d in enumerate(directions)]


def _joint_guess(problem, seg_fits):
    """Start the joint optimisation from the independent per-segment fits."""
    x = np.zeros(problem.n_vars)
    count = np.zeros(problem.n_vars)
    for j, (sub, sx, _) in enumerate(seg_fits):
        pairs = ((problem.k0_idx[j], sub.k0_idx[0]), (problem.k1_idx[j], sub.k1_idx[0]))
        for own, theirs in pairs:
            if own >= 0 and theirs >= 0:
                x[own] += sx[theirs]
                count[own] += 1
        if j in problem.len_idx:
            i, si = problem.len_idx[j], sub.len_idx[0]
            x[i:i + 2] = sx[si:si + 2]
            count[i:i + 2] = 1
    return x / np.maximum(count, 1)


def _build_legs(problem, ctrl, config, samples=400):
    s = np.linspace(0.0, 1.0, samples)
    basis = [_basis(s, d) for d in range(3)]
    pts, kappa, speed = problem.evaluate(ctrl, basis)
    d1 = np.einsum('ks,nkd->nsd', basis[1], ctrl)
    phi = np.arctan2(d1[..., 1], d1[..., 0])

    legs, seg_start = [], []
    seg = 0
    while seg < problem.n_seg:
        direction = problem.direction[seg]
        xy, ph, kp, arc = [], [], [], [0.0]
        while seg < problem.n_seg and problem.direction[seg] == direction:
            seg_start.append((len(legs), arc[-1]))
            first = slice(None) if not xy else slice(1, None)
            ds = np.diff(s) * 0.5 * (speed[seg, 1:] + speed[seg, :-1])
            xy.append(pts[seg, first])
            ph.append(phi[seg, first])
            kp.append(kappa[seg, first])
            arc.extend(arc[-1] + np.cumsum(ds))
            seg += 1
        arc = np.array(arc)
        peak = min(config.cruise_speed, arc[-1] / config.ramp_time)
        legs.append(_Leg(direction, arc, np.vstack(xy), np.unwrap(np.concatenate(ph)),
                         np.concatenate(kp), peak, config.ramp_time))
    return legs, seg_start


def plan_trajectory(waypoints, config=None):
    """Fit the smoothest path that meets every waypoint pose and time it at constant speed."""
    config = config or PlannerConfig()
    if len(waypoints) < 2:
        raise ValueError('need at least two waypoints')

    directions, seg_fits = _choose_directions(waypoints, config)
    problem = _Problem(waypoints, config, directions)
    x, _ = _pattern_search(problem.cost, _joint_guess(problem, seg_fits), problem.steps())
    ctrl = problem.segments(x)

    legs, seg_start = _build_legs(problem, ctrl, config)
    leg_start, t = [], 0.0
    for leg in legs:
        leg_start.append(t)
        t += leg.duration
    waypoint_times = [leg_start[i] + legs[i].time_at(s) for i, s in seg_start] + [t]

    dev, k_max, bounds, violations = problem.check(x)
    report = PlanReport(
        directions=[int(d) for d in directions],
        straight=list(problem.straight),
        stops=[j for j in range(1, len(directions)) if directions[j] != directions[j - 1]],
        max_deviation=[float(d) for d in dev],
        max_curvature=k_max,
        curvature_limit=float(problem.k_lim),
        arena_bounds=(tuple(problem.arena[0]), tuple(problem.arena[1])),
        path_bounds=bounds,
        violations=violations,
    )
    return Trajectory(legs, waypoint_times, ctrl, report)
