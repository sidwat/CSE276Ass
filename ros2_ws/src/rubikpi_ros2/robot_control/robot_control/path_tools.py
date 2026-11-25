"""
Polyline utilities shared by the HW4 planners (no ROS dependencies).

A path is a list of (x, y) points from the start to the goal. The controller runs it as
waypoints (x, y, theta): theta = nan marks a pass-through vertex (no stop, no final heading).
"""

import math

import numpy as np

from robot_control.waypoint_controller import wrap_angle


def polyline_length(points):
    return float(sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:])))


def leg_headings(points):
    """Heading of each leg: [atan2 of points[i] -> points[i+1]]."""
    return [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(points[:-1], points[1:])]


def rdp(points, epsilon):
    """
    Ramer-Douglas-Peucker: keep the fewest points such that the polyline stays within epsilon.

    Every removed point is within epsilon of the chord that replaces it, so the clearance of
    the simplified path is at most epsilon lower (clearance is 1-Lipschitz).
    """
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = np.array(pts[i]), np.array(pts[j])
        seg = b - a
        norm = float(np.hypot(*seg))
        inner = np.array(pts[i + 1:j])
        if norm < 1e-12:
            dist = np.hypot(*(inner - a).T)
        else:
            dist = np.abs(seg[0] * (inner[:, 1] - a[1]) - seg[1] * (inner[:, 0] - a[0])) / norm
        k = int(np.argmax(dist))
        if dist[k] > epsilon:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return [p for p, k in zip(pts, keep) if k]


def drop_short_legs(path, reference, min_leg, max_dev):
    """
    Remove vertices next to legs shorter than min_leg (start and goal are kept).

    The turn-drive-turn controller cannot drive a leg shorter than its replan_distance (it
    only turns), and every stop costs time. A vertex is removed only if the path then stays
    within max_dev of `reference` (the unsimplified chain), so the clearance drops by at most
    max_dev. Among the candidates the one with the smallest resulting deviation goes first.
    """
    path = list(path)
    while len(path) > 2:
        legs = [math.dist(a, b) for a, b in zip(path[:-1], path[1:])]
        short = [i for i, L in enumerate(legs) if L < min_leg]
        best = None
        for i in short:
            for k in (i, i + 1):  # either end of leg i
                if k == 0 or k == len(path) - 1:
                    continue
                cand = path[:k] + path[k + 1:]
                dev = max_deviation(reference, cand)
                if dev <= max_dev and (best is None or dev < best[0]):
                    best = (dev, cand)
        if best is None:
            break
        path = best[1]
    return path


def max_deviation(original, simplified):
    """Largest distance from a point of `original` to the polyline `simplified`."""
    worst = 0.0
    for p in original:
        best = math.inf
        for a, b in zip(simplified[:-1], simplified[1:]):
            best = min(best, point_segment_distance(p, a, b))
        worst = max(worst, best)
    return worst


def point_segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-18 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def resample(points, step=0.01):
    """Points every `step` metres along the polyline, and their arc length s."""
    xs, ys, ss = [points[0][0]], [points[0][1]], [0.0]
    s0 = 0.0
    for a, b in zip(points[:-1], points[1:]):
        L = math.dist(a, b)
        n = max(1, int(math.ceil(L / step)))
        for k in range(1, n + 1):
            t = k / n
            xs.append(a[0] + t * (b[0] - a[0]))
            ys.append(a[1] + t * (b[1] - a[1]))
            ss.append(s0 + t * L)
        s0 += L
    return np.array(xs), np.array(ys), np.array(ss)


def clearance_profile(world, points, step=0.01):
    """(s, clearance of the rotation centre, clearance to the box) along the path."""
    x, y, s = resample(points, step)
    return s, world.clearance(x, y), world.box.distance(x, y)


def make_waypoints(points, goal_theta=math.nan, intermediate='stop'):
    """
    Waypoints (x, y, theta) for the controller from a path (the start point is dropped).

    intermediate='stop': every vertex is a full stop whose heading is the OUTGOING leg, so the
    robot turns to face the next leg, settles, and the camera confirms that heading right
    before it drives on. 'pass': intermediate vertices get theta = nan (pass-through).
    The goal gets goal_theta, or the heading of the last leg if goal_theta is nan.
    """
    heads = leg_headings(points)
    waypoints = []
    for i in range(1, len(points)):
        x, y = points[i]
        if i == len(points) - 1:
            theta = goal_theta if not math.isnan(goal_theta) else heads[-1]
        elif intermediate == 'pass':
            theta = math.nan
        else:
            theta = heads[i]
        waypoints.append((float(x), float(y), float(theta)))
    return waypoints


def estimate_time(start_pose, waypoints, timing):
    """
    Time model T = sum(l / v) + sum(|turn| / w) + N_stop * t_stop + N_pass * t_pass.

    Mirrors the turn-drive-turn controller: at each waypoint turn to face the next one, drive,
    then (stop waypoints only) turn to the waypoint heading and settle.
    """
    v, w = timing['v'], timing['w']
    x, y, heading = start_pose
    drive = turn = 0.0
    stops = passes = 0
    for wx, wy, wt in waypoints:
        L = math.hypot(wx - x, wy - y)
        if L > 1e-9:
            psi = math.atan2(wy - y, wx - x)
            turn += abs(wrap_angle(psi - heading))
            heading = psi
            drive += L
        if math.isnan(wt):
            passes += 1
        else:
            turn += abs(wrap_angle(wt - heading))
            heading = wt
            stops += 1
        x, y = wx, wy
    t_drive, t_turn = drive / v, turn / w
    t_vertex = stops * timing['t_stop'] + passes * timing['t_pass']
    return {'drive_s': t_drive, 'turn_s': t_turn, 'vertex_s': t_vertex,
            'total_s': t_drive + t_turn + t_vertex, 'turn_rad': turn,
            'stops': stops, 'passes': passes}


def path_metrics(world, points, waypoints):
    s, clear, box = clearance_profile(world, points)
    r = world.robot_radius
    t = estimate_time(world.start, waypoints, world.timing)
    return {
        'length_m': polyline_length(points),
        'vertices': len(points),
        'waypoints': len(waypoints),
        'stops': t['stops'],
        'pass_through': t['passes'],
        'min_clearance_m': float(clear.min()),
        'min_body_gap_m': float(clear.min() - r),
        'mean_clearance_m': float(clear.mean()),
        'min_box_gap_m': float(box.min() - r),
        'est_time_s': t['total_s'],
        'est_drive_s': t['drive_s'],
        'est_turn_s': t['turn_s'],
        'est_vertex_s': t['vertex_s'],
        'turn_rad': t['turn_rad'],
    }


def write_waypoints(path, waypoints, header=''):
    with open(path, 'w') as f:
        for line in header.strip().splitlines():
            f.write(f'# {line}\n')
        f.write('# x [m], y [m], theta [rad] in the world frame; theta nan = pass-through\n')
        for x, y, t in waypoints:
            f.write(f'{x:.4f}, {y:.4f}, {"nan" if math.isnan(t) else f"{t:.4f}"}\n')
