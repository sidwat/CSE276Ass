"""
Turn-drive-turn plan built only from constant wheel commands, timed from calibration (no ROS).

The motors have a dead zone (below ~0.1 a wheel does not turn) and the turning response depends
strongly on how the wheels are combined, so every motion here is one of two fixed, calibrated
primitives: spin in place at +/-spin_cmd, or drive straight at +/-drive_cmd. Each segment is
driven forwards or backwards, whichever needs less total turning.
"""

from dataclasses import dataclass
import math

from robot_control.waypoint_controller import wrap_angle


@dataclass
class SegmentConfig:
    drive_cmd: float = 0.2              # straight-driving wheel command; keep >= 0.15 (dead zone)
    spin_cmd: float = 0.3               # wheel command magnitude when turning in place
    speed_per_unit: float = 1.26        # m/s per unit command, driving forwards
    reverse_speed_per_unit: float = 1.26
    drive_offset: float = 0.0           # m, distance a straight covers beyond cmd*speed*time
    wheel_trim: float = -0.06           # used for forward driving and spins
    reverse_trim: float = -0.06         # used for reverse driving
    ccw_deg_per_s: float = 83.3         # spin rate, counter-clockwise
    ccw_offset_deg: float = 10.0        # extra degrees from starting and stopping a CCW spin
    cw_deg_per_s: float = 93.3
    cw_offset_deg: float = 20.0
    min_turn_deg: float = 3.0           # turns smaller than this are skipped
    pause: float = 0.5                  # s at rest between motions so the robot settles
    allow_reverse: bool = True


@dataclass
class Motion:
    label: str
    left: float           # command as sent, trim included
    right: float
    duration: float
    start_time: float = 0.0
    waypoint: int = -1    # waypoint reached when this motion ends, else -1


def spin_duration(angle_deg, cfg):
    """Seconds of spinning for a signed angle (CCW +), from the rate-plus-offset calibration."""
    if angle_deg > 0:
        return max(0.0, (angle_deg - cfg.ccw_offset_deg) / cfg.ccw_deg_per_s)
    return max(0.0, (-angle_deg - cfg.cw_offset_deg) / cfg.cw_deg_per_s)


def plan_segments(waypoints, cfg=None):
    """Return (motions, notes) taking the robot through every waypoint pose."""
    cfg = cfg or SegmentConfig()
    motions, notes = [], []
    heading = waypoints[0][2]

    def add(label, left, right, duration, trim):
        motions.append(Motion(label, left * (1.0 - trim), right * (1.0 + trim), duration))
        motions.append(Motion('pause', 0.0, 0.0, cfg.pause))

    def turn_to(target):
        nonlocal heading
        angle = math.degrees(wrap_angle(target - heading))
        heading = target
        if abs(angle) < cfg.min_turn_deg:
            return
        offset = cfg.ccw_offset_deg if angle > 0 else cfg.cw_offset_deg
        if abs(angle) <= offset:
            notes.append(f'{angle:+.0f} deg turn is within the {offset:.0f} deg start/stop '
                         'offset, so it is skipped; calibrate small turns if this matters')
        s = cfg.spin_cmd if angle > 0 else -cfg.spin_cmd
        add(f'spin {angle:+.0f} deg', -s, s, spin_duration(angle, cfg), cfg.wheel_trim)

    for i in range(len(waypoints) - 1):
        x0, y0, _ = waypoints[i]
        x1, y1, theta1 = waypoints[i + 1]
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist > 1e-3:
            bearing = math.atan2(y1 - y0, x1 - x0)
            back = wrap_angle(bearing + math.pi)
            cost_fwd = abs(wrap_angle(bearing - heading)) + abs(wrap_angle(theta1 - bearing))
            cost_rev = abs(wrap_angle(back - heading)) + abs(wrap_angle(theta1 - back))
            reverse = cfg.allow_reverse and cost_rev < cost_fwd - 1e-6
            turn_to(back if reverse else bearing)
            if reverse:
                speed = cfg.drive_cmd * cfg.reverse_speed_per_unit
                duration = max(0.0, (dist - cfg.drive_offset) / speed)
                add(f'reverse {dist:.2f} m', -cfg.drive_cmd, -cfg.drive_cmd, duration,
                    cfg.reverse_trim)
            else:
                speed = cfg.drive_cmd * cfg.speed_per_unit
                duration = max(0.0, (dist - cfg.drive_offset) / speed)
                add(f'forward {dist:.2f} m', cfg.drive_cmd, cfg.drive_cmd, duration,
                    cfg.wheel_trim)
        turn_to(theta1)
        if motions:
            motions[-1].waypoint = i + 1

    t = 0.0
    for motion in motions:
        motion.start_time = t
        t += motion.duration
    return motions, notes
