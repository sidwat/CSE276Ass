"""Turn-drive-turn waypoint follower for a differential-drive robot (no ROS dependencies)."""

from dataclasses import dataclass
import math


def wrap_angle(angle):
    """Wrap an angle in radians to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, low, high):
    return max(low, min(high, value))


def load_waypoints(path):
    """Read 'x,y,theta' lines (metres, metres, radians), skipping blanks and '#' comments."""
    waypoints = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) != 3:
                raise ValueError(f'{path}:{lineno}: expected "x,y,theta", got {line!r}')
            waypoints.append(tuple(float(p) for p in parts))
    return waypoints


def body_to_wheel_cmds(v, w, speed_per_unit, wheel_base, max_cmd):
    """
    Convert a body velocity (v [m/s], w [rad/s]) into L/R motor commands.

    speed_per_unit is the wheel ground speed [m/s] produced by a command of 1.0, and wheel_base
    is the effective track width [m]. If either wheel would exceed max_cmd, both are scaled down
    together so the commanded turning radius is kept.
    """
    left = (v - w * wheel_base / 2.0) / speed_per_unit
    right = (v + w * wheel_base / 2.0) / speed_per_unit
    largest = max(abs(left), abs(right))
    if largest > max_cmd:
        left *= max_cmd / largest
        right *= max_cmd / largest
    return left, right


def wheel_cmds_to_body(left, right, speed_per_unit, wheel_base):
    """Inverse of body_to_wheel_cmds (without the saturation)."""
    v_left = left * speed_per_unit
    v_right = right * speed_per_unit
    return (v_left + v_right) / 2.0, (v_right - v_left) / wheel_base


@dataclass
class Gains:
    pos_tolerance: float = 0.05      # m, waypoint position counts as reached inside this radius
    heading_tolerance: float = 0.05  # rad (~3 deg)
    k_v: float = 1.0                 # (m/s) per m of remaining distance
    k_w: float = 2.0                 # (rad/s) per rad of heading error
    max_v: float = 0.15              # m/s
    min_v: float = 0.05              # m/s, below this the motors stall
    max_w: float = 1.0               # rad/s
    min_w: float = 0.4               # rad/s, below this the motors stall
    realign_angle: float = 0.5       # rad, stop and turn in place if heading drifts this far
    # m: closer than this, the robot never turns in place to face the waypoint (it finishes by
    # driving straight on); farther, it goes back for it even after arriving
    replan_distance: float = 0.1
    settle_time: float = 0.0         # s to stop and re-check the heading before a waypoint counts
    search_turn_time: float = 0.3    # s of turning per search step (heading not confirmed)
    search_look_time: float = 1.0    # s stopped after each search step, looking for a tag
    search_max_turn: float = 6.3     # rad of commanded search turning before giving up on it


class WaypointController:
    """
    Drive to each waypoint in turn: rotate to face it, drive to it, rotate to its heading.

    Call update() with the current pose estimate every control cycle; it returns the body
    velocity command to apply. settle_time needs update()'s dt to count time.

    With heading_confirmed (e.g. a landmark was seen during the settle), a waypoint counts once
    its heading has held for settle_time. Without it the estimate may only be the motion model's
    claim, so the robot SEARCHes: it keeps turning the way it last turned, a step at a time,
    stopping to look after each step, until a sighting confirms (and corrects) the heading.
    """

    ROTATE_TO_TARGET = 'rotate_to_target'
    DRIVE = 'drive'
    ROTATE_TO_HEADING = 'rotate_to_heading'
    SETTLE = 'settle'
    SEARCH = 'search'

    def __init__(self, waypoints, gains=None):
        self.waypoints = list(waypoints)
        self.gains = gains or Gains()
        self.index = 0
        self.phase = self.ROTATE_TO_TARGET
        self.settle_time = 0.0  # s spent stopped in SETTLE
        self.search_time = 0.0  # s since SEARCH started
        self.search_turned = 0.0  # rad of search turning commanded for this waypoint
        self.last_turn_sign = 1.0

    @property
    def done(self):
        return self.index >= len(self.waypoints)

    def update(self, x, y, theta, dt=0.0, heading_confirmed=True):
        """
        Return (v, w, reached) for the pose (x, y, theta), dt seconds after the previous call.

        reached is the index of the waypoint that was completed on this call, otherwise None.
        """
        g = self.gains
        while not self.done:
            goal_x, goal_y, goal_theta = self.waypoints[self.index]
            dx, dy = goal_x - x, goal_y - y
            dist = math.hypot(dx, dy)
            bearing_err = wrap_angle(math.atan2(dy, dx) - theta)

            if self.phase == self.ROTATE_TO_TARGET:
                if dist < g.replan_distance:  # too close to be worth turning to face it
                    self.phase = self.ROTATE_TO_HEADING
                elif abs(bearing_err) < g.heading_tolerance:
                    self.phase = self.DRIVE
                else:
                    return 0.0, self._turn_rate(bearing_err), None

            elif self.phase == self.DRIVE:
                # Close in, a few cm of sideways error swings the bearing wildly, so stop
                # steering and drive straight on until the waypoint is no longer ahead
                ahead = dx * math.cos(theta) + dy * math.sin(theta)
                near = dist < g.replan_distance
                if dist < g.pos_tolerance or (near and ahead < g.pos_tolerance):
                    self.phase = self.ROTATE_TO_HEADING
                elif not near and abs(bearing_err) > g.realign_angle:
                    self.phase = self.ROTATE_TO_TARGET
                elif near:
                    return clamp(g.k_v * ahead, g.min_v, g.max_v), 0.0, None
                else:
                    v = clamp(g.k_v * dist, g.min_v, g.max_v)
                    w = clamp(g.k_w * bearing_err, -g.max_w, g.max_w)
                    return v, w, None

            elif self.phase == self.ROTATE_TO_HEADING:
                if dist > g.replan_distance:
                    self.phase = self.ROTATE_TO_TARGET
                    continue
                heading_err = wrap_angle(goal_theta - theta)
                if abs(heading_err) < g.heading_tolerance:
                    self.phase = self.SETTLE
                    self.settle_time = 0.0
                    continue
                return 0.0, self._turn_rate(heading_err), None

            elif self.phase == self.SETTLE:  # stopped, waiting for the camera to confirm
                if dist > g.replan_distance:
                    self.phase = self.ROTATE_TO_TARGET
                    continue
                if abs(wrap_angle(goal_theta - theta)) > 2.0 * g.heading_tolerance:
                    self.phase = self.ROTATE_TO_HEADING
                    continue
                self.settle_time += dt
                if self.settle_time < g.settle_time:
                    return 0.0, 0.0, None
                if not heading_confirmed and self.search_turned < g.search_max_turn:
                    self.phase = self.SEARCH
                    self.search_time = 0.0
                    continue
                reached = self.index
                self.index += 1
                self.phase = self.ROTATE_TO_TARGET
                self.search_turned = 0.0
                return 0.0, 0.0, reached

            else:  # SEARCH: step-turn-look until a sighting confirms the heading
                if heading_confirmed:
                    self.phase = self.SETTLE
                    self.settle_time = 0.0
                    continue
                if self.search_turned >= g.search_max_turn:
                    self.phase = self.SETTLE  # give up searching; accept the estimate
                    continue
                self.search_time += dt
                step = self.search_time % (g.search_turn_time + g.search_look_time)
                if step < g.search_turn_time:
                    self.search_turned += g.min_w * dt
                    return 0.0, self.last_turn_sign * g.min_w, None
                return 0.0, 0.0, None

        return 0.0, 0.0, None

    def _turn_rate(self, err):
        g = self.gains
        self.last_turn_sign = math.copysign(1.0, err)
        return math.copysign(clamp(g.k_w * abs(err), g.min_w, g.max_w), err)
