"""
HW4 planning facade and run-folder layout (no ROS dependencies beyond optional ament_index).

  make_plan(world, 'fast' | 'safe' | 'weighted') -> Plan (path, waypoints, metrics, extras)
  write_plan(plan, world, folder)                -> waypoints.txt, landmarks.txt, plan.json,
                                                    world.yaml in that folder

Robot runs and simulations live in <workspace>/hw4_logs (or $HW4_LOG_DIR), one folder each:

  <mode>_<YYYY-MM-DD>_<HH-MM-SS>[_<tag>]/       e.g. safe_2026-11-23_14-02-11
      plan.json, waypoints.txt, landmarks.txt, world.yaml, run_info.json   (hw4_launch.py)
      trajectory.csv, waypoints.csv, measurements.csv                     (velocity_control)
      truth.csv                                                           (simulation only)
      bag/                                                                (record_bag:=true)
      eval/                                                               (hw4 eval)
"""

from dataclasses import dataclass, field
import glob
import json
import math
import os
import shutil
import time

from robot_control import hw3_paths
from robot_control.hw4_world import write_landmarks
from robot_control.path_tools import make_waypoints, path_metrics, write_waypoints

MODES = ('fast', 'safe', 'weighted')
DESCRIPTIONS = {
    'fast': 'minimum time: C-space visibility graph + A* (time cost)',
    'safe': 'maximum clearance: GVD on a clearance grid + Dijkstra',
    'weighted': 'trade-off: grid A* with cost (1 + alpha/d)',
}


@dataclass
class Plan:
    mode: str
    path: list
    waypoints: list
    metrics: dict
    detail: object = None          # FastPlan / SafePlan, for figures
    extras: dict = field(default_factory=dict)


def make_plan(world, mode, alpha=None):
    if mode == 'fast':
        from robot_control.visibility_planner import plan_fast
        detail = plan_fast(world)
        intermediate = 'pass' if world.fast['pass_through'] else 'stop'
        extras = {'rho': detail.rho, 'graph_nodes': len(detail.nodes),
                  'graph_edges': len(detail.edges), 'astar_expanded': detail.expanded,
                  'corner_merges': len(detail.merges),
                  'raw_vertices': len(detail.raw_path)}
    elif mode == 'safe':
        from robot_control.gvd_planner import plan_safe
        detail = plan_safe(world)
        intermediate = 'stop'
        extras = {'grid': [detail.grid.nx, detail.grid.ny], 'gvd_cells': int(detail.gvd.sum()),
                  'dijkstra_expanded': detail.expanded, 'raw_points': len(detail.raw_path),
                  'fallback_weighted': detail.fallback}
    elif mode == 'weighted':
        from robot_control.gvd_planner import plan_weighted
        a = world.weighted['alpha'] if alpha is None else alpha
        path, raw = plan_weighted(world, a)
        detail = None
        intermediate = 'stop'
        extras = {'alpha': a, 'raw_points': len(raw)}
        plan = Plan(mode, path, confirmable(world, make_waypoints(path, world.goal[2],
                                                                  intermediate)), {},
                    None, extras)
        plan.metrics = path_metrics(world, path, plan.waypoints)
        return plan
    else:
        raise ValueError(f'mode must be one of {MODES}')
    path = detail.path
    waypoints = confirmable(world, make_waypoints(path, world.goal[2], intermediate))
    return Plan(mode, path, waypoints, path_metrics(world, path, waypoints), detail, extras)


def confirmable(world, waypoints):
    """
    With camera.confirm_heading, turn each stop's heading to the nearest one with a tag in view.

    The controller settles at every stop and needs a tag sighting to confirm the heading (else
    it SEARCHes). Pass-through vertices are untouched, and so is an explicit goal heading.
    """
    if not world.camera.get('confirm_heading', False):
        return waypoints
    out = []
    for i, (x, y, th) in enumerate(waypoints):
        last = i == len(waypoints) - 1
        if not math.isnan(th) and not (last and not math.isnan(world.goal[2])):
            th = world.confirm_heading(x, y, th)
        out.append((x, y, th))
    return out


def _jsonable(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def plan_dict(plan, world):
    return _jsonable({
        'mode': plan.mode,
        'description': DESCRIPTIONS[plan.mode],
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'world_file': world.source,
        'start': list(world.start),
        'goal': list(world.goal),
        'robot_radius': world.robot_radius,
        'path': [list(p) for p in plan.path],
        'waypoints': [list(w) for w in plan.waypoints],
        'metrics': plan.metrics,
        'extras': plan.extras,
        'params': {'fast': world.fast, 'safe': world.safe, 'weighted': world.weighted,
                   'timing': world.timing},
    })


def write_plan(plan, world, folder, waypoints_name='waypoints.txt',
               plan_name='plan.json', landmarks_name='landmarks.txt', copy_world=True):
    os.makedirs(folder, exist_ok=True)
    m = plan.metrics
    header = (f'HW4 {plan.mode} plan ({DESCRIPTIONS[plan.mode]})\n'
              f'world: {world.source}\n'
              f'length {m["length_m"]:.3f} m, min body gap {m["min_body_gap_m"]:.3f} m, '
              f'estimated {m["est_time_s"]:.1f} s')
    paths = {'waypoints': os.path.join(folder, waypoints_name),
             'plan': os.path.join(folder, plan_name),
             'landmarks': os.path.join(folder, landmarks_name)}
    write_waypoints(paths['waypoints'], plan.waypoints, header)
    write_landmarks(world, paths['landmarks'])
    with open(paths['plan'], 'w') as f:
        json.dump(plan_dict(plan, world), f, indent=2)
    if copy_world and world.source and os.path.isfile(world.source):
        dst = os.path.join(folder, 'world.yaml')
        if os.path.abspath(dst) != os.path.abspath(world.source):
            shutil.copy(world.source, dst)
        paths['world'] = dst
    return paths


# ---- folders ------------------------------------------------------------------------------
def _ignored_dir(d):
    d = os.path.expanduser(d)
    os.makedirs(d, exist_ok=True)
    marker = os.path.join(d, 'COLCON_IGNORE')
    if not os.path.exists(marker):
        open(marker, 'w').close()
    return d


def logs_dir():
    return _ignored_dir(os.environ.get('HW4_LOG_DIR')
                        or os.path.join(hw3_paths.workspace_root(), 'hw4_logs'))


def plans_dir():
    return _ignored_dir(os.path.join(hw3_paths.workspace_root(), 'hw4_plans'))


def default_world_file():
    """Return the world file to edit (the source file behind the symlink install, if any)."""
    return os.path.realpath(os.path.join(hw3_paths.config_dir(), 'hw4_world.yaml'))


def default_params_file():
    return os.path.realpath(os.path.join(hw3_paths.config_dir(), 'hw4.yaml'))


def run_name(mode, tag=''):
    return f'{mode}_{hw3_paths.timestamp()}' + (f'_{tag}' if tag else '')


def sim_run_name(mode, seed):
    return f'sim_{mode}_seed{seed}_{hw3_paths.timestamp()}'


def is_run(d):
    return os.path.isfile(os.path.join(d, 'plan.json'))


def all_runs(include_sim=True):
    runs = [d for d in glob.glob(os.path.join(logs_dir(), '*')) if is_run(d)]
    if not include_sim:
        runs = [d for d in runs if not os.path.basename(d).startswith('sim_')]
    return sorted(runs, key=os.path.getmtime)


def resolve_run(spec=None):
    """Find a run folder from a path, 'latest', 'latest-sim' or a unique name part."""
    if spec and os.path.isdir(os.path.expanduser(spec)):
        return os.path.abspath(os.path.expanduser(spec))
    runs = all_runs()
    if not runs:
        raise FileNotFoundError(f'no runs in {logs_dir()}')
    sims = [d for d in runs if os.path.basename(d).startswith('sim_')]
    robot = [d for d in runs if d not in sims]
    if spec in (None, '', 'latest'):
        return (robot or runs)[-1]
    if spec == 'latest-sim':
        if not sims:
            raise FileNotFoundError('no simulation runs')
        return sims[-1]
    matches = [d for d in runs if spec in os.path.basename(d)]
    if len(matches) == 1:
        return matches[0]
    exact = [d for d in matches if os.path.basename(d) == spec]
    if exact:
        return exact[0]
    if not matches:
        raise FileNotFoundError(f'no run matching {spec!r} in {logs_dir()}')
    raise FileNotFoundError(f'{spec!r} matches several runs: '
                            + ', '.join(os.path.basename(d) for d in matches))
