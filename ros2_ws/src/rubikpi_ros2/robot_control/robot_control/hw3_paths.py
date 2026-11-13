"""
Where HW3 runs live and how they are named (no ROS dependencies beyond ament_index, optional).

All runs (robot and simulation) go into one folder, <workspace>/hw3_logs, unless the HW3_LOG_DIR
environment variable says otherwise. Each run is one self-contained folder:

  <path>_<dof>dof_<YYYY-MM-DD>_<HH-MM-SS>[_<tag>]/        e.g. square_3dof_2026-11-12_15-04-31
      run_info.json, start_pose.txt, *.csv, state_*.npz     (slam_logger.py)
      bag/                                                  (ros2 bag, with record_bag:=true)
      eval/                                                 (evaluate_slam.py: maps, results.md)
      landmarks_gt_used.txt                                 (ground truth the eval compared to)

Simulations are named sim_<path>_<dof>dof_seed<N>_<date>_<time>.
"""

import glob
import os
import time

# World-frame start pose of each path in the frame of landmarks_hw3_gt.txt: the measured arena
# setup places the robot at (0.9 m, 0.9 m) facing east for the runs (override per run with
# start:=x,y,theta on the launch or hw3 eval --start).
NOMINAL_START = {
    'square': (0.9, 0.9, 0.0),
    'octagon': (1.21, 0.79, 0.0),
}
ARENA_SIDE = 3.0  # m, 10 ft arena taken as 3.0 m


def _looks_like_workspace(d):
    return os.path.isdir(os.path.join(d, 'src')) and os.path.isdir(os.path.join(d, 'install'))


def workspace_root():
    """The colcon workspace this package was built in (the folder holding src/ and install/)."""
    try:
        from ament_index_python.packages import get_package_prefix
        ws = os.path.dirname(os.path.dirname(get_package_prefix('robot_control')))
        if _looks_like_workspace(ws):
            return ws
    except Exception:  # noqa: BLE001  (no ROS environment)
        pass
    d = os.path.dirname(os.path.realpath(__file__))
    while d != os.path.dirname(d):
        if _looks_like_workspace(d):
            return d
        d = os.path.dirname(d)
    return os.path.expanduser('~/CSE276Ass/ros2_ws')


def logs_dir():
    d = os.environ.get('HW3_LOG_DIR') or os.path.join(workspace_root(), 'hw3_logs')
    d = os.path.expanduser(d)
    os.makedirs(d, exist_ok=True)
    marker = os.path.join(d, 'COLCON_IGNORE')  # keep colcon from crawling the bags
    if not os.path.exists(marker):
        open(marker, 'w').close()
    return d


def config_dir():
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
    if os.path.isfile(os.path.join(here, 'hw3.yaml')):
        return here
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('robot_control'), 'config')
    except Exception:  # noqa: BLE001
        return here


def default_gt_file():
    """The ground-truth file to edit (the source file behind the symlink install, if any)."""
    return os.path.realpath(os.path.join(config_dir(), 'landmarks_hw3_gt.txt'))


def gt_is_template(path):
    """True while the ground-truth file is still the unmeasured template."""
    try:
        with open(path) as f:
            return 'TEMPLATE' in f.read()
    except OSError:
        return True


def timestamp():
    return time.strftime('%Y-%m-%d_%H-%M-%S')


def run_name(path, dof, tag=''):
    return f'{path}_{dof}dof_{timestamp()}' + (f'_{tag}' if tag else '')


def sim_run_name(path, dof, seed):
    return f'sim_{path}_{dof}dof_seed{seed}_{timestamp()}'


def is_run(d):
    return os.path.isfile(os.path.join(d, 'run_info.json'))


def run_started(d):
    """Start time of a run as a sortable string (run_info.json 'started', else folder mtime)."""
    try:
        import json
        with open(os.path.join(d, 'run_info.json')) as f:
            started = json.load(f).get('started')
        if started:
            return started
    except (OSError, ValueError):
        pass
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(d)))


def all_runs(include_sim=True):
    """Run folders, oldest first by start time."""
    runs = [d for d in glob.glob(os.path.join(logs_dir(), '*')) if is_run(d)]
    if not include_sim:
        runs = [d for d in runs if not os.path.basename(d).startswith('sim_')]
    return sorted(runs, key=run_started)


def resolve_run(spec=None, include_sim=True):
    """
    A run folder from a path, a folder name (or unique part of one), 'latest' or None.

    'latest'/None means the newest robot run (newest simulation if there are only simulations).
    """
    if spec and os.path.isdir(os.path.expanduser(spec)) and is_run(os.path.expanduser(spec)):
        return os.path.abspath(os.path.expanduser(spec))
    runs = all_runs(include_sim)
    if not runs:
        raise FileNotFoundError(f'no runs in {logs_dir()}')
    if spec in (None, '', 'latest'):
        robot = [d for d in runs if not os.path.basename(d).startswith('sim_')]
        return (robot or runs)[-1]
    if spec == 'latest-sim':
        sims = [d for d in runs if os.path.basename(d).startswith('sim_')]
        if not sims:
            raise FileNotFoundError('no simulation runs')
        return sims[-1]
    matches = [d for d in runs if spec in os.path.basename(d)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f'no run matching {spec!r} in {logs_dir()}')
    exact = [d for d in matches if os.path.basename(d) == spec]
    if exact:
        return exact[0]
    raise FileNotFoundError(f'{spec!r} matches several runs: '
                            + ', '.join(os.path.basename(d) for d in matches))
