"""
One entry point for HW3 runs in <workspace>/hw3_logs (see robot_control/hw3_paths.py).

  ros2 run robot_control hw3 list [--new]              runs, newest last, with their results
  ros2 run robot_control hw3 eval [RUN ...] [--start X Y TH] [--gt FILE]
                                                       maps + per-tag error vs ground truth
  ros2 run robot_control hw3 map [RUN]                 same as eval, prints the map to look at
  ros2 run robot_control hw3 sim square|octagon [--seed N] [--dof 2|3]   simulate, then eval
  ros2 run robot_control hw3 compare [RUN ...|all]     evaluate several runs, comparison table
  ros2 run robot_control hw3 diagnose [RUN]            did the robot move? why are sightings
                                                       rejected? wheel commands from the bag
  ros2 run robot_control hw3 gt                        ground-truth file status vs tags seen

RUN is a folder, a folder name or any unique part of one ("square", "04-42"), "latest"
(newest robot run, the default) or "latest-sim".
"""

import argparse
import bisect
import json
import math
import os
import shutil
import subprocess
import sys

from robot_control import hw3_paths


# --- helpers ------------------------------------------------------------------------------------

def _load_gt(path):
    from robot_control.localization import load_landmarks
    return load_landmarks(path)


def _info(run_dir):
    with open(os.path.join(run_dir, 'run_info.json')) as f:
        return json.load(f)


def _summary(run_dir):
    try:
        with open(os.path.join(run_dir, 'eval', 'summary.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _is_evaluated(run_dir):
    s = os.path.join(run_dir, 'eval', 'summary.json')
    return os.path.exists(s) and os.path.getmtime(s) >= os.path.getmtime(
        os.path.join(run_dir, 'run_info.json'))


def _cm(v):
    return '-' if v is None or (isinstance(v, float) and not math.isfinite(v)) \
        else f'{100 * v:.1f}'


def _write_start_pose(run_dir, start):
    from robot_control.slam_logger import START_POSE_TEMPLATE
    with open(os.path.join(run_dir, 'start_pose.txt'), 'w') as f:
        f.write(START_POSE_TEMPLATE.format(pose='{:.4f}, {:.4f}, {:.4f}'.format(*start)))


def _choose_gt(run_dir, gt_arg, quiet=False):
    """(gt dict, path or None, note). Template ground truth is only used for simulations."""
    info = _info(run_dir)
    if gt_arg:
        path = os.path.abspath(os.path.expanduser(gt_arg))
    elif info.get('simulated') and os.path.exists(os.path.join(run_dir, 'landmarks_gt_sim.txt')):
        path = os.path.join(run_dir, 'landmarks_gt_sim.txt')  # the map it was simulated in
    elif info.get('simulated') and info.get('gt_file') and os.path.exists(info['gt_file']):
        path = info['gt_file']  # a simulation is judged against the map it was simulated in
    else:
        path = hw3_paths.default_gt_file()
    if not os.path.exists(path):
        return {}, None, f'ground truth {path} not found: start-frame map only'
    if hw3_paths.gt_is_template(path) and not info.get('simulated'):
        return {}, None, (f'ground truth {path} is still the TEMPLATE (not measured): only the '
                          f'start-frame map is drawn. Fill in the tape-measured tag positions '
                          f'and re-run "hw3 eval".')
    return _load_gt(path), path, ''


# --- list ---------------------------------------------------------------------------------------

def cmd_list(args):
    runs = hw3_paths.all_runs()
    if args.new:
        runs = [r for r in runs if not _is_evaluated(r)]
    print(f'Runs in {hw3_paths.logs_dir()}' + (' (not yet evaluated)' if args.new else ''))
    if not runs:
        print('  (none)')
        return 0
    print(f'  {"run":44s} {"dur":>5s} {"done":>4s} {"N":>2s} {"err raw":>8s} '
          f'{"err aln":>8s} {"bag":>3s}  start pose')
    for r in runs:
        info, s = _info(r), _summary(r)
        summ = info.get('summary', {})
        start = _start_status(r)
        print(f'  {os.path.basename(r):44s} {summ.get("duration_s", 0):5.0f} '
              f'{"yes" if summ.get("completed_all_waypoints") else "no":>4s} '
              f'{summ.get("n_landmarks", "?"):>2} '
              f'{_cm(s["raw"].get("mean")) if s else "-":>8s} '
              f'{_cm(s["aligned"].get("mean")) if s else "-":>8s} '
              f'{"yes" if os.path.isdir(os.path.join(r, "bag")) else "no":>3s}  {start}')
    print('  err raw = mean landmark error via the start pose, err aln = after best-fit '
          'alignment [cm]; "-" = not evaluated against a measured ground truth')
    return 0


def _start_status(run_dir):
    from robot_control.evaluate_slam import read_start_pose
    p = read_start_pose(run_dir)
    return 'MISSING' if p is None else '({:.3f}, {:.3f}, {:.3f})'.format(*p)


# --- eval / map ---------------------------------------------------------------------------------

def evaluate_run(run_dir, gt_arg=None, start=None, arena=hw3_paths.ARENA_SIDE, quiet=False):
    from robot_control import evaluate_slam as ev
    if start is not None:
        _write_start_pose(run_dir, start)
    gt, gt_path, note = _choose_gt(run_dir, gt_arg)
    if gt_path:
        shutil.copy(gt_path, os.path.join(run_dir, 'landmarks_gt_used.txt'))
    run = ev.Run(run_dir)
    if run.start is None and gt:
        path = run.info.get('path')
        if path in hw3_paths.NOMINAL_START:
            run.start = hw3_paths.NOMINAL_START[path]
            note += (f' start_pose.txt is empty: used the nominal {path} start {run.start}; '
                     f'pass --start X Y TH with the measured one.')
    out = os.path.join(run_dir, 'eval')
    # these figures are only valid with this evaluation's ground truth
    for stale in ('map_world.png', 'map_aligned.png', 'map_overlay.png'):
        if os.path.exists(os.path.join(out, stale)):
            os.remove(os.path.join(out, stale))
    summary = ev.evaluate(run, gt, out, arena)
    figures = {name: os.path.join(out, name) for name in (
        'map_overlay.png', 'map_world.png', 'map_aligned.png', 'map_start_frame.png',
        'timeline.png', 'laps.png',
        'convergence.png', 'nis.png') if os.path.exists(os.path.join(out, name))}
    main_map = (figures.get('map_overlay.png') or figures.get('map_world.png')
                or figures.get('map_start_frame.png'))
    print(f'\n== {os.path.basename(run_dir)}')
    if note.strip():
        print(f'  NOTE: {note.strip()}')
    print(f'  ground truth: {gt_path or "none"}')
    print(f'  start pose (world): {run.start}')
    _print_tag_table(summary)
    print(f'  map: {main_map}')
    print(f'  report: {os.path.join(out, "results.md")}')
    if not quiet:
        print('  figures: ' + ', '.join(sorted(figures)))
    return summary, main_map


def _print_tag_table(summary):
    rows = summary.get('per_landmark_raw') or []
    aligned = {r['tag']: r for r in summary.get('per_landmark_aligned') or []}
    if not rows and not aligned:
        print(f'  landmarks: {summary["landmark_ids"]} (no ground-truth comparison)')
        return
    print('  tag  est x   est y   GT x    GT y    err [cm]  err aligned [cm]  std x,y [cm]  '
          'updates')
    for r in rows or aligned.values():
        a = aligned.get(r['tag'])
        print(f'  {r["tag"]:<4} {r["x"]:6.3f}  {r["y"]:6.3f}  {r["gt_x"]:6.3f}  {r["gt_y"]:6.3f}  '
              f'{_cm(r["err"]) if rows else "-":>8s}  {_cm(a["err"]) if a else "-":>16s}  '
              f'{_cm(r["std_x"])}, {_cm(r["std_y"]):<6s}  {int(r["sightings"])}')
    outl = [r['tag'] for r in summary.get('per_landmark_aligned') or []
            if not r.get('fit_inlier', True)]
    if outl:
        inl = [r['err'] for r in summary.get('per_landmark_aligned') if r.get('fit_inlier', True)]
        raw_in = [r['err'] for r in rows if r['tag'] not in outl]
        print(f'  outlier tags (ignored by the alignment fit): {outl}; without them: '
              f'{_cm(sum(raw_in) / len(raw_in)) if raw_in else "-"} cm via start pose, '
              f'{_cm(sum(inl) / len(inl))} cm after alignment ({len(inl)} tags)')
    raw, al = summary.get('raw', {}), summary.get('aligned', {})
    print(f'  MEAN landmark error: {_cm(raw.get("mean"))} cm via start pose, '
          f'{_cm(al.get("mean"))} cm after best-fit alignment '
          f'({raw.get("n") or al.get("n")} tags matched)')


def cmd_eval(args):
    runs = hw3_paths.all_runs() if args.all else [
        hw3_paths.resolve_run(r) for r in (args.runs or ['latest'])]
    if args.start and len(runs) != 1:
        sys.exit('--start applies to one run only')
    for r in runs:
        evaluate_run(r, args.gt, tuple(args.start) if args.start else None, args.arena,
                     args.quiet)
    return 0


# --- sim ----------------------------------------------------------------------------------------

def cmd_sim(args):
    cmd = [sys.executable, '-m', 'robot_control.simulate_slam', '--path', args.path,
           '--seed', str(args.seed)]
    if args.dof:
        cmd += ['--landmark-dof', str(args.dof)]
    if args.gt:
        cmd += ['--gt', args.gt]
    subprocess.run(cmd, check=True)
    evaluate_run(hw3_paths.resolve_run('latest-sim'), args.gt)
    return 0


# --- compare ------------------------------------------------------------------------------------

def cmd_compare(args):
    from robot_control.evaluate_slam import write_comparison
    if not args.runs or args.runs == ['all']:
        runs = hw3_paths.all_runs(include_sim=args.sim)
    else:
        runs = [hw3_paths.resolve_run(r) for r in args.runs]
    summaries = [evaluate_run(r, args.gt, quiet=True)[0] for r in runs]
    out = os.path.join(hw3_paths.logs_dir(), f'comparison_{hw3_paths.timestamp()}')
    write_comparison(summaries, out)
    with open(os.path.join(out, 'comparison.md')) as f:
        print(f.read())
    return 0


# --- gt -----------------------------------------------------------------------------------------

def cmd_gt(args):
    path = args.gt or hw3_paths.default_gt_file()
    print(f'Ground truth file: {path}')
    if hw3_paths.gt_is_template(path):
        print('  STATUS: TEMPLATE - positions are placeholders, not measurements. Replace every '
              'line with "id, x, y, yaw" of your tags (world frame: origin at the arena\'s '
              'south-west corner, x east, y north, yaw = direction the tag faces) and delete '
              'the word TEMPLATE from the header.')
    gt = _load_gt(path)
    print(f'  {len(gt)} tags: ' + ', '.join(
        f'{t}=({x:.3f}, {y:.3f}, {math.degrees(yaw):.0f} deg)' for t, (x, y, yaw) in
        sorted(gt.items())))
    seen = set()
    for r in hw3_paths.all_runs(include_sim=False):
        seen |= set(_info(r).get('summary', {}).get('landmark_ids', []))
    if seen:
        print(f'  tags mapped in robot runs: {sorted(seen)}')
        print(f'  mapped but NOT in the ground truth: {sorted(seen - set(gt)) or "none"}')
        print(f'  in the ground truth but never mapped: {sorted(set(gt) - seen) or "none"}')
    return 0


# --- diagnose -----------------------------------------------------------------------------------

def _segments(traj):
    """Consecutive trajectory rows with the same phase: [(phase, t0, t1, dist, turn)]."""
    segs = []
    for a, b in zip(traj, traj[1:]):
        dt = b['t'] - a['t']
        if not segs or segs[-1][0] != a['phase']:
            segs.append([a['phase'], a['t'], b['t'], 0.0, 0.0])
        s = segs[-1]
        s[2] = b['t']
        s[3] += abs(a['v']) * dt
        s[4] += a['w'] * dt
    return segs


def _sightings_by_tag(meas):
    by = {}
    for m in meas:
        by.setdefault(int(m['tag']), []).append(m)
    return by


def _near(ms, times, t, window):
    """Mean (range, bearing) of sightings within window of t, or None."""
    i, j = bisect.bisect_left(times, t - window), bisect.bisect_right(times, t + window)
    if j - i < 1:
        return None
    sel = ms[i:j]
    return (sum(m['range'] for m in sel) / len(sel), sum(m['bearing'] for m in sel) / len(sel))


def _observed_motion(segs, meas, kind):
    """
    Per segment: (commanded, camera-observed) motion from tags seen at both ends.

    drive: |change in range| (a lower bound of the distance travelled);
    rotate: |change in bearing| vs the commanded turn.
    """
    by = _sightings_by_tag(meas)
    times = {t: [m['t'] for m in ms] for t, ms in by.items()}
    out = []
    for phase, t0, t1, dist, turn in segs:
        if kind == 'drive' and (phase != 'drive' or dist < 0.2):
            continue
        if kind == 'rotate' and (not phase.startswith('rotate') or abs(turn) < 0.3):
            continue
        best = None
        for tag, ms in by.items():
            a, b = _near(ms, times[tag], t0, 0.8), _near(ms, times[tag], t1, 0.8)
            if a and b:
                d = abs(b[0] - a[0]) if kind == 'drive' else abs(
                    math.atan2(math.sin(b[1] - a[1]), math.cos(b[1] - a[1])))
                best = d if best is None else max(best, d)
        if best is not None:
            out.append((t0, t1, dist if kind == 'drive' else abs(turn), best))
    return out


def _bag_commands(run_dir):
    """[(t, L, R)] from bag/ (motor_commands), or None if there is no bag or no rosbag2_py."""
    bag = os.path.join(run_dir, 'bag')
    if not os.path.isdir(bag):
        return None
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from std_msgs.msg import Float32MultiArray
    except ImportError:
        return None
    meta = os.path.join(bag, 'metadata.yaml')
    storage = 'mcap'
    if os.path.exists(meta):
        with open(meta) as f:
            storage = 'sqlite3' if 'sqlite3' in f.read() else 'mcap'
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=storage),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/motor_commands']))
    rows, t0 = [], None
    while reader.has_next():
        _, data, t = reader.read_next()
        t0 = t0 or t
        m = deserialize_message(data, Float32MultiArray)
        if len(m.data) >= 2:
            rows.append(((t - t0) * 1e-9, float(m.data[0]), float(m.data[1])))
    return rows


def cmd_diagnose(args):
    from robot_control.evaluate_slam import Run
    run_dir = hw3_paths.resolve_run(args.run)
    run = Run(run_dir)
    info, summ = run.info, run.info.get('summary', {})
    traj, meas = run.trajectory, run.measurements
    p = info.get('params', {})
    print(f'== diagnose {os.path.basename(run_dir)}')
    print(f'  path {info.get("path")}, duration {summ.get("duration_s", 0):.0f} s, '
          f'all waypoints reached: {summ.get("completed_all_waypoints")}, waypoints logged '
          f'{len(run.waypoints)}')
    verdicts = []

    segs = _segments(traj) if len(traj) > 1 else []
    by_phase = {}
    for phase, t0, t1, dist, turn in segs:
        acc = by_phase.setdefault(phase, [0.0, 0.0, 0.0])
        acc[0] += t1 - t0
        acc[1] += dist
        acc[2] += abs(turn)
    print('\n  Time and COMMANDED motion per controller phase:')
    for phase, (t, d, th) in by_phase.items():
        print(f'    {phase:18s} {t:6.1f} s   distance {d:6.2f} m   turn '
              f'{math.degrees(th):7.0f} deg')
    if by_phase.get('search', [0])[0] > 0.2 * max(summ.get('duration_s', 1), 1):
        verdicts.append('More than 20% of the run was spent in SEARCH: the filter repeatedly '
                        'lost confidence in its heading.')

    spu, wb = float(p.get('speed_per_unit', 1.26)), float(p.get('wheel_base', 0.68))
    trim = float(p.get('wheel_trim', 0.0))
    floor = float(p.get('min_wheel_cmd', 0.0) or 0.0)
    print('\n  Wheel commands implied by the controller limits (before trim'
          + (f', floor min_wheel_cmd {floor:.2f}' if floor else '') + '):')
    for name, v, w in (('max_v (driving)', p.get('max_v'), 0.0),
                       ('min_v (end of a leg)', p.get('min_v'), 0.0),
                       ('max_w (turn in place)', 0.0, p.get('max_w')),
                       ('min_w (turn in place)', 0.0, p.get('min_w'))):
        if v is None or w is None:
            continue
        cmd = max((abs(float(v)) + abs(float(w)) * wb / 2.0) / spu, floor)
        flag = '   <-- below the dead-band' if cmd < args.deadband else ''
        print(f'    {name:24s} -> |wheel| {cmd:.3f}{flag}')
        if cmd < args.deadband and 'driv' in name:
            verdicts.append(f'max_v = {p.get("max_v")} m/s gives wheel commands of only '
                            f'{cmd:.3f} < dead-band {args.deadband}: the robot cannot drive. '
                            f'Raise max_v to at least {args.deadband * spu:.2f} m/s.')
        elif cmd < args.deadband and 'end of a leg' in name:
            verdicts.append(f'min_v = {p.get("min_v")} m/s gives {cmd:.3f} < dead-band: the end '
                            f'of every leg stalls while the filter thinks it is moving. Raise '
                            f'min_v to about {args.deadband * spu:.2f} m/s.')

    drives = _observed_motion(segs, meas, 'drive')
    if drives:
        stalled = [d for d in drives if d[3] < max(0.05, 0.15 * d[2])]
        print(f'\n  Drive segments where the camera can check the motion: {len(drives)}')
        for t0, t1, cmd, obs in drives[:12]:
            print(f'    t {t0:6.1f}-{t1:6.1f} s  commanded {cmd:5.2f} m   camera saw range '
                  f'change {obs:5.2f} m' + ('   <-- DID NOT MOVE' if obs < max(
                      0.05, 0.15 * cmd) else ''))
        if len(stalled) >= max(1, len(drives) // 2):
            verdicts.append(f'In {len(stalled)}/{len(drives)} drive segments the tag ranges did '
                            f'not change although the controller commanded motion: the robot '
                            f'did not move (wheel commands below the motor dead-band, low '
                            f'battery, or motor board not powered).')
    turns = _observed_motion(segs, meas, 'rotate')
    if turns:
        ratios = sorted(o / c for _, _, c, o in turns if c > 0)
        med = ratios[len(ratios) // 2]
        print(f'\n  Turns checked against tag bearings: {len(turns)}, median actual/commanded '
              f'= {med:.2f} (only turns where one tag stays in view at both ends)')
        if med < 0.6:
            verdicts.append(f'Turns achieve only about {100 * med:.0f}% of the commanded angle: '
                            f'raise min_w / max_w or recalibrate wheel_base.')

    kinds = {}
    for m in meas:
        kinds[m['kind']] = kinds.get(m['kind'], 0) + 1
    upd, rej = kinds.get('update', 0), kinds.get('rejected', 0)
    print(f'\n  Sightings: {kinds}')
    if upd + rej:
        print(f'  Re-observations rejected by the Mahalanobis gate: {rej}/{upd + rej} '
              f'({100 * rej / (upd + rej):.0f}%), gate {p.get("gate")}')
        d2 = sorted(m['mahalanobis2'] for m in meas if m['kind'] == 'rejected')
        if d2:
            print(f'  median Mahalanobis^2 of rejected sightings: {d2[len(d2) // 2]:.1f}')
        first_bad = None
        window = 25.0
        tmax = max(m['t'] for m in meas)
        print(f'  per {window:.0f} s window (update / rejected):')
        t = 0.0
        while t <= tmax:
            u = sum(1 for m in meas if m['kind'] == 'update' and t <= m['t'] < t + window)
            r = sum(1 for m in meas if m['kind'] == 'rejected' and t <= m['t'] < t + window)
            if u + r:
                print(f'    {t:5.0f}-{t + window:5.0f} s  {u:4d} / {r:4d}')
            if first_bad is None and r > max(10, 2 * u):
                first_bad = t
            t += window
        if rej > upd:
            verdicts.append(
                'Most re-observations were rejected. The gate is working as intended: it '
                'means the predicted pose (from COMMANDED wheel speeds) disagrees with the '
                'camera, i.e. the robot did not do what was commanded'
                + (f' (from about t = {first_bad:.0f} s).' if first_bad is not None else '.'))
    if traj:
        hs = max(r['std_theta'] for r in traj)
        print(f'\n  Max heading std: {math.degrees(hs):.0f} deg')
        if hs > math.radians(45):
            verdicts.append(f'Heading std reached {math.degrees(hs):.0f} deg: landmarks added in '
                            f'that state (see results.md, landmark discovery) are unreliable.')
        odom_end = (traj[-1]['odom_x'], traj[-1]['odom_y'])
        print(f'  Dead reckoning ended at ({odom_end[0]:.2f}, {odom_end[1]:.2f}) m, the filter '
              f'at ({traj[-1]["x"]:.2f}, {traj[-1]["y"]:.2f}) m (true end of a full path: 0, 0)')

    try:
        cmds = _bag_commands(run_dir)
    except RuntimeError as e:  # bag still being recorded (no metadata yet) or damaged
        print(f'\n  (bag not readable yet: {str(e).splitlines()[0]})')
        cmds = False
    if cmds:
        moving = [(abs(L), abs(R)) for _, L, R in cmds if abs(L) > 1e-3 or abs(R) > 1e-3]
        weak = [x for x in moving if max(x) < args.deadband]
        print(f'\n  Bag /motor_commands: {len(cmds)} messages, {len(moving)} non-zero, '
              f'{len(weak)} ({100 * len(weak) / max(len(moving), 1):.0f}%) with both wheels '
              f'below the dead-band {args.deadband}')
        hist = {}
        for _, L, R in cmds:
            k = (round(L, 2), round(R, 2))
            hist[k] = hist.get(k, 0) + 1
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:8]
        print('  most frequent [L, R] (as sent, after trim): '
              + ', '.join(f'[{k[0]:+.2f}, {k[1]:+.2f}] x{n}' for k, n in top))
        if len(weak) > 0.3 * max(len(moving), 1):
            verdicts.append(f'{100 * len(weak) / len(moving):.0f}% of the non-zero wheel '
                            f'commands in the bag are below {args.deadband}.')
    elif cmds is None:
        print('\n  (no bag/ in this run: record with record_bag:=true to check wheel commands)')
    print(f'  wheel_trim {trim:+.2f}: sent L = cmd x {1 - trim:.2f}, R = cmd x {1 + trim:.2f}')

    gt_path = hw3_paths.default_gt_file()
    print(f'\n  start pose: {_start_status(run_dir)}; ground truth: '
          f'{"TEMPLATE (not measured)" if hw3_paths.gt_is_template(gt_path) else gt_path}')
    print('\n  VERDICT:')
    for v in verdicts or ['No obvious problem found.']:
        print(f'   - {v}')
    return 0


# --- main ---------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog='hw3', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('list', help='list runs')
    p.add_argument('--new', action='store_true', help='only runs not evaluated yet')
    p.set_defaults(func=cmd_list)

    for name in ('eval', 'map'):
        p = sub.add_parser(name, help='evaluate runs: maps, per-tag error vs ground truth')
        p.add_argument('runs', nargs='*', help='runs (default: latest)')
        p.add_argument('--all', action='store_true', help='every run in the logs folder')
        p.add_argument('--gt', help='ground-truth file (default: config/landmarks_hw3_gt.txt)')
        p.add_argument('--start', type=float, nargs=3, metavar=('X0', 'Y0', 'TH0'),
                       help='measured world start pose; also written to start_pose.txt')
        p.add_argument('--arena', type=float, default=hw3_paths.ARENA_SIDE,
                       help='arena side [m], 0 hides it')
        p.add_argument('--quiet', action='store_true')
        p.set_defaults(func=cmd_eval)

    p = sub.add_parser('sim', help='simulate a run offline, then evaluate it')
    p.add_argument('path', choices=sorted(hw3_paths.NOMINAL_START))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dof', type=int, choices=(2, 3))
    p.add_argument('--gt', help='true tag positions for the simulation')
    p.set_defaults(func=cmd_sim)

    p = sub.add_parser('compare', help='evaluate several runs and tabulate them')
    p.add_argument('runs', nargs='*', help='runs, or "all" (default: all robot runs)')
    p.add_argument('--sim', action='store_true', help='with "all": include simulations')
    p.add_argument('--gt')
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser('diagnose', help='health check of one run')
    p.add_argument('run', nargs='?', default='latest')
    p.add_argument('--deadband', type=float, default=0.15,
                   help='smallest wheel command that moves the robot (measure it!)')
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser('gt', help='ground-truth file status')
    p.add_argument('--gt')
    p.set_defaults(func=cmd_gt)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f'hw3: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
