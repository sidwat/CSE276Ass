"""Read detector metadata once; navigation stores only surveyed world poses."""
from pathlib import Path
import numpy as np
import yaml

# Optional tuning overrides remain supported, but need not clutter experiment.yaml.
DEFAULTS = {
    'filter': {'process_std': [.05, .05, .12], 'visual_std': [.06, .06, .12], 'nis_gate': 11.34},
    'vision': {'tf_wait': .05, 'max_age': .15, 'max_range': 3., 'min_margin': 30.},
    'control': {'blind_timeout': .5, 'pause_at_waypoint': 2., 'max_v': .12, 'max_w': .6},
}


def default_detector_config():
    from ament_index_python.packages import get_package_share_directory
    return str(Path(get_package_share_directory('apriltag_ros'))/'cfg/tags_36h11.yaml')


def load_detector_config(path):
    """Read the same file passed unchanged to apriltag_ros; never generate overrides."""
    with open(path) as stream:
        parameters = yaml.safe_load(stream)['/**']['ros__parameters']
    if parameters.get('pose_estimation_method', 'pnp') != 'pnp':
        raise ValueError('Navigation requires AprilTag pnp TF output (matching surveyed tag axes)')
    family = 'tag'+parameters.get('family', '36h11')
    tag = parameters.get('tag', {})
    ids, frames, sizes = tag.get('ids', []), tag.get('frames', []), tag.get('sizes', [])
    if (any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids) or
            (frames and len(frames) != len(ids)) or (sizes and len(sizes) != len(ids))):
        raise ValueError('AprilTag IDs must be unique; frames/sizes must match IDs')
    if any(not isinstance(f, str) or not f for f in frames) or len(set(frames)) != len(frames):
        raise ValueError('AprilTag frame names must be nonempty and unique')
    if any(not np.isfinite(s) or s <= 0 for s in [parameters.get('size', 1.), *sizes]):
        raise ValueError('AprilTag sizes must be finite and positive')
    return {'family': family, 'frames': dict(zip(ids, frames)),
            'max_hamming': parameters.get('max_hamming', 0)}


def valid_pose(pose):
    return (isinstance(pose, dict) and np.shape(pose.get('xyz')) == (3,) and
            np.shape(pose.get('rpy')) == (3,) and
            np.isfinite(pose['xyz'] + pose['rpy']).all())


def load_config(path, enable, detector_path=None):
    with open(path) as stream:
        cfg = yaml.safe_load(stream)
    for section, defaults in DEFAULTS.items():
        cfg[section] = {**defaults, **cfg.get(section, {})}
    cfg['_detector'] = load_detector_config(detector_path or default_detector_config())
    tags = {}
    if not isinstance(cfg.get('tags', {}), dict):
        raise ValueError('Use tags: {printed_id: {xyz: [...], rpy: [...]}}')
    for tag_id, pose in cfg.get('tags', {}).items():
        if type(tag_id) is not int or tag_id < 0:
            raise ValueError('Tag map keys must be nonnegative integer IDs')
        if pose is not None and not isinstance(pose, dict):
            raise ValueError(f'Invalid world pose for tag {tag_id}')
        if pose is None or pose.get('xyz') is None or pose.get('rpy') is None:
            continue  # unsurveyed tag
        if not valid_pose(pose):
            raise ValueError(f'Invalid world pose for tag {tag_id}')
        if cfg['_detector']['frames'] and tag_id not in cfg['_detector']['frames']:
            raise ValueError(f'Tag {tag_id} is not enabled in the AprilTag frame mapping')
        tags[tag_id] = pose
    m = cfg['motion']
    if not (m['speed_per_unit'] > 0 and m['wheel_base'] > 0 and
            0 < m['max_cmd'] <= .5 and abs(m['wheel_trim']) < 1):
        raise ValueError('Invalid motion calibration')
    mount = cfg['base_camera']
    complete = mount.get('xyz') is not None and mount.get('rpy') is not None
    if complete and not valid_pose(mount):
        raise ValueError('Invalid camera mounting transform')
    if enable and (not cfg.get('reviewed', False) or not complete or not tags):
        raise ValueError('Motion requires reviewed=true, measured camera mount and a surveyed tag map')
    return cfg, tags, complete
