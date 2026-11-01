"""One detector config, surveyed world poses only in navigation config."""
from pathlib import Path

import pytest
import yaml

from hw2_navigation.configuration import load_config, load_detector_config

DETECTOR = Path(__file__).parents[2]/'rubikpi_ros2/apriltag_ros/cfg/tags_36h11.yaml'
EXPERIMENT = Path(__file__).parents[1]/'config/experiment.yaml'


def test_existing_detector_names_and_sizes_are_read_without_modification():
    original = DETECTOR.read_bytes()
    metadata = load_detector_config(DETECTOR)
    params = yaml.safe_load(original)['/**']['ros__parameters']
    assert metadata['frames'] == dict(zip(params['tag']['ids'], params['tag']['frames']))
    assert metadata['max_hamming'] == params['max_hamming']
    assert DETECTOR.read_bytes() == original


def test_minimal_world_map_can_enable_motion_without_exactly_eight_tags(tmp_path):
    cfg = yaml.safe_load(EXPERIMENT.read_text())
    cfg['reviewed'] = True
    cfg['base_camera'] = {'xyz': [0, 0, .2], 'rpy': [0, 0, 0]}
    cfg['tags'] = {0: {'xyz': [1, 0, .2], 'rpy': [0, 0, 0]}}
    file = tmp_path/'experiment.yaml'
    file.write_text(yaml.safe_dump(cfg))
    loaded, tags, mount = load_config(file, True, DETECTOR)
    assert mount and set(tags) == {0}
    assert set(tags[0]) == {'xyz', 'rpy'}  # no size/frame/family duplication
    assert loaded['control']['max_v'] == .12


def test_empty_map_still_refuses_motion(tmp_path):
    cfg = yaml.safe_load(EXPERIMENT.read_text())
    cfg['reviewed'] = True
    cfg['base_camera'] = {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]}
    file = tmp_path/'experiment.yaml'
    file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='surveyed tag map'):
        load_config(file, True, DETECTOR)


def test_detector_frame_changes_are_used_without_nav_config_changes(tmp_path):
    detector = yaml.safe_load(DETECTOR.read_text())
    detector['/**']['ros__parameters']['tag'] = {
        'ids': [9], 'frames': ['my_tag_nine'], 'sizes': [.2]}
    detector_path = tmp_path/'detector.yaml'
    detector_path.write_text(yaml.safe_dump(detector))
    cfg = yaml.safe_load(EXPERIMENT.read_text())
    cfg['tags'] = {9: {'xyz': [1, 2, .3], 'rpy': [0, 0, 0]}}
    file = tmp_path/'experiment.yaml'
    file.write_text(yaml.safe_dump(cfg))
    loaded, tags, _ = load_config(file, False, detector_path)
    assert loaded['_detector']['frames'] == {9: 'my_tag_nine'} and 9 in tags
    # Use an explicit restricted fixture, rather than assuming the user's ID 9 is absent.
    restricted = tmp_path/'restricted.yaml'
    detector['/**']['ros__parameters']['tag'] = {'ids': [0], 'frames': ['tag_0']}
    restricted.write_text(yaml.safe_dump(detector))
    with pytest.raises(ValueError, match='not enabled'):
        load_config(file, False, restricted)


def test_optional_tuning_overrides_do_not_require_other_fields(tmp_path):
    cfg = yaml.safe_load(EXPERIMENT.read_text())
    cfg['control'] = {'max_v': .08}
    file = tmp_path/'experiment.yaml'
    file.write_text(yaml.safe_dump(cfg))
    loaded, _, _ = load_config(file, False, DETECTOR)
    assert loaded['control']['max_v'] == .08 and loaded['control']['max_w'] == .6
