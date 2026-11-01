"""Pose/TF association regressions without a ROS graph or physical camera."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as S

import numpy as np
import pytest

from hw2_navigation.configuration import load_config
from hw2_navigation.core import PoseFilter
from hw2_navigation.visual_localization import VisualLocalizer, transform_matrix


def fixture():
    cfg, _, _ = load_config(
        Path(__file__).parents[1]/'config/experiment.yaml', False,
        Path(__file__).parents[2]/'rubikpi_ros2/apriltag_ros/cfg/tags_36h11.yaml')
    tag = dict(xyz=[0, 0, 1], rpy=[0, 0, 0])
    stamp = S(sec=10, nanosec=0)
    header = S(frame_id='camera_frame', stamp=stamp)
    d = S(id=0, family='tag36h11', hamming=0, decision_margin=100.)
    msg = S(header=header, detections=[d])
    tf = S(header=header, child_frame_id='tag_0',
           transform=S(translation=S(x=0., y=0., z=1.), rotation=S(x=0., y=0., z=0., w=1.)))
    return cfg, tag, msg, tf


def test_tf_arrives_after_detection_and_is_fused_only_once():
    cfg, tag, msg, tf = fixture()
    localizer = VisualLocalizer(cfg, {0: tag}, np.eye(4))
    filter_ = PoseFilter()
    assert localizer.enqueue(msg, 10_000_000_000) == 'waiting_for_tag_tf'
    result = localizer.process(10_010_000_000, lambda *args: None, filter_)
    assert not result.accepted and len(localizer.pending) == 1
    calls = []
    def lookup(camera, frame, stamp):
        calls.append((camera, frame, stamp.sec))
        return tf
    result = localizer.process(10_050_000_000, lookup, filter_)
    assert result.accepted and result.age == pytest.approx(.05)
    assert calls == [('camera_frame', 'tag_0', 10)]
    assert np.allclose(filter_.x, 0)
    prior = filter_.P.copy()
    assert not localizer.process(10_060_000_000, lookup, filter_).accepted
    assert np.array_equal(filter_.P, prior)
    assert localizer.enqueue(msg, 10_060_000_000) == 'stale_or_out_of_order_image'


def test_missing_tf_expires_without_fusing_or_blocking():
    cfg, tag, msg, _ = fixture()
    localizer = VisualLocalizer(cfg, {0: tag}, np.eye(4))
    localizer.enqueue(msg, 10_000_000_000)
    result = localizer.process(10_200_000_000, lambda *args: None, PoseFilter())
    assert not result.accepted and not localizer.pending
    assert result.reason == 'expired_waiting_for_tag_tf'


@pytest.mark.parametrize('fault', ['timestamp', 'frame', 'quaternion', 'range', 'family'])
def test_invalid_observations_do_not_update_filter(fault):
    cfg, tag, msg, tf = deepcopy(fixture())
    if fault == 'timestamp':
        tf.header = S(frame_id='camera_frame', stamp=S(sec=9, nanosec=0))
    elif fault == 'frame':
        tf.child_frame_id = 'wrong_tag'
    elif fault == 'quaternion':
        tf.transform.rotation.w = 0.
    elif fault == 'range':
        tf.transform.translation.z = 10.
    else:
        msg.detections[0].family = 'tag25h9'
    localizer = VisualLocalizer(cfg, {0: tag}, np.eye(4))
    localizer.enqueue(msg, 10_000_000_000)
    f = PoseFilter()
    prior = f.P.copy()
    result = localizer.process(10_010_000_000, lambda *args: tf, f)
    assert not result.accepted
    assert np.array_equal(f.P, prior)


def test_only_one_correction_per_image_and_innovation_gate_retained():
    cfg, tag, msg, tf = fixture()
    msg.detections.append(deepcopy(msg.detections[0]))
    localizer = VisualLocalizer(cfg, {0: tag}, np.eye(4))
    localizer.enqueue(msg, 10_000_000_000)
    class Filter:
        calls = 0
        def correct(self, *args):
            self.calls += 1
            return self.calls == 2, 1.0
    f = Filter()
    result = localizer.process(10_010_000_000, lambda *args: tf, f)
    assert result.accepted and result.rejected == 1 and f.calls == 2


def test_quaternion_rotation_is_not_a_rodrigues_vector():
    _, _, _, tf = fixture()
    tf.transform.rotation.z = tf.transform.rotation.w = np.sqrt(.5)
    matrix = transform_matrix(tf.transform)
    assert np.allclose(matrix[:3, :3] @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(matrix[:3, 3], [0, 0, 1])


def test_one_missing_tf_does_not_prevent_using_another_valid_tag():
    cfg, tag, msg, tf = fixture()
    other = deepcopy(msg.detections[0])
    other.id = 1
    msg.detections.append(other)
    localizer = VisualLocalizer(cfg, {0: tag, 1: dict(tag)}, np.eye(4))
    localizer.enqueue(msg, 10_000_000_000)
    lookup = lambda camera, frame, stamp: tf if frame == 'tag_0' else None
    f = PoseFilter()
    assert not localizer.process(10_010_000_000, lookup, f).accepted
    assert localizer.process(10_060_000_000, lookup, f).accepted
