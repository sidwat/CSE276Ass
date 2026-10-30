import math
import numpy as np
from hw2_navigation.core import PoseFilter, base_pose, transform, wrap
from hw2_navigation.simulate import run


def test_full_transform_chain_recovers_base_with_optical_mount():
    wb = transform([.6, 1.2, 0], [0, 0, .8])
    bc = transform([.15, 0, .22], [-math.pi/2, 0, -math.pi/2])
    wt = transform([2, 2, .4], [math.pi/2, 0, math.pi])
    ct = np.linalg.inv(wb @ bc) @ wt
    assert np.allclose(base_pose(wt, ct, bc), [.6, 1.2, .8])


def test_yaw_wrap_outlier_rejection_and_covariance():
    f = PoseFilter()
    f.x[2] = math.pi-.01
    ok, _ = f.correct([0, 0, -math.pi+.01], np.diag([.01]*3))
    assert ok and abs(wrap(f.x[2]-math.pi)) < .02
    prior = f.x.copy()
    ok, _ = f.correct([50, 50, 0], np.eye(3)*.01)
    assert not ok and np.allclose(prior, f.x)
    assert np.linalg.eigvalsh(f.P).min() > 0
    assert not f.predict(1, 0, 1)


def test_closed_loop_bias_and_dropout():
    _, baseline = run('baseline')
    rows, filtered = run('filtered')
    assert filtered['complete']
    assert len(filtered['waypoint_errors']) == 4
    assert filtered['waypoint_errors'][-1]['translation_error_m'] < .15
    assert (filtered['waypoint_errors'][-1]['translation_error_m'] <
            baseline['waypoint_errors'][-1]['translation_error_m'])
    blind = [r for r in rows if 25.7 < r[0] < 27]
    assert blind and all(r[7] == r[8] == 0 for r in blind)
