import numpy as np
import pytest

from delaycp.mondrian import MondrianPID
from delaycp.online import PIDQuantileTracker
from delaycp.types import Action, Thresholds


def _stream(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.01).astype(int)
    p = np.where(y == 1, rng.beta(3, 2, n), rng.beta(1, 8, n))
    return p, y


def test_per_class_coverage_converges_and_pooled_hides_fraud() -> None:
    alpha_f, alpha_l = 0.1, 0.05
    p, y = _stream(300_000, seed=0)
    m = MondrianPID(alpha_f, alpha_l, lr=0.01)
    pooled = PIDQuantileTracker(0.05, lr=0.005)
    hit = {0: [], 1: []}
    pooled_hit = {0: [], 1: []}
    for pi, yi in zip(p, y, strict=True):
        th = m.thresholds
        q_pool = pooled.threshold
        s_true = 1.0 - pi if yi == 1 else pi
        hit[yi].append(s_true <= (th.q_1 if yi == 1 else th.q_0))
        pooled_hit[yi].append(s_true <= q_pool)
        m.update(pi, int(yi), th)
        pooled.update(s_true, q_pool)
    fraud_cov = np.mean(hit[1][-2000:])
    legit_cov = np.mean(hit[0][-50_000:])
    assert abs(legit_cov - (1 - alpha_l)) < 0.01
    assert abs(fraud_cov - (1 - alpha_f)) < 0.07
    # pooled tracker: marginal coverage fine (dominated by legit), fraud under-covered
    marginal = np.mean(pooled_hit[0] + pooled_hit[1])
    assert abs(marginal - 0.95) < 0.01
    assert np.mean(pooled_hit[1]) < 1 - alpha_f - 0.15


def test_action_mapping_at_boundaries() -> None:
    m = MondrianPID(0.1, 0.1, lr=0.1)
    m._trackers[0]._q_p = 0.3  # t_high
    m._trackers[1]._q_p = 0.4  # t_low = 0.6
    assert m.thresholds == Thresholds(0.3, 0.4)
    p = np.array([0.0, 0.3, 0.45, 0.6, 1.0])
    assert list(m.predict_actions(p)) == [
        Action.APPROVE,
        Action.APPROVE,
        Action.REVIEW,
        Action.DECLINE,
        Action.DECLINE,
    ]
    assert m.predict_sets(p).tolist() == [
        [True, False],
        [True, False],
        [False, False],
        [False, True],
        [False, True],
    ]
    # overlapping sets -> REVIEW
    m._trackers[0]._q_p = 0.7  # t_high
    m._trackers[1]._q_p = 0.6  # t_low = 0.4
    assert m.predict_actions(np.array([0.5]))[0] == Action.REVIEW
    assert m.predict_sets(np.array([0.5])).tolist() == [[True, True]]


def test_update_only_touches_true_class() -> None:
    m = MondrianPID(0.1, 0.1, lr=0.1)
    th = m.thresholds
    m.update(0.9, 1, th)
    assert m._trackers[0].n_updates == 0 and m._trackers[1].n_updates == 1
    # uses thresholds_used, not current: score 0.1 > q_used=0.05 is a miss
    before = m.thresholds.q_1
    m.update(0.9, 1, Thresholds(q_0=0.9, q_1=0.05))
    assert m.thresholds.q_1 > before


def test_small_class_warning() -> None:
    p, y = _stream(2000, seed=1)
    m = MondrianPID(0.1, 0.1, lr=0.01, min_class_count=50)
    with pytest.warns(UserWarning, match="class 1"):
        m.fit_calibration(p[:1000], y[:1000])
    big = MondrianPID(0.1, 0.1, lr=0.01, min_class_count=5)
    big.fit_calibration(p, y)
