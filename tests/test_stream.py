import numpy as np
import pytest

from delaycp.mondrian import MondrianPID
from delaycp.stream import OnlineConformalClassifier


def _stream(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.where(y == 1, rng.beta(3, 2, n), rng.beta(1, 8, n))
    return p, y


def _run(
    p: np.ndarray, y: np.ndarray, tau: int, batch: int = 1, **kw: object
) -> tuple[OnlineConformalClassifier, list[int]]:
    clf = OnlineConformalClassifier(0.1, 0.05, lr=0.01, **kw)  # type: ignore[arg-type]
    actions: list[int] = []
    n = len(p)
    for start in range(0, n, batch):
        idx = np.arange(start, min(start + batch, n))
        t = idx.astype(float)
        actions += list(clf.predict(idx.tolist(), t, p[idx]))
        # labels for step s - tau arrive at step s
        due = idx - tau
        due = due[due >= 0]
        if due.size:
            clf.add_labels(due.tolist(), y[due], (due + tau).astype(float))
        clf.advance(float(idx[-1]))
    return clf, actions


def test_tau_zero_matches_direct_mondrian() -> None:
    p, y = _stream(5_000, seed=1)
    clf, actions = _run(p, y, tau=0, history_size=10)
    direct = MondrianPID(0.1, 0.05, lr=0.01)
    expected = []
    for pi, yi in zip(p, y, strict=True):
        th = direct.thresholds
        expected.append(int(direct.predict_actions([pi])[0]))
        direct.update(float(pi), int(yi), th)
    assert actions == expected
    assert clf.thresholds == direct.thresholds
    assert clf.pending_count == 0


def test_delayed_coverage_converges() -> None:
    n, tau = 200_000, 200
    p, y = _stream(n, seed=2)
    clf, _ = _run(p, y, tau=tau, batch=10, history_size=n)
    assert clf.pending_count > 0
    ups = clf.history.updates
    ys = np.array([u.y for u in ups])
    miss = np.array([u.miscovered for u in ups])
    assert abs(miss[ys == 0][-50_000:].mean() - 0.05) < 0.01
    assert abs(miss[ys == 1][-5_000:].mean() - 0.1) < 0.04


def test_history_is_bounded_and_optional() -> None:
    p, y = _stream(500, seed=3)
    off, _ = _run(p, y, tau=5)
    assert len(off.history.predictions) == 0 and len(off.history.updates) == 0
    on, actions = _run(p, y, tau=5, history_size=50)
    assert len(on.history.predictions) == 50 and len(on.history.updates) == 50
    last = on.history.predictions[-1]
    assert last.pred_time == 499.0 and last.action == actions[-1]


def test_miscoverage_uses_thresholds_in_effect_at_prediction() -> None:
    clf = OnlineConformalClassifier(0.1, 0.1, lr=0.5, history_size=10)
    th0 = clf.thresholds  # q_0 = q_1 = 0.9
    clf.predict(["a", "b"], [0.0, 1.0], [0.95, 0.5])
    clf.add_labels(["b"], [0], [2.0])
    clf.advance(2.0)  # moves q_0 away from th0
    clf.add_labels(["a"], [0], [3.0])
    clf.advance(3.0)
    rec = clf.history.updates[-1]
    assert clf.history.predictions[0].thresholds_used == th0
    assert rec.miscovered  # 0.95 > th0.q_0 = 0.9


def test_maturity_imputes_and_counts() -> None:
    clf = OnlineConformalClassifier(0.1, 0.1, lr=0.01, maturity=5.0, history_size=5)
    clf.predict(["a"], [0.0], [0.2])
    assert clf.advance(4.0) == 0
    assert clf.advance(5.0) == 1
    assert clf.history.updates[0].matured and clf.history.updates[0].y == 0


def test_validation() -> None:
    clf = OnlineConformalClassifier(0.1, 0.1, lr=0.01)
    clf.predict([1, 2], [1.0, 2.0], [0.1, 0.2])
    with pytest.raises(ValueError, match="non-decreasing"):
        clf.predict([3], [1.5], [0.1])
    with pytest.raises(ValueError, match="equal length"):
        clf.predict([4, 5], [3.0], [0.1])
    with pytest.raises(ValueError, match="history_size"):
        OnlineConformalClassifier(0.1, 0.1, lr=0.01, history_size=-1)
