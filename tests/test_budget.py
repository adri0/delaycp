import math
from typing import Any

import numpy as np
import pytest

from delaycp.budget import BudgetController
from delaycp.stream import OnlineConformalClassifier
from delaycp.types import Action

BUDGET = 0.02


def _stream(n: int, seed: int, drift_at: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """2% fraud; after ``drift_at`` fraud moves to much lower scores."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.02).astype(int)
    fraud = rng.beta(3, 2, n)
    if drift_at is not None:
        fraud[drift_at:] = rng.beta(1, 6, n - drift_at)
    p = np.where(y == 1, fraud, rng.beta(1, 8, n))
    return p, y


def _run(
    clf: OnlineConformalClassifier,
    p: np.ndarray,
    y: np.ndarray,
    tau: int,
    batch: int = 10,
    probe_at: tuple[int, ...] = (),
) -> tuple[np.ndarray, dict[int, tuple[float, int]]]:
    """Labels for step ``s`` arrive at ``s + tau``; probe coverage after given steps."""
    actions = np.empty(len(p), dtype=int)
    probes: dict[int, tuple[float, int]] = {}
    for start in range(0, len(p), batch):
        idx = np.arange(start, min(start + batch, len(p)))
        actions[idx] = clf.predict(idx.tolist(), idx.astype(float), p[idx])
        due = idx - tau
        due = due[due >= 0]
        if due.size:
            clf.add_labels(due.tolist(), y[due], (due + tau).astype(float))
        clf.advance(float(idx[-1]))
        for s in probe_at:
            if s in idx:
                est = clf.estimated_fraud_coverage(300)
                probes[s] = (est.coverage, est.n_labels)
    return actions, probes


def _budget_clf(**kw: Any) -> OnlineConformalClassifier:
    return OnlineConformalClassifier(
        0.1, 0.05, lr=0.01, mode="budget", review_budget=BUDGET, budget_lr=0.02, **kw
    )


def test_review_rate_converges_to_budget() -> None:
    p, y = _stream(100_000, seed=0)
    clf = _budget_clf()
    actions, _ = _run(clf, p, y, tau=500)
    review = actions == Action.REVIEW
    assert abs(review[20_000:].mean() - BUDGET) < 0.002
    th = clf.thresholds
    assert th.t_low <= th.t_high


def test_drift_lowers_estimated_coverage_but_not_review_rate() -> None:
    n, drift, tau = 120_000, 60_000, 5_000
    p, y = _stream(n, seed=1, drift_at=drift)
    clf = _budget_clf()
    probe_before = drift - 10
    probe_lag = drift + tau - 10  # drift has happened, but its labels are not out yet
    probe_after = n - 1
    actions, probes = _run(clf, p, y, tau=tau, probe_at=(probe_before, probe_lag, probe_after))
    review = actions == Action.REVIEW

    cov_before, n_before = probes[probe_before]
    cov_lag, _ = probes[probe_lag]
    cov_after, n_after = probes[probe_after]
    assert n_before == n_after == 300
    assert cov_before > 0.6
    assert abs(cov_lag - cov_before) < 0.1  # coverage drop only visible after the delay
    assert cov_after < cov_before - 0.3

    # review rate stays on budget before and after the drift
    assert abs(review[20_000:drift].mean() - BUDGET) < 0.002
    assert abs(review[drift + 5_000 :].mean() - BUDGET) < 0.002


def test_coverage_mode_tracks_fraud_instead() -> None:
    # Same drift in coverage mode: fraud coverage recovers, review rate blows up.
    n, drift, tau = 120_000, 60_000, 5_000
    p, y = _stream(n, seed=1, drift_at=drift)
    clf = OnlineConformalClassifier(0.1, 0.05, lr=0.01)
    actions, probes = _run(clf, p, y, tau=tau, probe_at=(n - 1,))
    assert probes[n - 1][0] > 0.8
    assert (actions[-20_000:] == Action.REVIEW).mean() > 5 * BUDGET


def test_budget_mode_still_tracks_legit_coverage() -> None:
    p, y = _stream(100_000, seed=2)
    clf = _budget_clf(history_size=100_000)
    _run(clf, p, y, tau=500)
    legit_miss = np.array([u.miscovered for u in clf.history.updates if u.y == 0])
    assert abs(legit_miss[-50_000:].mean() - 0.05) < 0.01


def test_fraud_labels_do_not_move_thresholds_in_budget_mode() -> None:
    clf = _budget_clf()
    clf.predict(["a"], [0.0], [0.99])
    th = clf.thresholds
    clf.add_labels(["a"], [1], [1.0])
    assert clf.advance(1.0) == 1
    assert clf.thresholds == th
    assert clf.estimated_fraud_coverage(10) == (1.0, 1)


def test_calibration_sets_initial_review_rate() -> None:
    p, y = _stream(20_000, seed=3)
    clf = _budget_clf()
    clf.fit_calibration(p, y)
    th = clf.thresholds
    rate = np.mean((p >= th.t_low) & (p <= th.t_high))
    assert abs(rate - BUDGET) < 1e-3


def test_estimated_coverage_window_and_empty() -> None:
    clf = OnlineConformalClassifier(0.1, 0.1, lr=0.01, fraud_log_size=5)
    cov, n = clf.estimated_fraud_coverage(3)
    assert math.isnan(cov) and n == 0
    # t_low starts at 0.1: 0.05 is auto-approved (miss), the rest are covered
    clf.predict(list(range(6)), np.arange(6.0), [0.05, 0.5, 0.5, 0.05, 0.5, 0.5])
    clf.add_labels(list(range(6)), [1] * 6, np.arange(6.0))
    clf.advance(0.0)  # only tx 0 is released
    assert clf.estimated_fraud_coverage(3) == (0.0, 1)
    clf.advance(5.0)
    assert clf.estimated_fraud_coverage(3) == (2 / 3, 3)
    assert clf.estimated_fraud_coverage(5) == (0.8, 5)
    with pytest.raises(ValueError, match="window"):
        clf.estimated_fraud_coverage(6)


def test_controller_direction_and_clip() -> None:
    c = BudgetController(0.1, lr=0.5, t_low_init=0.4)
    c.update(0.3)  # over budget -> narrow (raise t_low)
    assert c.t_low == pytest.approx(0.5)
    c.update(0.0)  # under budget -> widen
    assert c.t_low == pytest.approx(0.45)
    assert c.effective_t_low(0.3) == 0.3  # clipped for decisions only
    assert c.t_low == pytest.approx(0.45)
    assert c.n_updates == 2


def test_validation() -> None:
    with pytest.raises(ValueError, match="review_budget"):
        OnlineConformalClassifier(0.1, 0.1, lr=0.01, mode="budget")
    with pytest.raises(ValueError, match="only to budget mode"):
        OnlineConformalClassifier(0.1, 0.1, lr=0.01, review_budget=0.01)
    with pytest.raises(ValueError, match="mode"):
        OnlineConformalClassifier(0.1, 0.1, lr=0.01, mode="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="review_budget"):
        BudgetController(0.0, lr=0.1, t_low_init=0.5)
    with pytest.raises(ValueError, match="lr"):
        BudgetController(0.1, lr=0.0, t_low_init=0.5)
