import math

import numpy as np
import pytest

from delaycp.metrics import (
    ActionRates,
    Cost,
    action_rates,
    class_coverage,
    cost,
    coverage_measured_at,
    fraud_auto_approved_rate,
    marginal_coverage,
    rolling_coverage,
    summary,
    worst_window_coverage,
)
from delaycp.mondrian import actions_from_thresholds, sets_from_thresholds
from delaycp.stream import CoverageEstimate
from delaycp.types import Action, Thresholds

A, R, D = Action.APPROVE, Action.REVIEW, Action.DECLINE

# idx:      0       1      2       3       4       5        6       7
Y = np.array([0, 0, 1, 0, 1, 1, 0, 0])
SETS = np.array(
    [
        [True, False],  # APPROVE, legit covered
        [True, True],  # REVIEW, covered
        [True, False],  # APPROVE, fraud missed
        [False, True],  # DECLINE, legit missed
        [False, True],  # DECLINE, fraud covered
        [False, False],  # REVIEW (empty), fraud missed
        [True, False],  # APPROVE, legit covered
        [False, False],  # REVIEW (empty), legit missed
    ]
)
ACTIONS = np.array([A, R, A, D, D, R, A, R])
# covered: [1, 1, 0, 0, 1, 0, 1, 0]


def test_class_and_marginal_coverage() -> None:
    cov = class_coverage(Y, SETS)
    assert cov[0] == CoverageEstimate(pytest.approx(3 / 5), 5)
    assert cov[1] == CoverageEstimate(pytest.approx(1 / 3), 3)
    assert marginal_coverage(Y, SETS) == pytest.approx(0.5)


def test_class_coverage_missing_class_is_nan() -> None:
    cov = class_coverage([0, 0], [[True, False], [False, True]])
    assert cov[0] == (0.5, 2)
    assert math.isnan(cov[1].coverage) and cov[1].n_labels == 0
    assert math.isnan(marginal_coverage([], np.empty((0, 2), dtype=bool)))


def test_rolling_coverage() -> None:
    np.testing.assert_allclose(
        rolling_coverage(Y, SETS, 3), [2 / 3, 1 / 3, 1 / 3, 1 / 3, 2 / 3, 1 / 3]
    )
    np.testing.assert_allclose(rolling_coverage(Y, SETS, 2, cls=1), [0.5, 0.5])
    np.testing.assert_allclose(rolling_coverage(Y, SETS, 2, cls=0), [1.0, 0.5, 0.5, 0.5])
    assert rolling_coverage(Y, SETS, 4, cls=1).size == 0
    with pytest.raises(ValueError):
        rolling_coverage(Y, SETS, 0)
    with pytest.raises(ValueError):
        rolling_coverage(Y, SETS, 2, cls=2)


def test_worst_window_coverage() -> None:
    assert worst_window_coverage(Y, SETS, 2) == pytest.approx(0.5)  # cls=1 by default
    assert worst_window_coverage(Y, SETS, 3, cls=None) == pytest.approx(1 / 3)
    assert worst_window_coverage(Y, SETS, 2, cls=0) == pytest.approx(0.5)
    assert math.isnan(worst_window_coverage(Y, SETS, 4))


def test_action_rates() -> None:
    assert action_rates(ACTIONS) == ActionRates(
        3 / 8, 3 / 8, 2 / 8, pytest.approx(math.nan, nan_ok=True)
    )
    assert action_rates(ACTIONS, SETS) == ActionRates(3 / 8, 3 / 8, 2 / 8, 2 / 8)
    with pytest.raises(ValueError):
        action_rates(ACTIONS, SETS[:-1])


def test_fraud_auto_approved_rate() -> None:
    # frauds got APPROVE, DECLINE, REVIEW(empty): 1/3 approved, though 2/3 miscovered
    assert fraud_auto_approved_rate(Y, ACTIONS) == pytest.approx(1 / 3)
    assert math.isnan(fraud_auto_approved_rate([0, 0], [A, D]))


def test_cost() -> None:
    # 1 approved fraud (idx 2), 1 declined legit (idx 3), 3 reviews
    c = cost(Y, ACTIONS, c_missed_fraud=100, c_false_decline=10, c_review=1)
    assert c == Cost(113.0, pytest.approx(113 / 8))
    assert math.isnan(cost([], [], 1, 1, 1).per_transaction)


def test_coverage_measured_at() -> None:
    label_times = [1, 2, 3, 4, 5, 6, np.inf, np.nan]
    assert coverage_measured_at(4.5, Y, SETS, label_times) == (0.5, 4)
    assert coverage_measured_at(4.5, Y, SETS, label_times, cls=1) == (0.0, 1)
    # boundary: a label arriving exactly at `now` counts
    assert coverage_measured_at(5, Y, SETS, label_times, cls=1) == (0.5, 2)
    # later labels never arrive, so the observed legit coverage overstates the truth
    observed = coverage_measured_at(100, Y, SETS, label_times, cls=0)
    assert observed == (pytest.approx(2 / 3), 3)
    assert class_coverage(Y, SETS)[0].coverage == pytest.approx(3 / 5)
    nothing = coverage_measured_at(0, Y, SETS, label_times)
    assert math.isnan(nothing.coverage) and nothing.n_labels == 0


def test_summary() -> None:
    rep = summary(Y, SETS, ACTIONS)
    assert rep["n"] == 8
    assert rep["class_coverage"][1] == (pytest.approx(1 / 3), 3)
    assert rep["marginal_coverage"] == pytest.approx(0.5)
    assert rep["action_rates"].empty == pytest.approx(0.25)
    assert rep["fraud_auto_approved_rate"] == pytest.approx(1 / 3)
    assert "cost" not in rep and "worst_window_fraud_coverage" not in rep

    full = summary(
        Y, SETS, ACTIONS, fraud_window=2, c_missed_fraud=100, c_false_decline=10, c_review=1
    )
    assert full["worst_window_fraud_coverage"] == pytest.approx(0.5)
    assert full["cost"].total == 113.0
    with pytest.raises(ValueError, match="all of"):
        summary(Y, SETS, ACTIONS, c_review=1)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        marginal_coverage([0, 2], SETS[:2])
    with pytest.raises(ValueError, match="shape"):
        class_coverage(Y, SETS[:, :1])
    with pytest.raises(ValueError):
        fraud_auto_approved_rate(Y, ACTIONS[:-1])
    with pytest.raises(ValueError):
        coverage_measured_at(1.0, Y, SETS, [1.0])


def test_consistent_with_mondrian_mapping() -> None:
    p = np.array([0.0, 0.3, 0.45, 0.6, 1.0])
    th = Thresholds(q_0=0.3, q_1=0.4)  # t_high=0.3, t_low=0.6 -> 0.45 has an empty set
    sets = sets_from_thresholds(p, th)
    actions = actions_from_thresholds(p, th)
    rates = action_rates(actions, sets)
    assert rates == ActionRates(0.4, 0.2, 0.4, 0.2)
    assert rates.empty <= rates.review
