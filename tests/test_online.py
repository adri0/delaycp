from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from delaycp.online import PIDQuantileTracker


def _run(tr: PIDQuantileTracker, scores: np.ndarray) -> np.ndarray:
    errs = np.zeros(len(scores))
    for i, s in enumerate(scores):
        q = tr.threshold
        errs[i] = float(s > q)
        tr.update(float(s), q)
    return errs


def test_iid_uniform_converges_to_alpha() -> None:
    rng = np.random.default_rng(0)
    errs = _run(PIDQuantileTracker(0.1, lr=0.05), rng.uniform(size=20_000))
    assert abs(errs.mean() - 0.1) < 0.01


def test_iid_uniform_with_integrator() -> None:
    rng = np.random.default_rng(1)
    tr = PIDQuantileTracker(0.1, lr=0.05, ki=0.5, c_sat=2.0)
    errs = _run(tr, rng.uniform(size=20_000))
    assert abs(errs.mean() - 0.1) < 0.01


@settings(max_examples=100, deadline=None)
@given(
    scores=st.lists(st.floats(0.0, 1.0), min_size=1, max_size=300),
    alpha=st.floats(0.01, 0.99),
    lr=st.floats(0.001, 1.0),
    q_init=st.floats(-1.0, 2.0),
)
def test_p_only_long_run_bound(
    scores: list[float], alpha: float, lr: float, q_init: float
) -> None:
    # q_{T+1} = q_1 + lr * sum(err - alpha), and q stays in [-lr*alpha, b + lr*(1-alpha)]
    # (b = 1 is the score bound), so |avg err - alpha| <= max(...) / (lr * T).
    # This bound is derived here rather than copied from the paper; it allows
    # any q_init (not only q_init in [0, b]) and has the paper's O(1/T) rate.
    tr = PIDQuantileTracker(alpha, lr, q_init=q_init)
    errs = _run(tr, np.asarray(scores))
    n = len(scores)
    bound = max(1.0 + lr * (1 - alpha) - q_init, q_init + lr * alpha) / (lr * n)
    assert abs(errs.mean() - alpha) <= bound + 1e-9


def test_recovers_after_abrupt_shift() -> None:
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.uniform(0, 0.3, 3000), rng.uniform(0.5, 1.0, 3000)])
    errs = _run(PIDQuantileTracker(0.1, lr=0.05), scores)
    assert errs[3000:3010].mean() > 0.9  # shift is felt immediately
    assert abs(errs[-2000:].mean() - 0.1) < 0.03


def _reference_quantile(scores: np.ndarray, alpha: float, lr: float) -> np.ndarray:
    """Minimal port of ``quantile`` in conformal-time-series core/methods.py.

    ``proportional_lr=False``, ``KI=0``, ``ahead=1`` (no delay). Returns the q
    in effect at each step.

    The integrator is left out on purpose: the reference sums errors over
    ``[:t]`` (excluding the newest), one step behind the paper, which
    ``PIDQuantileTracker`` follows, so the two would not match exactly. The
    integrator is tested against the closed-form formula instead.
    """
    qs = np.zeros(len(scores))
    qts = np.zeros(len(scores))
    covereds = np.zeros(len(scores))
    for t in range(len(scores)):
        covereds[t] = qs[t] >= scores[t]
        grad = alpha if covereds[t] else -(1 - alpha)
        if t < len(scores) - 1:
            qts[t + 1] = qts[t] - lr * grad
            qs[t + 1] = qts[t + 1]
    return qs


def test_matches_reference_without_delay() -> None:
    rng = np.random.default_rng(3)
    scores = rng.beta(2, 5, size=500)
    alpha, lr = 0.1, 0.03
    expected = _reference_quantile(scores, alpha, lr)
    tr = PIDQuantileTracker(alpha, lr, q_init=0.0)  # reference starts at q = 0
    got = np.zeros(len(scores))
    for i, s in enumerate(scores):
        got[i] = tr.threshold
        tr.update(float(s), tr.threshold)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_delayed_update_uses_q_used() -> None:
    tr = PIDQuantileTracker(0.1, lr=0.1, q_init=0.5)
    tr.update(0.4, q_used=0.3)  # error against q_used, although 0.4 <= current q
    assert tr.threshold == pytest.approx(0.5 + 0.1 * 0.9)
    assert tr.n_updates == 1


def test_integrator_matches_formula_and_saturates() -> None:
    alpha, ki, c = 0.1, 0.5, 2.0
    tr = PIDQuantileTracker(alpha, lr=0.01, ki=ki, c_sat=c, q_init=0.5)
    for _ in range(10):
        tr.update(2.0, q_used=0.0)  # always an error
    t, x = 10, 10 * (1 - alpha)
    expected_q_p = 0.5 + 0.01 * x
    arg = x * math.log(t) / (t * c)
    assert arg < math.pi / 2
    assert tr.threshold == pytest.approx(expected_q_p + ki * math.tan(arg))
    tr2 = PIDQuantileTracker(alpha, lr=0.01, ki=ki, c_sat=0.1)
    for _ in range(10):
        tr2.update(2.0, q_used=0.0)
    assert tr2.threshold == math.inf


def test_init_from_scores_finite_sample_and_unclipped() -> None:
    tr = PIDQuantileTracker(0.1, lr=0.1)
    tr.init_from_scores(np.arange(1, 10) / 10)  # n=9, rank=ceil(10*0.9)=9
    assert tr.threshold == pytest.approx(0.9)
    tr.init_from_scores([0.1, 0.2, 0.3, 0.4])  # rank 5 > n -> max score
    assert tr.threshold == pytest.approx(0.4)
    tr.update(0.0, q_used=1.0)
    assert tr.threshold < 0.4
    for _ in range(100):
        tr.update(0.0, q_used=1.0)
    assert tr.threshold < 0.0  # not clipped


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        PIDQuantileTracker(0.0, 0.1)
    with pytest.raises(ValueError):
        PIDQuantileTracker(0.1, 0.0)
    with pytest.raises(ValueError):
        PIDQuantileTracker(0.1, 0.1).init_from_scores([])
