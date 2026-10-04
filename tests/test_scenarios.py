"""Integration tests: check delaycp's claims on simulated streams.

The scenarios live in :mod:`benchmarks.scenarios`, shared with
``benchmarks/scenario_report.py``. Every run is seeded, so these tests are
deterministic; the tolerances below are nevertheless chosen so that they
would hold for almost any seed, not tuned to the ones used.

How tolerances are chosen
-------------------------
- **Long-run coverage of the online method** is checked with no statistical
  slack, against the exact identity ``miscoverage - alpha = (q_T - q_0) /
  (lr * T)`` and the a-priori bound :func:`benchmarks.scenarios.long_run_bound`
  derived from it (see that module's docstring). Both hold for every
  realisation, adversarial ones included.
- **Static (split conformal) coverage and windowed coverage** are binomial
  proportions. With ``n`` test examples their standard error is
  ``sqrt(a (1 - a) / n)``. A split-conformal threshold computed from ``m``
  calibration examples adds a coverage error with variance of about
  ``a (1 - a) / (m + 2)`` (the coverage is Beta distributed). Tolerances are
  ``Z = 4`` standard errors: a two-sided miss probability of about 6e-5 per
  check.
- **Rolling-coverage error** is compared with its value for independent
  Bernoulli outcomes, ``E|mean - p| ~ sqrt(2 / pi) * sqrt(p (1 - p) / w)``
  for windows of ``w`` examples. A stable tracker does no worse, because
  its errors are negatively autocorrelated.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest

from benchmarks import scenarios as sc
from delaycp.stream import OnlineConformalClassifier

pytestmark = pytest.mark.slow

Z = 4.0
EXACT = 1e-9  # the long-run identity holds up to float rounding
TAUS = (1, 10, 100, 1000)
MATURITIES = (14.0, 30.0, 60.0, 90.0, 180.0)
DRIFTS: tuple[sc.DriftKind, ...] = ("abrupt_prevalence_shift", "score_drift")


def _se(a: float, *ns: float) -> float:
    """Standard error of a coverage error from several independent sources of sizes ``ns``."""
    return math.sqrt(sum(a * (1 - a) / n for n in ns))


def _binomial_mae(a: float, w: int) -> float:
    """``E|mean - p|`` for ``w`` independent Bernoulli(``a``) outcomes (normal approx.)."""
    return math.sqrt(2 / math.pi) * math.sqrt(a * (1 - a) / w)


# Each scenario is simulated once per session and shared by the tests below.
stationary = cache(sc.scenario_stationary)
fixed_delay = cache(sc.scenario_fixed_delay)
maturity = cache(sc.scenario_maturity)
drift = cache(sc.scenario_drift)
tiny_fraud = cache(sc.scenario_tiny_fraud)
budget = cache(sc.scenario_budget)
adversarial = cache(sc.scenario_adversarial)


# --- 1. stationary, no delay -------------------------------------------------


@pytest.mark.parametrize("cls", [0, 1])
def test_stationary_online_coverage_on_target(cls: int) -> None:
    r = stationary()
    name = "fraud" if cls else "legit"
    assert abs(r[f"identity_gap_{name}"]) < EXACT
    assert abs(r[f"online_{name}_cov"] - sc.TARGET[cls]) <= r[f"bound_{name}"]


def test_stationary_split_conformal_valid_and_comparable() -> None:
    r = stationary()
    a = sc.ALPHA_FRAUD
    se = _se(a, r["n_cal_fraud"] + 2, r["n_fraud"])
    # split conformal is valid on exchangeable data ...
    assert r["split_fraud_cov"] >= sc.TARGET[1] - Z * se
    # ... and the online method matches it within sampling error
    assert abs(r["online_fraud_cov"] - r["split_fraud_cov"]) <= Z * se
    n_legit = 200_000 - r["n_fraud"]
    se_l = _se(sc.ALPHA_LEGIT, sc.N_CAL - r["n_cal_fraud"] + 2, n_legit)
    assert abs(r["online_legit_cov"] - r["split_legit_cov"]) <= Z * se_l


# --- 2. fixed delay ----------------------------------------------------------


@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("cls", [0, 1])
def test_fixed_delay_long_run_coverage(tau: int, cls: int) -> None:
    r = fixed_delay(tau)
    name = "fraud" if cls else "legit"
    assert abs(r[f"identity_gap_{name}"]) < EXACT
    assert abs(r[f"{name}_cov"] - sc.TARGET[cls]) <= r[f"bound_{name}"]


@pytest.mark.parametrize("tau", TAUS)
def test_fixed_delay_fraud_rolling_error_at_noise_floor(tau: int) -> None:
    # Fraud tracker: k * D = lr * f * D ~ 0.01 * 0.83 * 37 < 1/e even at tau = 1000.
    r = fixed_delay(tau)
    assert r["fraud_rolling_mae"] <= _binomial_mae(sc.ALPHA_FRAUD, sc.FRAUD_WINDOW)


@pytest.mark.parametrize(
    "tau",
    [
        1,
        10,
        100,
        pytest.param(
            1000,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "both trackers share lr; ~1000 legit labels in flight puts the "
                    "legit tracker at k * D ~ 8 > pi/2, so it oscillates (see "
                    "benchmarks/scenarios.py). Long-run coverage still holds."
                ),
            ),
        ),
    ],
)
def test_fixed_delay_legit_rolling_error_at_noise_floor(tau: int) -> None:
    r = fixed_delay(tau)
    assert r["legit_rolling_mae"] <= _binomial_mae(sc.ALPHA_LEGIT, sc.LEGIT_WINDOW)


# --- 3. chargeback delays and maturity window --------------------------------


@pytest.mark.parametrize("m", MATURITIES)
def test_maturity_fraud_coverage_unbiased(m: float) -> None:
    # Late frauds are dropped from the fraud class independently of their score.
    r = maturity(m)
    se = _se(sc.ALPHA_FRAUD, r["n_cal_fraud_seen"] + 2, r["n_fraud"])
    assert abs(r["fraud_cov"] - sc.TARGET[1]) <= Z * se


@pytest.mark.parametrize("m", MATURITIES)
def test_maturity_legit_bias_matches_prediction(m: float) -> None:
    r = maturity(m)
    se = _se(sc.ALPHA_LEGIT, sc.N_CAL * (1 - sc.FRAUD_RATE), r["n_legit"])
    assert abs(r["legit_cov"] - r["predicted_legit_cov"]) <= Z * se
    # undiscovered fraud makes measured legit coverage understate the truth
    assert r["legit_cov"] > r["apparent_legit_cov"]


def test_maturity_bias_shrinks_with_window() -> None:
    rows = [maturity(m) for m in MATURITIES]
    contamination = [r["contamination"] for r in rows]
    assert contamination == sorted(contamination, reverse=True)
    # a short window gives a detectable overshoot of the legit target
    short = rows[0]
    se = _se(sc.ALPHA_LEGIT, sc.N_CAL * (1 - sc.FRAUD_RATE), short["n_legit"])
    assert short["legit_cov"] - sc.TARGET[0] > Z * se


# --- 4. prevalence shift and score drift --------------------------------------


@pytest.mark.parametrize("kind", DRIFTS)
def test_drift_online_recovers_fraud_coverage(kind: sc.DriftKind) -> None:
    r = drift(kind)
    se = _se(sc.ALPHA_FRAUD, r["n_fraud_post"])
    assert abs(r["online_fraud_cov_post"] - sc.TARGET[1]) <= Z * se
    assert abs(r["identity_gap_fraud"]) < EXACT
    assert abs(r["online_fraud_cov"] - sc.TARGET[1]) <= r["bound_fraud"]
    assert abs(r["online_legit_cov"] - sc.TARGET[0]) <= r["bound_legit"]


def test_score_drift_breaks_static_split_conformal() -> None:
    r = drift("score_drift")
    se = _se(sc.ALPHA_FRAUD, r["n_fraud_post"])
    assert r["static_fraud_cov_post"] < sc.TARGET[1] - Z * se


def test_prevalence_shift_leaves_static_per_class_coverage_unchanged() -> None:
    # Mondrian invariance: the class-conditional score laws do not change.
    r = drift("abrupt_prevalence_shift")
    se = _se(sc.ALPHA_FRAUD, r["n_fraud_pre"], r["n_fraud_post"])
    assert abs(r["static_fraud_cov_post"] - r["static_fraud_cov_pre"]) <= Z * se


# --- 5. tiny fraud class -----------------------------------------------------


def test_tiny_fraud_calibration_warns() -> None:
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.beta(1, 12, 1_000), rng.beta(5, 2, 20)])
    y = np.r_[np.zeros(1_000, dtype=int), np.ones(20, dtype=int)]
    clf = OnlineConformalClassifier(sc.ALPHA_FRAUD, sc.ALPHA_LEGIT, lr=sc.LR)
    with pytest.warns(UserWarning, match="class 1 has only 20 calibration examples"):
        clf.fit_calibration(p, y)


@pytest.mark.parametrize("n_cal_fraud", [0, 10, 30])
def test_tiny_fraud_behaviour_is_sane(n_cal_fraud: int) -> None:
    r = tiny_fraud(n_cal_fraud)
    assert r["warned"] == 1.0
    assert r["finite_thresholds"] == 1.0
    assert abs(r["identity_gap_fraud"]) < EXACT
    assert abs(r["fraud_cov"] - sc.TARGET[1]) <= r["bound_fraud"]
    assert abs(r["legit_cov"] - sc.TARGET[0]) <= r["bound_legit"]
    # the fraud warm start cannot inflate the review queue much: stationary is ~5%
    assert r["review"] < 0.1


# --- 6. budget mode under drift ----------------------------------------------


@pytest.mark.parametrize("kind", DRIFTS)
def test_budget_review_rate_on_target(kind: sc.DriftKind) -> None:
    r = budget(kind)
    # deterministic, from the controller's telescoping identity
    assert abs(r["review"] - sc.REVIEW_BUDGET) <= r["budget_bound"]
    # every 10,000-transaction window after the change, within sampling error
    se = _se(sc.REVIEW_BUDGET, sc.BUDGET_WINDOW * sc.BUDGET_BATCH)
    assert r["max_rolling_dev_post"] <= Z * se


def test_budget_fraud_coverage_estimate_tracks_truth_under_drift() -> None:
    # Fraud coverage is an output in budget mode; under score drift it drops
    # below target, and the label-based estimate must reveal that.
    r = budget("score_drift")
    se = _se(r["fraud_cov_post"], 1_000)
    assert abs(r["estimated_fraud_cov_last_1000"] - r["fraud_cov_post"]) <= Z * se
    assert r["fraud_cov_post"] < sc.TARGET[1] - Z * se


# --- 7. adversarial ----------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.05, 0.5])
def test_adversarial_long_run_guarantee(fraction: float) -> None:
    r = adversarial(fraction)
    assert abs(r["identity_gap_fraud"]) < EXACT
    assert abs(r["fraud_cov"] - sc.TARGET[1]) <= r["bound_fraud"]


def test_adversarial_cost_of_guarantee() -> None:
    # Below alpha the adversary can be absorbed with a sane band; above it
    # t_low is driven to about 0 and most traffic goes to review.
    assert adversarial(0.05)["review"] < 0.1
    assert adversarial(0.5)["review"] > 0.5
