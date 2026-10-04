"""Validation scenarios for delaycp, shared by the test suite and the report.

Each ``scenario_*`` function simulates a stream with :mod:`delaycp.simulate`,
runs :class:`~delaycp.stream.OnlineConformalClassifier` on it (and, where it
makes sense, a static split-conformal baseline with the calibration
thresholds frozen) and returns a flat ``dict`` of numbers.
``tests/test_scenarios.py`` asserts on them and
``benchmarks/scenario_report.py`` tabulates them. Everything is deterministic
given ``seed``.

The long-run guarantee
----------------------
With ``ki = 0`` a class tracker obeys ``q_T - q_0 = lr * sum_i (err_i - alpha)``
over its released updates, each error judged against the threshold in effect
at prediction time. Once every label has been released, the class's long-run
miscoverage over its ``T`` predictions therefore satisfies, *exactly*::

    miscoverage - alpha = (q_T - q_0) / (lr * T).

``q`` can leave ``[0, 1]`` only by a bounded amount: above 1 every new
prediction is covered, so only the at most ``D`` updates already in flight
(predicted, label not yet released) can push it further, and symmetrically
below 0. Hence ``|q_T - q_0| <= max(q_0, 1) - min(q_0, 0) + lr * (D + 1)``,
which gives the a-priori bound :func:`long_run_bound`. It holds for *any*
label sequence, delay pattern or adversary, so tests can assert it without a
statistical tolerance.

Delay and the shared step size
------------------------------
Both class trackers share one ``lr``. A quantile tracker whose updates arrive
``D`` steps late behaves like ``x' = -k x(t - D)`` with ``k = lr * f`` (``f``
the score density at the threshold): it settles smoothly for ``k D < 1/e``
and oscillates, with bounded amplitude, for ``k D > pi/2``. Legit labels
outnumber fraud labels about 49 to 1 here, so the legit tracker has about 49
times as many updates in flight and reaches the unstable regime first. At
``lr = 0.01``, with ``f ~ 0.8`` at ``t_high``, that happens once more than
roughly 200 legit labels are pending. Scenario 2 measures this. Scenarios 3
and 6 do not exercise the fraud tracker's adaptivity, so they use
``LR_SLOW``, a step size at which the legit tracker is stable under their
delays.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from delaycp import metrics
from delaycp.mondrian import actions_from_thresholds, sets_from_thresholds
from delaycp.simulate import ReplayResult, Stream, inject_delays, make_stream, replay
from delaycp.stream import OnlineConformalClassifier
from delaycp.types import Action, Thresholds

ALPHA_FRAUD = 0.1
ALPHA_LEGIT = 0.05
ALPHA = {0: ALPHA_LEGIT, 1: ALPHA_FRAUD}
TARGET = {0: 1.0 - ALPHA_LEGIT, 1: 1.0 - ALPHA_FRAUD}
LR = 0.01
LR_SLOW = 5e-6  # k * D < 1/e for the legit tracker with up to ~90k labels in flight
FRAUD_RATE = 0.02
N_CAL = 20_000  # ~400 calibration frauds
FRAUD_WINDOW = 200  # frauds per rolling-coverage window
LEGIT_WINDOW = 2_000  # legits per rolling-coverage window
DELAY_MEDIAN = 3.0  # days, lognormal label delay on a 100-day horizon
DELAY_SIGMA = 0.8

Cal = tuple[NDArray[np.float64], NDArray[np.int_]]


# --- running ---------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One online run plus what is needed to check it.

    ``th_init`` are the thresholds after calibration (also the static
    split-conformal thresholds) and ``th_final`` those after every label has
    been released. ``max_in_flight[c]`` is the largest number of class-``c``
    predictions awaiting their label at any time. ``static_sets`` and
    ``static_actions`` come from split conformal (``th_init`` frozen), or are
    ``None`` when the scores depend on the online thresholds (adversarial).
    """

    res: ReplayResult
    clf: OnlineConformalClassifier
    lr: float
    th_init: Thresholds
    th_final: Thresholds
    max_in_flight: dict[int, int]
    static_sets: NDArray[np.bool_] | None
    static_actions: NDArray[np.int_] | None


def make_clf(
    *,
    lr: float = LR,
    maturity: float | None = None,
    mode: Literal["coverage", "budget"] = "coverage",
    review_budget: float | None = None,
    budget_lr: float | None = None,
) -> OnlineConformalClassifier:
    """Classifier with the suite's coverage targets."""
    return OnlineConformalClassifier(
        ALPHA_FRAUD,
        ALPHA_LEGIT,
        lr=lr,
        maturity=maturity,
        mode=mode,
        review_budget=review_budget,
        budget_lr=budget_lr,
    )


def _max_in_flight(
    t: NDArray[np.float64],
    lt: NDArray[np.float64],
    y: NDArray[np.int_],
    batch_size: int,
) -> dict[int, int]:
    """Peak number of same-class predictions whose label is not yet released.

    :func:`~delaycp.simulate.replay` releases labels only at batch starts, so
    a label is released at the first batch start at or after its arrival.
    """
    starts = t[::batch_size]
    idx = np.searchsorted(starts, lt, side="left")
    release = np.where(idx < starts.size, starts[np.minimum(idx, starts.size - 1)], np.inf)
    out: dict[int, int] = {}
    for c in (0, 1):
        m = y == c
        k = int(m.sum())
        times = np.concatenate([release[m], t[m]])
        delta = np.concatenate([-np.ones(k, dtype=int), np.ones(k, dtype=int)])
        # a release at time r happens before the predictions at r: ends sort first
        order = np.lexsort((delta, times))
        out[c] = int(np.max(np.cumsum(delta[order]), initial=0))
    return out


def run_online(
    clf: OnlineConformalClassifier,
    stream: Stream,
    label_times: NDArray[np.float64],
    cal: Cal,
    *,
    lr: float = LR,
    timestamps: NDArray[np.float64] | None = None,
    batch_size: int = 1,
) -> Run:
    """Calibrate the fresh ``clf``, replay ``stream`` and release every remaining label.

    ``lr`` must be the step size ``clf`` was built with.
    """
    t = stream.timestamps if timestamps is None else timestamps
    clf.fit_calibration(*cal)
    th_init = clf.thresholds
    p = stream.p_fraud
    res = replay(clf, t, p, stream.y, label_times, batch_size=batch_size)
    clf.advance(math.inf)
    static = p if isinstance(p, np.ndarray) else None
    return Run(
        res=res,
        clf=clf,
        lr=lr,
        th_init=th_init,
        th_final=clf.thresholds,
        max_in_flight=_max_in_flight(t, label_times, stream.y, batch_size),
        static_sets=None if static is None else sets_from_thresholds(static, th_init),
        static_actions=None if static is None else actions_from_thresholds(static, th_init),
    )


def calibration(n: int, rng: np.random.Generator) -> Cal:
    """Labelled calibration data from the stationary score model."""
    s = make_stream(n, FRAUD_RATE, rng)
    assert isinstance(s.p_fraud, np.ndarray)
    return s.p_fraud, s.y


def _lognormal_delays(s: Stream, rng: np.random.Generator) -> NDArray[np.float64]:
    return inject_delays(
        s.timestamps, s.y, "lognormal", rng, median=DELAY_MEDIAN, sigma=DELAY_SIGMA
    )


# --- checks ----------------------------------------------------------------


def _q(th: Thresholds, cls: int) -> float:
    return th.q_1 if cls == 1 else th.q_0


def coverage(
    run: Run,
    cls: int,
    sets: NDArray[np.bool_] | None = None,
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Coverage of class ``cls`` (online sets unless ``sets`` is given), optionally on ``mask``."""
    s = run.res.sets if sets is None else sets
    y = run.res.y
    if mask is not None:
        y, s = y[mask], s[mask]
    return metrics.class_coverage(y, s)[cls].coverage


def long_run_identity_gap(run: Run, cls: int) -> float:
    """``(miscoverage - alpha) - (q_T - q_0) / (lr T)``: zero up to rounding.

    Valid in coverage mode when every label was released with its true label
    (no maturity imputation).
    """
    n_cls = int(np.sum(run.res.y == cls))
    drift = (_q(run.th_final, cls) - _q(run.th_init, cls)) / (run.lr * n_cls)
    return (1.0 - coverage(run, cls) - ALPHA[cls]) - drift


def long_run_bound(run: Run, cls: int) -> float:
    """A-priori bound on ``|miscoverage - alpha|`` for class ``cls``; see the module doc."""
    q0 = _q(run.th_init, cls)
    n_cls = int(np.sum(run.res.y == cls))
    span = max(q0, 1.0) - min(q0, 0.0) + run.lr * (run.max_in_flight[cls] + 1)
    return span / (run.lr * n_cls)


def rolling_error(run: Run, cls: int) -> tuple[float, float]:
    """Mean absolute rolling-coverage error and the worst (lowest) window, for class ``cls``.

    Windows hold ``FRAUD_WINDOW`` frauds or ``LEGIT_WINDOW`` legits.
    """
    window = FRAUD_WINDOW if cls == 1 else LEGIT_WINDOW
    rc = metrics.rolling_coverage(run.res.y, run.res.sets, window, cls=cls)
    return float(np.mean(np.abs(rc - TARGET[cls]))), float(np.min(rc))


def _review(actions: NDArray[np.int_] | None) -> float:
    assert actions is not None
    return metrics.action_rates(actions).review


# --- scenarios -------------------------------------------------------------


def scenario_stationary(n: int = 200_000, seed: int = 1) -> dict[str, float]:
    """1. Stationary stream; each label is released at the next prediction."""
    rng = np.random.default_rng(seed)
    cal = calibration(N_CAL, rng)
    s = make_stream(n, FRAUD_RATE, rng)
    run = run_online(make_clf(), s, s.timestamps.copy(), cal)
    return {
        "n_fraud": float(np.sum(s.y == 1)),
        "n_cal_fraud": float(np.sum(cal[1] == 1)),
        "online_fraud_cov": coverage(run, 1),
        "online_legit_cov": coverage(run, 0),
        "split_fraud_cov": coverage(run, 1, run.static_sets),
        "split_legit_cov": coverage(run, 0, run.static_sets),
        "online_review": _review(run.res.actions),
        "split_review": _review(run.static_actions),
        "identity_gap_fraud": long_run_identity_gap(run, 1),
        "identity_gap_legit": long_run_identity_gap(run, 0),
        "bound_fraud": long_run_bound(run, 1),
        "bound_legit": long_run_bound(run, 0),
    }


def scenario_fixed_delay(tau: int, n: int = 200_000, seed: int = 2) -> dict[str, float]:
    """2. Stationary stream, one transaction per step, every label ``tau`` steps late.

    The same stream and calibration set are used for every ``tau``.
    """
    rng = np.random.default_rng(seed)
    cal = calibration(N_CAL, rng)
    s = make_stream(n, FRAUD_RATE, rng)
    t = np.arange(n, dtype=float)
    lt = inject_delays(t, s.y, "fixed", rng, delay=float(tau))
    run = run_online(make_clf(), s, lt, cal, timestamps=t)
    mae_f, worst_f = rolling_error(run, 1)
    mae_l, worst_l = rolling_error(run, 0)
    return {
        "tau": float(tau),
        "fraud_cov": coverage(run, 1),
        "legit_cov": coverage(run, 0),
        "fraud_rolling_mae": mae_f,
        "fraud_worst_window": worst_f,
        "legit_rolling_mae": mae_l,
        "legit_worst_window": worst_l,
        "in_flight_fraud": float(run.max_in_flight[1]),
        "in_flight_legit": float(run.max_in_flight[0]),
        "identity_gap_fraud": long_run_identity_gap(run, 1),
        "identity_gap_legit": long_run_identity_gap(run, 0),
        "bound_fraud": long_run_bound(run, 1),
        "bound_legit": long_run_bound(run, 0),
    }


CHARGEBACK_MEDIAN = 30.0  # days
CHARGEBACK_SIGMA = 0.8
P_NEVER = 0.05  # frauds never reported
MATURITY_HORIZON = 365.0  # days


def _matured_labels(
    t: NDArray[np.float64], y: NDArray[np.int_], maturity: float, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Chargeback label times, and the labels as known at maturity (late fraud -> 0)."""
    lt = inject_delays(
        t,
        y,
        "lognormal",
        rng,
        median=CHARGEBACK_MEDIAN,
        sigma=CHARGEBACK_SIGMA,
        p_never=P_NEVER,
        legit_delay=maturity,
    )
    y_matured = np.where((y == 1) & (lt - t <= maturity), 1, 0)
    return lt, y_matured


def scenario_maturity(maturity: float, n: int = 100_000, seed: int = 3) -> dict[str, float]:
    """3. Lognormal chargeback delays; legit confirmed after ``maturity`` days.

    Frauds not reported within the window (late, or never reported) are
    imputed as legit, in calibration as well as online, as production would
    see them. The fraud class only loses a score-independent subsample, so
    fraud coverage stays unbiased. The legit class is contaminated: a
    fraction ``c`` of its examples are undiscovered frauds, nearly all
    scored above ``t_high``. Holding miscoverage ``alpha_legit`` on the
    imputed class then gives true legit miscoverage of about
    ``(alpha_legit - c * m_F) / (1 - c)`` with ``m_F = P(p > t_high | fraud)``,
    so legit coverage overshoots its target (``predicted_legit_cov``).

    Uses ``LR_SLOW``: with ``n / horizon * maturity`` legit labels in flight
    the legit tracker is unstable at ``LR`` (see the module doc).
    """
    rng = np.random.default_rng(seed)
    p_cal, y_cal = calibration(N_CAL, rng)
    t_cal = np.sort(rng.uniform(0.0, MATURITY_HORIZON, N_CAL))
    _, y_cal_matured = _matured_labels(t_cal, y_cal, maturity, rng)
    s = make_stream(n, FRAUD_RATE, rng, horizon=MATURITY_HORIZON)
    lt, y_matured = _matured_labels(s.timestamps, s.y, maturity, rng)
    clf = make_clf(lr=LR_SLOW, maturity=maturity)
    run = run_online(clf, s, lt, (p_cal, y_cal_matured), lr=LR_SLOW, batch_size=10)
    y, r = s.y, run.res
    fraud = y == 1
    c = float(np.sum(fraud & (y_matured == 0)) / np.sum(y_matured == 0))
    m_f = float(np.mean(r.p_fraud[fraud] > r.t_high[fraud]))
    apparent = metrics.class_coverage(y_matured, r.sets)
    return {
        "maturity": maturity,
        "unreported_fraud_share": float(np.mean(y_matured[fraud] == 0)),
        "contamination": c,
        "n_legit": float(np.sum(y == 0)),
        "n_fraud": float(np.sum(fraud)),
        "n_cal_fraud_seen": float(np.sum(y_cal_matured == 1)),
        "fraud_cov": coverage(run, 1),
        "legit_cov": coverage(run, 0),
        "apparent_fraud_cov": apparent[1].coverage,
        "apparent_legit_cov": apparent[0].coverage,
        "predicted_legit_cov": 1.0 - (ALPHA_LEGIT - c * m_f) / (1.0 - c),
        "review": _review(r.actions),
    }


DriftKind = Literal["abrupt_prevalence_shift", "score_drift"]
CHANGE_AT = 50.0  # days into the 100-day horizon
RECOVERED_FROM = 60.0  # "after recovery": ~3 median label delays past the change


def scenario_drift(kind: DriftKind, n: int = 200_000, seed: int = 4) -> dict[str, float]:
    """4. Abrupt change at day 50; lognormal label delays (median 3 days).

    Coverage is reported before the change and from day 60 on, after
    recovery. Under a pure prevalence shift the class-conditional score
    distributions do not change, so per-class (Mondrian) coverage is
    unaffected for static and online alike; score drift is what breaks the
    static thresholds.
    """
    rng = np.random.default_rng(seed)
    cal = calibration(N_CAL, rng)
    s = make_stream(n, FRAUD_RATE, rng, kind, change_at=CHANGE_AT / 100.0)
    run = run_online(make_clf(), s, _lognormal_delays(s, rng), cal, batch_size=10)
    t = s.timestamps
    pre, post = t < CHANGE_AT, t >= RECOVERED_FROM
    assert run.static_actions is not None
    return {
        "n_fraud_pre": float(np.sum(s.y[pre] == 1)),
        "n_fraud_post": float(np.sum(s.y[post] == 1)),
        "online_fraud_cov_pre": coverage(run, 1, mask=pre),
        "online_fraud_cov_post": coverage(run, 1, mask=post),
        "static_fraud_cov_pre": coverage(run, 1, run.static_sets, pre),
        "static_fraud_cov_post": coverage(run, 1, run.static_sets, post),
        "online_legit_cov_post": coverage(run, 0, mask=post),
        "static_legit_cov_post": coverage(run, 0, run.static_sets, post),
        "online_review_post": _review(run.res.actions[post]),
        "static_review_post": _review(run.static_actions[post]),
        "online_fraud_cov": coverage(run, 1),
        "online_legit_cov": coverage(run, 0),
        "bound_fraud": long_run_bound(run, 1),
        "bound_legit": long_run_bound(run, 0),
        "identity_gap_fraud": long_run_identity_gap(run, 1),
    }


def scenario_tiny_fraud(n_cal_fraud: int, n: int = 100_000, seed: int = 5) -> dict[str, float]:
    """5. Calibration set with only ``n_cal_fraud`` frauds (and 5,000 legits).

    ``warned`` is 1 if :meth:`fit_calibration` warned about the fraud class.
    """
    rng = np.random.default_rng(seed)
    p_cal, y_cal = calibration(N_CAL, rng)
    keep = np.concatenate(
        [np.flatnonzero(y_cal == 1)[:n_cal_fraud], np.flatnonzero(y_cal == 0)[:5_000]]
    )
    s = make_stream(n, FRAUD_RATE, rng)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = run_online(make_clf(), s, s.timestamps.copy(), (p_cal[keep], y_cal[keep]))
    warned = any("class 1" in str(w.message) for w in caught)
    finite = all(math.isfinite(_q(th, c)) for th in (run.th_init, run.th_final) for c in (0, 1))
    first = np.arange(n) < n // 10
    return {
        "n_cal_fraud": float(n_cal_fraud),
        "warned": float(warned),
        "finite_thresholds": float(finite),
        "init_t_low": run.th_init.t_low,
        "fraud_cov": coverage(run, 1),
        "legit_cov": coverage(run, 0),
        "split_fraud_cov": coverage(run, 1, run.static_sets),
        "fraud_cov_first_10pct": coverage(run, 1, mask=first),
        "review": _review(run.res.actions),
        "bound_fraud": long_run_bound(run, 1),
        "bound_legit": long_run_bound(run, 0),
        "identity_gap_fraud": long_run_identity_gap(run, 1),
    }


REVIEW_BUDGET = 0.03
BUDGET_BATCH = 100
BUDGET_LR = 0.02
BUDGET_WINDOW = 100  # batches (10,000 transactions) per rolling window


def scenario_budget(kind: DriftKind, n: int = 200_000, seed: int = 6) -> dict[str, float]:
    """6. Budget mode under drift: the REVIEW rate should stay at ``REVIEW_BUDGET``.

    With ``B`` equal batches the controller obeys
    ``mean review - budget = (t_T - t_0) / (budget_lr * B)``. Its state stays
    in ``[min(t_0, 0), max(t_0, max t_high)]`` up to one step (below 0 the
    band holds every score ``<= t_high``, above ``t_high`` it is empty), which
    gives ``budget_bound``.

    Uses ``LR_SLOW`` for the legit tracker: budget mode does not update the
    fraud tracker, and at ``LR`` the legit tracker oscillates under these
    delays (see the module doc), which moves ``t_high`` and the band with it.
    The budget controller keeps its own step size ``BUDGET_LR``.
    """
    if n % BUDGET_BATCH:
        raise ValueError("n must be a multiple of BUDGET_BATCH")
    rng = np.random.default_rng(seed)
    cal = calibration(N_CAL, rng)
    s = make_stream(n, FRAUD_RATE, rng, kind, change_at=CHANGE_AT / 100.0)
    clf = make_clf(lr=LR_SLOW, mode="budget", review_budget=REVIEW_BUDGET, budget_lr=BUDGET_LR)
    run = run_online(clf, s, _lognormal_delays(s, rng), cal, lr=LR_SLOW, batch_size=BUDGET_BATCH)
    review = run.res.actions == Action.REVIEW
    batch_rates = review.reshape(-1, BUDGET_BATCH).mean(axis=1)
    csum = np.concatenate([[0.0], np.cumsum(batch_rates)])
    rolling = (csum[BUDGET_WINDOW:] - csum[:-BUDGET_WINDOW]) / BUDGET_WINDOW
    window_start = s.timestamps[::BUDGET_BATCH][: rolling.size]
    t = s.timestamps
    pre, post = t < CHANGE_AT, t >= RECOVERED_FROM
    t0 = run.th_init.t_low
    top = max(t0, float(np.max(run.res.t_high)))
    n_batches = n // BUDGET_BATCH
    return {
        "review": float(np.mean(review)),
        "budget_bound": (top - min(t0, 0.0) + BUDGET_LR) / (BUDGET_LR * n_batches),
        "review_pre": float(np.mean(review[pre])),
        "review_post": float(np.mean(review[post])),
        "max_rolling_dev": float(np.max(np.abs(rolling - REVIEW_BUDGET))),
        "max_rolling_dev_post": float(
            np.max(np.abs(rolling[window_start >= RECOVERED_FROM] - REVIEW_BUDGET))
        ),
        "fraud_cov_pre": coverage(run, 1, mask=pre),
        "fraud_cov_post": coverage(run, 1, mask=post),
        "estimated_fraud_cov_last_1000": run.clf.estimated_fraud_coverage(1_000).coverage,
        "legit_cov": coverage(run, 0),
    }


def scenario_adversarial(fraction: float, n: int = 200_000, seed: int = 7) -> dict[str, float]:
    """7. A ``fraction`` of frauds is placed just below the live ``t_low``.

    The adversary sees the thresholds in force when each batch is scored.
    Labels arrive with lognormal delays (median 3 days).
    """
    rng = np.random.default_rng(seed)
    cal = calibration(N_CAL, rng)
    clf = make_clf()
    s = make_stream(
        n,
        FRAUD_RATE,
        rng,
        "adversarial",
        thresholds_fn=lambda: clf.thresholds,
        adversarial_fraction=fraction,
    )
    run = run_online(clf, s, _lognormal_delays(s, rng), cal, batch_size=10)
    mae_f, worst_f = rolling_error(run, 1)
    return {
        "fraction": fraction,
        "fraud_cov": coverage(run, 1),
        "legit_cov": coverage(run, 0),
        "fraud_rolling_mae": mae_f,
        "fraud_worst_window": worst_f,
        "fraud_auto_approved": metrics.fraud_auto_approved_rate(run.res.y, run.res.actions),
        "review": _review(run.res.actions),
        "mean_t_low_late": float(np.mean(run.res.t_low[n // 2 :])),
        "bound_fraud": long_run_bound(run, 1),
        "bound_legit": long_run_bound(run, 0),
        "identity_gap_fraud": long_run_identity_gap(run, 1),
    }
