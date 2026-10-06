"""Synthetic streams, label-delay injection and replay, for tests and examples.

Typical use::

    rng = np.random.default_rng(0)
    s = make_stream(50_000, 0.01, rng, "score_drift")
    lt = inject_delays(s.timestamps, s.y, "lognormal", rng, median=30.0, sigma=0.8)
    res = replay(clf, s.timestamps, s.p_fraud, s.y, lt, batch_size=100)
    metrics.summary(res.y, res.sets, res.actions)

Timestamps are in days. Everything is deterministic given ``rng``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from delaycp.mondrian import sets_from_thresholds
from delaycp.stream import OnlineConformalClassifier
from delaycp.types import Thresholds

__all__ = [
    "AdversarialScores",
    "DelayDist",
    "ReplayResult",
    "Scenario",
    "Stream",
    "inject_delays",
    "make_stream",
    "replay",
]

DelayDist = Literal["fixed", "lognormal", "mixture"]
Scenario = Literal[
    "stationary", "abrupt_prevalence_shift", "score_drift", "seasonal_spike", "adversarial"
]

# Score model: an imperfect classifier whose class-conditional p_fraud overlap.
_LEGIT_BETA = (1.0, 12.0)  # mean ~0.08
_FRAUD_BETA = (5.0, 2.0)  # mean ~0.71
_DRIFTED_FRAUD_BETA = (2.0, 4.0)  # mean ~0.33: new fraud the model scores low


def _lognormal(
    rng: np.random.Generator, median: float, sigma: float, n: int
) -> NDArray[np.float64]:
    if median <= 0 or sigma < 0:
        raise ValueError("lognormal needs median > 0 and sigma >= 0")
    out: NDArray[np.float64] = median * np.exp(sigma * rng.standard_normal(n))
    return out


def inject_delays(
    timestamps: ArrayLike,
    y: ArrayLike,
    dist: DelayDist,
    rng: np.random.Generator,
    *,
    p_never: float = 0.0,
    legit_delay: float | None = None,
    **params: float,
) -> NDArray[np.float64]:
    """Draw a label arrival time for every transaction.

    Parameters
    ----------
    timestamps : array_like of float
        Prediction times.
    y : array_like of int
        True labels (``1`` = fraud).
    dist : {"fixed", "lognormal", "mixture"}
        Delay distribution, with parameters passed as keywords:

        - ``"fixed"``: ``delay``.
        - ``"lognormal"`` (chargeback-like): ``median`` and ``sigma``, the
          standard deviation of ``log(delay)``.
        - ``"mixture"``: with probability ``p_fast`` a lognormal with median
          ``fast_median``, otherwise one with median ``slow_median`` (the
          late tail); both use ``sigma`` (default ``0.5``).
    rng : numpy.random.Generator
        Source of randomness.
    p_never : float, default 0.0
        Probability that a fraud is never reported (label time ``inf``).
        Legit labels are always eventually known.
    legit_delay : float, optional
        If given, legit labels arrive after exactly this delay (e.g. the
        maturity window after which "no chargeback" confirms legit) instead
        of following ``dist``.

    Returns
    -------
    numpy.ndarray
        ``timestamps + delay``, ``inf`` for never-reported frauds.
    """
    t = np.asarray(timestamps, dtype=float)
    labels = np.asarray(y, dtype=int)
    if t.ndim != 1 or t.shape != labels.shape:
        raise ValueError("timestamps and y must be 1-D and equal length")
    if not 0.0 <= p_never <= 1.0:
        raise ValueError("p_never must be in [0, 1]")
    if legit_delay is not None and legit_delay < 0:
        raise ValueError("legit_delay must be >= 0")
    n = t.size
    kw = dict(params)

    def take(name: str, default: float | None = None) -> float:
        if name in kw:
            return float(kw.pop(name))
        if default is None:
            raise TypeError(f'dist="{dist}" requires parameter {name!r}')
        return default

    if dist == "fixed":
        d = take("delay")
        if d < 0:
            raise ValueError("delay must be >= 0")
        delay = np.full(n, d)
    elif dist == "lognormal":
        delay = _lognormal(rng, take("median"), take("sigma"), n)
    elif dist == "mixture":
        p_fast = take("p_fast")
        fast, slow, sigma = take("fast_median"), take("slow_median"), take("sigma", 0.5)
        if not 0.0 <= p_fast <= 1.0:
            raise ValueError("p_fast must be in [0, 1]")
        is_fast = rng.random(n) < p_fast
        delay = np.where(is_fast, _lognormal(rng, fast, sigma, n), _lognormal(rng, slow, sigma, n))
    else:
        raise ValueError(f"unknown dist: {dist!r}")
    if kw:
        raise TypeError(f'unexpected parameters for dist="{dist}": {sorted(kw)}')

    if legit_delay is not None:
        delay = np.where(labels == 0, legit_delay, delay)
    never = (labels == 1) & (rng.random(n) < p_never)
    out: NDArray[np.float64] = np.where(never, np.inf, t + delay)
    return out


class AdversarialScores:
    """Lazy ``p_fraud`` for the ``"adversarial"`` scenario.

    Scores of adversarial frauds depend on the thresholds in force when they
    are scored, which exist only while the stream is replayed. Calling
    ``scores(start, stop)`` asks ``thresholds_fn`` for the current thresholds
    and places those frauds uniformly in ``[t_low - margin, t_low)`` (clipped
    to ``[0, 1]``), i.e. just below the auto-approve edge. Call it in stream
    order, just before predicting rows ``start:stop``; [`replay`][] does
    this. The realised scores are kept in ``values`` (``nan`` until set).

    All randomness is drawn up front, so results are deterministic given the
    generator and the sequence of thresholds.
    """

    def __init__(
        self,
        base: NDArray[np.float64],
        adversarial: NDArray[np.bool_],
        offsets: NDArray[np.float64],
        thresholds_fn: Callable[[], Thresholds],
    ) -> None:
        self._base = base
        self._adv = adversarial
        self._offsets = offsets
        self._thresholds_fn = thresholds_fn
        self.values = np.where(adversarial, np.nan, base)

    def __len__(self) -> int:
        return int(self._base.size)

    def __call__(self, start: int, stop: int) -> NDArray[np.float64]:
        sl = slice(start, stop)
        t_low = self._thresholds_fn().t_low
        adv = np.clip(t_low - self._offsets[sl], 0.0, 1.0)
        self.values[sl] = np.where(self._adv[sl], adv, self._base[sl])
        out: NDArray[np.float64] = self.values[sl].copy()
        return out


@dataclass(frozen=True)
class Stream:
    """A synthetic stream from [`make_stream`][].

    ``p_fraud`` is an array, or [`AdversarialScores`][] for the
    ``"adversarial"`` scenario. ``fraud_rate`` is the per-row probability
    that ``y = 1`` (the ground-truth prevalence path).
    """

    timestamps: NDArray[np.float64]
    p_fraud: NDArray[np.float64] | AdversarialScores
    y: NDArray[np.int_]
    fraud_rate: NDArray[np.float64]


def make_stream(
    n: int,
    base_fraud_rate: float,
    rng: np.random.Generator,
    scenario: Scenario = "stationary",
    *,
    horizon: float = 100.0,
    change_at: float = 0.5,
    shift_factor: float = 5.0,
    spike_window: tuple[float, float] = (0.6, 0.7),
    thresholds_fn: Callable[[], Thresholds] | None = None,
    adversarial_fraction: float = 0.5,
    margin: float = 0.02,
) -> Stream:
    """Generate a synthetic transaction stream scored by an imperfect model.

    Legit scores are ``Beta(1, 12)`` and fraud scores ``Beta(5, 2)``, so the
    classes overlap. Timestamps are sorted uniform draws on ``[0, horizon]``.

    Parameters
    ----------
    n : int
        Number of transactions.
    base_fraud_rate : float
        Fraud prevalence outside any shift or spike.
    rng : numpy.random.Generator
        Source of randomness.
    scenario : str, default "stationary"
        - ``"stationary"``: nothing changes.
        - ``"abrupt_prevalence_shift"``: from time ``change_at * horizon`` the
          fraud rate is ``shift_factor * base_fraud_rate``.
        - ``"score_drift"``: from ``change_at * horizon`` fraud scores come
          from ``Beta(2, 4)``; the model now scores new fraud lower.
        - ``"seasonal_spike"``: the fraud rate is ``shift_factor`` times
          higher inside ``spike_window`` (fractions of ``horizon``).
        - ``"adversarial"``: a fraction ``adversarial_fraction`` of frauds
          sit just below the current ``t_low``; requires ``thresholds_fn``
          and returns [`AdversarialScores`][] as ``p_fraud``.
    thresholds_fn : callable, optional
        Returns the thresholds in force now, e.g. ``lambda: clf.thresholds``.
        Only for ``"adversarial"``.
    margin : float, default 0.02
        Adversarial frauds land in ``[t_low - margin, t_low)``.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if (scenario == "adversarial") != (thresholds_fn is not None):
        raise ValueError('thresholds_fn is required for, and only for, scenario="adversarial"')
    t = np.sort(rng.uniform(0.0, horizon, n))
    after = t >= change_at * horizon
    rate = np.full(n, base_fraud_rate)
    if scenario == "abrupt_prevalence_shift":
        rate[after] *= shift_factor
    elif scenario == "seasonal_spike":
        lo, hi = spike_window
        rate[(t >= lo * horizon) & (t < hi * horizon)] *= shift_factor
    elif scenario not in ("stationary", "score_drift", "adversarial"):
        raise ValueError(f"unknown scenario: {scenario!r}")
    if np.any((rate < 0) | (rate > 1)):
        raise ValueError("fraud rate must stay in [0, 1]")
    y = (rng.random(n) < rate).astype(int)

    legit = rng.beta(*_LEGIT_BETA, n)
    fraud = rng.beta(*_FRAUD_BETA, n)
    if scenario == "score_drift":
        fraud = np.where(after, rng.beta(*_DRIFTED_FRAUD_BETA, n), fraud)
    base = np.where(y == 1, fraud, legit)

    p: NDArray[np.float64] | AdversarialScores = base
    if thresholds_fn is not None:
        if not 0.0 <= adversarial_fraction <= 1.0 or margin <= 0:
            raise ValueError("need adversarial_fraction in [0, 1] and margin > 0")
        adv = (y == 1) & (rng.random(n) < adversarial_fraction)
        # strictly inside (0, margin] so the score lands strictly below t_low
        offsets = margin * (1.0 - rng.random(n))
        p = AdversarialScores(base, adv, offsets, thresholds_fn)
    return Stream(timestamps=t, p_fraud=p, y=y, fraud_rate=rate)


@dataclass(frozen=True)
class ReplayResult:
    """Per-transaction outputs of [`replay`][], in stream order.

    ``sets`` and ``actions`` were produced with the thresholds in effect at
    prediction time (``t_low``, ``t_high``), so they can go straight into
    [`delaycp.metrics`][].
    """

    y: NDArray[np.int_]
    p_fraud: NDArray[np.float64]
    sets: NDArray[np.bool_]
    actions: NDArray[np.int_]
    t_low: NDArray[np.float64]
    t_high: NDArray[np.float64]
    label_times: NDArray[np.float64]


def replay(
    clf: OnlineConformalClassifier,
    timestamps: ArrayLike,
    p_fraud: ArrayLike | Callable[[int, int], Any],
    y: ArrayLike,
    label_times: ArrayLike,
    batch_size: int = 1,
) -> ReplayResult:
    """Run ``clf`` through a stream, revealing each label only at its label time.

    For each batch of ``batch_size`` consecutive rows: release every label
    that has arrived by the batch's first timestamp (``clf.advance``), predict
    the batch, then hand ``clf`` the batch's labels with their future arrival
    times (the delay buffer holds them until then; ``inf`` means never).

    Parameters
    ----------
    clf : OnlineConformalClassifier
        Fresh or warm-started classifier; it is mutated.
    timestamps : array_like of float
        Non-decreasing prediction times.
    p_fraud : array_like of float, or callable
        Scores, or ``f(start, stop)`` returning the scores of rows
        ``start:stop`` (called once per batch, in order, before predicting),
        such as [`AdversarialScores`][].
    y : array_like of int
        True labels.
    label_times : array_like of float
        Label arrival times from [`inject_delays`][]; ``nan`` is treated as
        ``inf``.
    batch_size : int, default 1
        Rows predicted together with the same thresholds.
    """
    t = np.asarray(timestamps, dtype=float)
    labels = np.asarray(y, dtype=int)
    lt = np.nan_to_num(np.asarray(label_times, dtype=float), nan=np.inf)
    n = t.size
    if t.ndim != 1 or labels.shape != (n,) or lt.shape != (n,):
        raise ValueError("timestamps, y and label_times must be 1-D and equal length")
    if np.any(lt < t):
        raise ValueError("label_times must not precede timestamps")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    score_fn: Callable[[int, int], Any]
    if callable(p_fraud):
        score_fn = p_fraud
    else:
        p_arr = np.asarray(p_fraud, dtype=float)
        if p_arr.shape != (n,):
            raise ValueError("p_fraud must have the same length as timestamps")

        def score_fn(start: int, stop: int) -> NDArray[np.float64]:
            return p_arr[start:stop]

    p_out = np.empty(n)
    sets = np.empty((n, 2), dtype=bool)
    actions = np.empty(n, dtype=int)
    t_low = np.empty(n)
    t_high = np.empty(n)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        clf.advance(float(t[start]))
        th = clf.thresholds
        p_b = np.asarray(score_fn(start, stop), dtype=float)
        if p_b.shape != (stop - start,):
            raise ValueError("score callable returned the wrong number of scores")
        ids = list(range(start, stop))
        actions[start:stop] = clf.predict(ids, t[start:stop], p_b)
        clf.add_labels(ids, labels[start:stop], lt[start:stop])
        p_out[start:stop] = p_b
        sets[start:stop] = sets_from_thresholds(p_b, th)
        t_low[start:stop] = th.t_low
        t_high[start:stop] = th.t_high
    return ReplayResult(labels, p_out, sets, actions, t_low, t_high, lt)
