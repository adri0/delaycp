"""Evaluation metrics for conformal fraud decisions.

Pure functions over aligned arrays: ``y`` (0/1 labels), ``sets`` (``(n, 2)``
bool, column ``c`` True iff class ``c`` is in the set) and ``actions``
(:class:`~delaycp.types.Action` ints). Coverage of an example is judged on the
set it was given at prediction time, so pass the sets that were actually
produced, not ones recomputed from current thresholds.
"""

from __future__ import annotations

import math
from typing import NamedTuple, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from delaycp.stream import CoverageEstimate
from delaycp.types import Action

__all__ = [
    "ActionRates",
    "Cost",
    "Summary",
    "action_rates",
    "class_coverage",
    "cost",
    "coverage_measured_at",
    "fraud_auto_approved_rate",
    "marginal_coverage",
    "rolling_coverage",
    "summary",
    "worst_window_coverage",
]


class ActionRates(NamedTuple):
    """Fraction of transactions per action; ``empty`` is ``nan`` if sets were not given.

    Empty sets are routed to REVIEW, so ``empty`` is a subset of ``review``.
    """

    approve: float
    review: float
    decline: float
    empty: float


class Cost(NamedTuple):
    """Total cost and mean cost per transaction (``nan`` if there are none)."""

    total: float
    per_transaction: float


class Summary(TypedDict, total=False):
    """Report returned by :func:`summary`; optional keys appear only when requested."""

    n: int
    class_coverage: dict[int, CoverageEstimate]
    marginal_coverage: float
    action_rates: ActionRates
    fraud_auto_approved_rate: float
    worst_window_fraud_coverage: float
    cost: Cost


def _labels(y: ArrayLike) -> NDArray[np.int_]:
    labels = np.asarray(y, dtype=int)
    if labels.ndim != 1:
        raise ValueError("y must be 1-D")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("y must contain only 0 and 1")
    return labels


def _labels_and_sets(y: ArrayLike, sets: ArrayLike) -> tuple[NDArray[np.int_], NDArray[np.bool_]]:
    labels = _labels(y)
    s = np.asarray(sets, dtype=bool)
    if s.shape != (labels.shape[0], 2):
        raise ValueError("sets must have shape (len(y), 2)")
    return labels, s


def _covered(labels: NDArray[np.int_], s: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Whether each example's true label is in its set."""
    return s[np.arange(labels.shape[0]), labels]


def _rate(mask: NDArray[np.bool_]) -> float:
    return float(np.mean(mask)) if mask.size else math.nan


def class_coverage(y: ArrayLike, sets: ArrayLike) -> dict[int, CoverageEstimate]:
    """Per-class coverage ``P(y in set | y = c)`` with the number of examples of each class.

    Returns
    -------
    dict[int, CoverageEstimate]
        Keys ``0`` and ``1``; coverage is ``nan`` for a class with no examples.
    """
    labels, s = _labels_and_sets(y, sets)
    hit = _covered(labels, s)
    out: dict[int, CoverageEstimate] = {}
    for c in (0, 1):
        mask = labels == c
        out[c] = CoverageEstimate(_rate(hit[mask]), int(mask.sum()))
    return out


def marginal_coverage(y: ArrayLike, sets: ArrayLike) -> float:
    """Fraction of examples whose true label is in the set (``nan`` if empty).

    Dominated by the majority (legit) class; prefer :func:`class_coverage`.
    """
    labels, s = _labels_and_sets(y, sets)
    return _rate(_covered(labels, s))


def rolling_coverage(
    y: ArrayLike, sets: ArrayLike, window: int, cls: int | None = None
) -> NDArray[np.float64]:
    """Coverage over each run of ``window`` consecutive examples.

    Parameters
    ----------
    y, sets
        Labels and prediction sets, in stream order.
    window : int
        Window length, counted in examples of class ``cls`` when it is given
        (so ``cls=1`` gives coverage over each ``window`` consecutive frauds).
    cls : {0, 1} or None
        Restrict to examples of this class.

    Returns
    -------
    numpy.ndarray
        Length ``max(m - window + 1, 0)`` where ``m`` is the number of
        examples used; entry ``i`` covers examples ``i .. i + window - 1``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    labels, s = _labels_and_sets(y, sets)
    hit = _covered(labels, s)
    if cls is not None:
        if cls not in (0, 1):
            raise ValueError("cls must be 0, 1 or None")
        hit = hit[labels == cls]
    if hit.size < window:
        return np.empty(0, dtype=float)
    csum = np.concatenate([[0], np.cumsum(hit, dtype=np.int64)])
    out: NDArray[np.float64] = (csum[window:] - csum[:-window]) / window
    return out


def worst_window_coverage(
    y: ArrayLike, sets: ArrayLike, window: int, cls: int | None = 1
) -> float:
    """Minimum of :func:`rolling_coverage` (``nan`` if fewer than ``window`` examples).

    A one-sided version of the local coverage error of Bhatnagar et al.
    (2023), ``max_windows |coverage - (1 - alpha)|``: for fraud, only
    under-coverage is harmful, so the worst window is the lowest one.
    """
    rc = rolling_coverage(y, sets, window, cls)
    return float(np.min(rc)) if rc.size else math.nan


def action_rates(actions: ArrayLike, sets: ArrayLike | None = None) -> ActionRates:
    """Fraction of APPROVE, REVIEW and DECLINE actions, plus the empty-set rate.

    Parameters
    ----------
    actions : array_like of int
        :class:`~delaycp.types.Action` values.
    sets : array_like of bool, optional
        ``(n, 2)`` prediction sets. Needed for ``empty``, since an empty set
        and a two-class set both map to REVIEW.
    """
    a = np.asarray(actions, dtype=int)
    if a.ndim != 1:
        raise ValueError("actions must be 1-D")
    empty = math.nan
    if sets is not None:
        s = np.asarray(sets, dtype=bool)
        if s.shape != (a.shape[0], 2):
            raise ValueError("sets must have shape (len(actions), 2)")
        empty = _rate(np.asarray(~s.any(axis=1)))
    return ActionRates(
        approve=_rate(a == Action.APPROVE),
        review=_rate(a == Action.REVIEW),
        decline=_rate(a == Action.DECLINE),
        empty=empty,
    )


def fraud_auto_approved_rate(y: ArrayLike, actions: ArrayLike) -> float:
    """Fraction of frauds that were APPROVEd, ``P(APPROVE | y = 1)`` (``nan`` if no fraud).

    At most ``1 - fraud coverage``: a fraud with an empty set is miscovered
    but sent to REVIEW, not approved.
    """
    labels = _labels(y)
    a = np.asarray(actions, dtype=int)
    if a.shape != labels.shape:
        raise ValueError("y and actions must have equal length")
    return _rate(a[labels == 1] == Action.APPROVE)


def cost(
    y: ArrayLike,
    actions: ArrayLike,
    c_missed_fraud: float,
    c_false_decline: float,
    c_review: float,
) -> Cost:
    """Operational cost of the decisions.

    Each approved fraud costs ``c_missed_fraud``, each declined legit costs
    ``c_false_decline`` and each REVIEW costs ``c_review`` whatever the label
    (the review is assumed to resolve it correctly). Correct automatic
    decisions cost nothing.
    """
    labels = _labels(y)
    a = np.asarray(actions, dtype=int)
    if a.shape != labels.shape:
        raise ValueError("y and actions must have equal length")
    n_missed = int(np.sum((labels == 1) & (a == Action.APPROVE)))
    n_false_decline = int(np.sum((labels == 0) & (a == Action.DECLINE)))
    n_review = int(np.sum(a == Action.REVIEW))
    total = c_missed_fraud * n_missed + c_false_decline * n_false_decline + c_review * n_review
    return Cost(float(total), total / a.size if a.size else math.nan)


def coverage_measured_at(
    now: float,
    y: ArrayLike,
    sets: ArrayLike,
    label_times: ArrayLike,
    cls: int | None = None,
) -> CoverageEstimate:
    """Coverage over only the labels known by ``now``, as production would see it.

    Compare with :func:`class_coverage` / :func:`marginal_coverage` on all
    labels (the truth) to see how much delay hides. Recent predictions,
    which reflect the newest thresholds, are the ones still missing.

    Parameters
    ----------
    now : float
        Evaluation time; a label counts iff ``label_time <= now``.
    y, sets
        Labels and prediction sets.
    label_times : array_like of float
        Arrival time of each label; ``inf`` or ``nan`` for never.
    cls : {0, 1} or None
        Restrict to examples of this class.
    """
    labels, s = _labels_and_sets(y, sets)
    lt = np.asarray(label_times, dtype=float)
    if lt.shape != labels.shape:
        raise ValueError("label_times must have the same length as y")
    known = lt <= now  # False for nan
    if cls is not None:
        if cls not in (0, 1):
            raise ValueError("cls must be 0, 1 or None")
        known &= labels == cls
    hit = _covered(labels, s)[known]
    return CoverageEstimate(_rate(hit), int(hit.size))


def summary(
    y: ArrayLike,
    sets: ArrayLike,
    actions: ArrayLike,
    *,
    fraud_window: int | None = None,
    c_missed_fraud: float | None = None,
    c_false_decline: float | None = None,
    c_review: float | None = None,
) -> Summary:
    """One-call report of the metrics in this module.

    Parameters
    ----------
    y, sets, actions
        Aligned labels, prediction sets and actions.
    fraud_window : int, optional
        If given, include ``worst_window_fraud_coverage`` over windows of this
        many frauds.
    c_missed_fraud, c_false_decline, c_review : float, optional
        If all are given, include ``cost``. Giving only some is an error.
    """
    labels, s = _labels_and_sets(y, sets)
    report: Summary = {
        "n": int(labels.size),
        "class_coverage": class_coverage(labels, s),
        "marginal_coverage": marginal_coverage(labels, s),
        "action_rates": action_rates(actions, s),
        "fraud_auto_approved_rate": fraud_auto_approved_rate(labels, actions),
    }
    if fraud_window is not None:
        report["worst_window_fraud_coverage"] = worst_window_coverage(
            labels, s, fraud_window, cls=1
        )
    costs = (c_missed_fraud, c_false_decline, c_review)
    if all(c is not None for c in costs):
        assert c_missed_fraud is not None and c_false_decline is not None
        assert c_review is not None
        report["cost"] = cost(labels, actions, c_missed_fraud, c_false_decline, c_review)
    elif any(c is not None for c in costs):
        raise ValueError("give all of c_missed_fraud, c_false_decline, c_review or none")
    return report
