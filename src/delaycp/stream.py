"""Streaming orchestrator: predict, buffer, and update with delayed labels."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from delaycp.budget import BudgetController
from delaycp.delay import DelayBuffer
from delaycp.mondrian import MondrianPID, actions_from_thresholds
from delaycp.types import Action, Thresholds

__all__ = [
    "CoverageEstimate",
    "History",
    "OnlineConformalClassifier",
    "PredictionRecord",
    "UpdateRecord",
]


class CoverageEstimate(NamedTuple):
    """Empirical fraud coverage over the most recent fraud labels.

    ``coverage`` is ``nan`` when ``n_labels`` is 0.
    """

    coverage: float
    n_labels: int


@dataclass(frozen=True)
class PredictionRecord:
    """Log entry for one prediction."""

    tx_id: Hashable
    pred_time: float
    p_fraud: float
    thresholds_used: Thresholds
    action: Action


@dataclass(frozen=True)
class UpdateRecord:
    """Log entry for one threshold update.

    ``miscovered`` is judged against ``thresholds_used`` (those in effect at
    prediction time). ``matured`` marks labels imputed at maturity.
    """

    tx_id: Hashable
    y: int
    miscovered: bool
    matured: bool


@dataclass
class History:
    """Bounded logs; the oldest entries are discarded first."""

    predictions: deque[PredictionRecord] = field(default_factory=deque)
    updates: deque[UpdateRecord] = field(default_factory=deque)


class OnlineConformalClassifier:
    """Mondrian PID conformal classifier for streams with delayed labels.

    Combines [`MondrianPID`][delaycp.mondrian.MondrianPID] and
    [`DelayBuffer`][delaycp.delay.DelayBuffer]. Each prediction is registered with
    the thresholds in effect when it is made; its update is applied only once
    its label arrives (or it matures), in release order. Batches must arrive
    in time order.

    In ``mode="coverage"`` both band edges are conformal thresholds. In
    ``mode="budget"`` ``t_high`` still comes from the legit tracker, but
    ``t_low`` comes from a [`BudgetController`][delaycp.budget.BudgetController] that
    holds the REVIEW rate near ``review_budget``; fraud labels are then used
    only for [`estimated_fraud_coverage`][]. See [`delaycp.budget`][] for
    the trade-off between the two modes.

    Parameters
    ----------
    alpha_fraud, alpha_legit : float
        Target miscoverage per class.
    lr : float
        PID step size.
    ki : float, default 0.0
        PID integral gain.
    maturity : float or None, default None
        Maximum label wait; see [`DelayBuffer`][delaycp.delay.DelayBuffer].
    unlabeled_at_maturity : int or None, default 0
        Label imputed for items that mature unlabeled; ``None`` drops them.
    history_size : int, default 0
        Maximum number of records kept *per log* (predictions and updates).
        ``0`` disables logging.
    mode : {"coverage", "budget"}, default "coverage"
        What ``t_low`` targets: fraud coverage, or the REVIEW rate.
    review_budget : float, optional
        Target REVIEW rate (e.g. ``0.005``). Required in budget mode and
        rejected in coverage mode.
    budget_lr : float, optional
        Per-batch step size of the budget controller. Defaults to ``lr``.
    fraud_log_size : int, default 10_000
        Number of recent fraud outcomes kept for
        [`estimated_fraud_coverage`][].
    """

    def __init__(
        self,
        alpha_fraud: float,
        alpha_legit: float,
        lr: float,
        ki: float = 0.0,
        maturity: float | None = None,
        unlabeled_at_maturity: int | None = 0,
        history_size: int = 0,
        mode: Literal["coverage", "budget"] = "coverage",
        review_budget: float | None = None,
        budget_lr: float | None = None,
        fraud_log_size: int = 10_000,
    ) -> None:
        if history_size < 0:
            raise ValueError("history_size must be >= 0")
        if fraud_log_size < 1:
            raise ValueError("fraud_log_size must be >= 1")
        self._model = MondrianPID(alpha_fraud, alpha_legit, lr, ki=ki)
        self._budget: BudgetController | None = None
        if mode == "budget":
            if review_budget is None:
                raise ValueError("budget mode requires review_budget")
            self._budget = BudgetController(
                review_budget,
                lr if budget_lr is None else budget_lr,
                t_low_init=self._model.thresholds.t_low,
            )
        elif mode == "coverage":
            if review_budget is not None or budget_lr is not None:
                raise ValueError("review_budget and budget_lr apply only to budget mode")
        else:
            raise ValueError('mode must be "coverage" or "budget"')
        self._mode = mode
        self._fraud_covered: deque[bool] = deque(maxlen=fraud_log_size)
        self._buffer = DelayBuffer(maturity, unlabeled_at_maturity)
        self._log = history_size > 0
        self.history = History(
            predictions=deque(maxlen=history_size), updates=deque(maxlen=history_size)
        )
        self._last_time = -np.inf

    @property
    def mode(self) -> Literal["coverage", "budget"]:
        """``"coverage"`` or ``"budget"``."""
        return self._mode

    @property
    def thresholds(self) -> Thresholds:
        """Thresholds that the next prediction will use.

        In budget mode ``q_1 = 1 - t_low`` with ``t_low`` from the budget
        controller, clipped so that ``t_low <= t_high``.
        """
        th = self._model.thresholds
        if self._budget is None:
            return th
        return Thresholds(q_0=th.q_0, q_1=1.0 - self._budget.effective_t_low(th.t_high))

    @property
    def pending_count(self) -> int:
        """Predictions awaiting a label or maturity."""
        return self._buffer.pending_count

    def fit_calibration(self, p_fraud: ArrayLike, y: ArrayLike) -> None:
        """Warm-start the thresholds from labelled calibration data.

        In budget mode ``t_low`` is set so the calibration REVIEW rate matches
        the budget, given the warm-started ``t_high``.
        """
        self._model.fit_calibration(p_fraud, y)
        if self._budget is not None:
            self._budget.init_from_scores(p_fraud, self._model.thresholds.t_high)

    def predict(
        self,
        tx_ids: Sequence[Hashable],
        timestamps: ArrayLike,
        p_fraud: ArrayLike,
    ) -> np.ndarray:
        """Predict actions for a batch and register each with the current thresholds.

        In budget mode the batch's REVIEW rate then updates ``t_low``, so the
        next batch already reflects it (no label needed).

        Returns
        -------
        numpy.ndarray
            Integer [`Action`][delaycp.types.Action] values, one per transaction.

        Raises
        ------
        ValueError
            On length mismatch, or timestamps not in non-decreasing order
            (within the batch and relative to earlier batches).
        """
        t = np.asarray(timestamps, dtype=float)
        p = np.asarray(p_fraud, dtype=float)
        if not (len(tx_ids) == t.shape[0] == p.shape[0]) or t.ndim != 1 or p.ndim != 1:
            raise ValueError("tx_ids, timestamps and p_fraud must be 1-D and equal length")
        if t.size == 0:
            return np.empty(0, dtype=int)
        if t[0] < self._last_time or np.any(np.diff(t) < 0):
            raise ValueError("timestamps must be in non-decreasing order")
        th = self.thresholds
        actions = actions_from_thresholds(p, th)
        for i, tx_id in enumerate(tx_ids):
            self._buffer.register(tx_id, float(t[i]), float(p[i]), th)
            if self._log:
                self.history.predictions.append(
                    PredictionRecord(tx_id, float(t[i]), float(p[i]), th, Action(actions[i]))
                )
        self._last_time = float(t[-1])
        if self._budget is not None:
            self._budget.update(float(np.mean(actions == Action.REVIEW)))
        return actions

    def add_labels(
        self,
        tx_ids: Sequence[Hashable],
        ys: ArrayLike,
        label_times: ArrayLike,
    ) -> None:
        """Record arrived labels; they take effect at the next [`advance`][]."""
        y = np.asarray(ys, dtype=int)
        lt = np.asarray(label_times, dtype=float)
        if not (len(tx_ids) == y.shape[0] == lt.shape[0]):
            raise ValueError("tx_ids, ys and label_times must have equal length")
        for tx_id, yi, ti in zip(tx_ids, y, lt, strict=True):
            self._buffer.add_label(tx_id, int(yi), float(ti))

    def advance(self, now: float) -> int:
        """Apply every update ready at ``now``, in release order.

        Returns
        -------
        int
            Number of labels released (in budget mode, fraud labels only feed
            [`estimated_fraud_coverage`][]).
        """
        released = self._buffer.pop_ready(now)
        for r in released:
            q_used = r.thresholds_used.q_1 if r.y == 1 else r.thresholds_used.q_0
            score = 1.0 - r.p_fraud if r.y == 1 else r.p_fraud
            miscovered = bool(score > q_used)
            if r.y == 1:
                self._fraud_covered.append(not miscovered)
            if self._log:
                self.history.updates.append(UpdateRecord(r.tx_id, r.y, miscovered, r.matured))
            if self._budget is None or r.y == 0:
                self._model.update(r.p_fraud, r.y, r.thresholds_used)
        return len(released)

    def estimated_fraud_coverage(self, window: int) -> CoverageEstimate:
        """Fraction of the last ``window`` released fraud labels that were covered.

        A fraud is covered iff ``p_fraud >= t_low`` with the ``t_low`` in
        effect when it was predicted (i.e. it was not auto-approved). Labels
        enter in release order, so the estimate lags by the label delay.
        Available in both modes; in budget mode it is the only fraud-side
        feedback.

        Parameters
        ----------
        window : int
            Number of most recent fraud labels to use, in
            ``[1, fraud_log_size]``.

        Returns
        -------
        CoverageEstimate
            Coverage and the number of fraud labels it is based on
            (``min(window, fraud labels seen)``).
        """
        maxlen = self._fraud_covered.maxlen
        assert maxlen is not None
        if not 1 <= window <= maxlen:
            raise ValueError(f"window must be in [1, fraud_log_size={maxlen}]")
        n = min(window, len(self._fraud_covered))
        if n == 0:
            return CoverageEstimate(math.nan, 0)
        recent = list(self._fraud_covered)[-n:]
        return CoverageEstimate(sum(recent) / n, n)
