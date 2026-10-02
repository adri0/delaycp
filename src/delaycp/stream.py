"""Streaming orchestrator: predict, buffer, and update with delayed labels."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from delaycp.delay import DelayBuffer
from delaycp.mondrian import MondrianPID
from delaycp.types import Action, Thresholds

__all__ = ["History", "OnlineConformalClassifier", "PredictionRecord", "UpdateRecord"]


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

    Combines :class:`~delaycp.mondrian.MondrianPID` and
    :class:`~delaycp.delay.DelayBuffer`. Each prediction is registered with
    the thresholds in effect when it is made; its update is applied only once
    its label arrives (or it matures), in release order. Batches must arrive
    in time order.

    Parameters
    ----------
    alpha_fraud, alpha_legit : float
        Target miscoverage per class.
    lr : float
        PID step size.
    ki : float, default 0.0
        PID integral gain.
    maturity : float or None, default None
        Maximum label wait; see :class:`~delaycp.delay.DelayBuffer`.
    unlabeled_at_maturity : int or None, default 0
        Label imputed for items that mature unlabeled; ``None`` drops them.
    history_size : int, default 0
        Maximum number of records kept *per log* (predictions and updates).
        ``0`` disables logging.
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
    ) -> None:
        if history_size < 0:
            raise ValueError("history_size must be >= 0")
        self._model = MondrianPID(alpha_fraud, alpha_legit, lr, ki=ki)
        self._buffer = DelayBuffer(maturity, unlabeled_at_maturity)
        self._log = history_size > 0
        self.history = History(
            predictions=deque(maxlen=history_size), updates=deque(maxlen=history_size)
        )
        self._last_time = -np.inf

    @property
    def thresholds(self) -> Thresholds:
        """Current per-class thresholds."""
        return self._model.thresholds

    @property
    def pending_count(self) -> int:
        """Predictions awaiting a label or maturity."""
        return self._buffer.pending_count

    def fit_calibration(self, p_fraud: ArrayLike, y: ArrayLike) -> None:
        """Warm-start the thresholds from labelled calibration data."""
        self._model.fit_calibration(p_fraud, y)

    def predict(
        self,
        tx_ids: Sequence[Hashable],
        timestamps: ArrayLike,
        p_fraud: ArrayLike,
    ) -> np.ndarray:
        """Predict actions for a batch and register each with the current thresholds.

        Returns
        -------
        numpy.ndarray
            Integer :class:`~delaycp.types.Action` values, one per transaction.

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
        actions = self._model.predict_actions(p)
        th = self._model.thresholds
        for i, tx_id in enumerate(tx_ids):
            self._buffer.register(tx_id, float(t[i]), float(p[i]), th)
            if self._log:
                self.history.predictions.append(
                    PredictionRecord(tx_id, float(t[i]), float(p[i]), th, Action(actions[i]))
                )
        self._last_time = float(t[-1])
        return actions

    def add_labels(
        self,
        tx_ids: Sequence[Hashable],
        ys: ArrayLike,
        label_times: ArrayLike,
    ) -> None:
        """Record arrived labels; they take effect at the next :meth:`advance`."""
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
            Number of threshold updates applied.
        """
        released = self._buffer.pop_ready(now)
        for r in released:
            if self._log:
                q_used = r.thresholds_used.q_1 if r.y == 1 else r.thresholds_used.q_0
                score = 1.0 - r.p_fraud if r.y == 1 else r.p_fraud
                self.history.updates.append(
                    UpdateRecord(r.tx_id, r.y, bool(score > q_used), r.matured)
                )
            self._model.update(r.p_fraud, r.y, r.thresholds_used)
        return len(released)
