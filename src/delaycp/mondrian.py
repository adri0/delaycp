"""Mondrian (per-class) conformal wrapper producing sets and actions."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from delaycp.online import PIDQuantileTracker
from delaycp.types import Action, Thresholds


def sets_from_thresholds(p_fraud: ArrayLike, th: Thresholds) -> NDArray[np.bool_]:
    """Boolean ``(n, 2)`` array; column ``y`` is True iff class ``y`` is in the set."""
    p = np.asarray(p_fraud, dtype=float)
    return np.stack([p <= th.t_high, p >= th.t_low], axis=1)


def actions_from_thresholds(p_fraud: ArrayLike, th: Thresholds) -> NDArray[np.int_]:
    """Map sets to actions: ``{0}`` APPROVE, ``{1}`` DECLINE, ``{0,1}``/empty REVIEW."""
    sets = sets_from_thresholds(p_fraud, th)
    actions = np.full(sets.shape[0], Action.REVIEW, dtype=int)
    actions[sets[:, 0] & ~sets[:, 1]] = Action.APPROVE
    actions[~sets[:, 0] & sets[:, 1]] = Action.DECLINE
    return actions


class MondrianPID:
    """One :class:`PIDQuantileTracker` per class, on ``s(x, y) = 1 - p_hat(y | x)``.

    Parameters
    ----------
    alpha_fraud, alpha_legit : float
        Target miscoverage for the fraud (1) and legit (0) classes.
    lr : float
        Step size of both trackers.
    ki : float, default 0.0
        Integral gain of both trackers.
    min_class_count : int, default 50
        Warn in :meth:`fit_calibration` if a class has fewer examples.
    """

    def __init__(
        self,
        alpha_fraud: float,
        alpha_legit: float,
        lr: float,
        ki: float = 0.0,
        min_class_count: int = 50,
    ) -> None:
        self.min_class_count = min_class_count
        self._trackers = (
            PIDQuantileTracker(alpha_legit, lr, ki=ki),
            PIDQuantileTracker(alpha_fraud, lr, ki=ki),
        )

    def fit_calibration(self, p_fraud: ArrayLike, y: ArrayLike) -> None:
        """Warm-start each tracker from its own class's scores."""
        p = np.asarray(p_fraud, dtype=float)
        labels = np.asarray(y, dtype=int)
        for cls, tracker in enumerate(self._trackers):
            p_cls = p[labels == cls]
            if p_cls.size < self.min_class_count:
                warnings.warn(
                    f"class {cls} has only {p_cls.size} calibration examples "
                    f"(< min_class_count={self.min_class_count}); its threshold is unreliable",
                    UserWarning,
                    stacklevel=2,
                )
            if p_cls.size:
                tracker.init_from_scores(p_cls if cls == 0 else 1.0 - p_cls)

    @property
    def thresholds(self) -> Thresholds:
        """Current per-class thresholds."""
        return Thresholds(q_0=self._trackers[0].threshold, q_1=self._trackers[1].threshold)

    def predict_sets(self, p_fraud: ArrayLike) -> NDArray[np.bool_]:
        """Boolean ``(n, 2)`` array; column ``y`` is True iff class ``y`` is in the set."""
        return sets_from_thresholds(p_fraud, self.thresholds)

    def predict_actions(self, p_fraud: ArrayLike) -> NDArray[np.int_]:
        """Map sets to actions: ``{0}`` APPROVE, ``{1}`` DECLINE, ``{0,1}``/empty REVIEW."""
        return actions_from_thresholds(p_fraud, self.thresholds)

    def update(self, p_fraud: float, y: int, thresholds_used: Thresholds) -> None:
        """Update only the true class's tracker, with the threshold used at prediction time."""
        if y == 1:
            self._trackers[1].update(1.0 - p_fraud, thresholds_used.q_1)
        elif y == 0:
            self._trackers[0].update(p_fraud, thresholds_used.q_0)
        else:
            raise ValueError("y must be 0 or 1")
