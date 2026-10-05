"""Budget mode: hold the REVIEW rate near a target instead of fraud coverage.

Coverage mode vs budget mode
----------------------------
In *coverage mode* both band edges are conformal thresholds: ``t_low`` is
driven by matured fraud labels so that fraud miscoverage tracks
``alpha_fraud``. The review queue is whatever that takes: when fraud drifts
towards lower scores, ``t_low`` falls and REVIEW volume grows, possibly beyond
what analysts can handle. The correction also waits for the label delay.

In *budget mode* ``t_low`` is instead steered so that the fraction of REVIEW
actions tracks ``review_budget``. The review indicator is known as soon as a
prediction is made, so the controller reacts immediately and the queue stays
on budget. The price is that fraud coverage is no longer guaranteed; it
becomes an *output* that can only be estimated from delayed fraud labels, and
a drift that pushes fraud into lower scores shows up as a coverage drop one
label delay later. ``t_high`` (and hence legit coverage, i.e. the rate of
auto-declined good customers) is a coverage target in both modes.

Use coverage mode when missing fraud is the binding constraint, and budget
mode when review capacity is.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["BudgetController"]


class BudgetController:
    r"""Online tracker steering ``t_low`` so the REVIEW rate tracks a budget.

    After each batch with observed review rate ``r``::

        t_low <- t_low + lr * (r - review_budget)

    so the band ``[t_low, t_high]`` narrows when over budget and widens when
    under it. This is quantile tracking (Angelopoulos, Candes & Tibshirani,
    2023) with the REVIEW indicator in place of the miscoverage indicator.

    Parameters
    ----------
    review_budget : float
        Target fraction of REVIEW actions, in ``(0, 1)``.
    lr : float
        Step size, ``> 0``.
    t_low_init : float
        Initial ``t_low``.

    Notes
    -----
    The internal state is never clipped; only the value used for decisions
    is, via [`effective_t_low`][] (``min(t_low, t_high)``). If the state
    exceeds ``t_high`` the band collapses to a point, the review rate drops
    below budget and the state is driven back down, so it stays bounded
    whenever the budget is attainable (``review_budget`` below the fraction of
    scores ``<= t_high``). Then, since ``t_low_T - t_low_0 = lr * sum(r - b)``,
    the *batch-averaged* review rate converges to the budget at rate
    ``O(1 / (lr * T))``. Each batch counts equally regardless of its size, so
    with unequal batch sizes this is not the per-transaction rate.
    """

    def __init__(self, review_budget: float, lr: float, t_low_init: float) -> None:
        if not 0.0 < review_budget < 1.0:
            raise ValueError("review_budget must be in (0, 1)")
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        self._budget = review_budget
        self._lr = lr
        self._t_low = float(t_low_init)
        self._n = 0

    @property
    def review_budget(self) -> float:
        """Target REVIEW rate."""
        return self._budget

    @property
    def t_low(self) -> float:
        """Unclipped ``t_low`` state (may lie outside ``[0, 1]``)."""
        return self._t_low

    @property
    def n_updates(self) -> int:
        """Number of batches seen so far."""
        return self._n

    def effective_t_low(self, t_high: float) -> float:
        """``t_low`` used for decisions: the state clipped to at most ``t_high``."""
        return min(self._t_low, t_high)

    def update(self, review_rate: float) -> None:
        """Incorporate the review rate observed on one batch."""
        if not 0.0 <= review_rate <= 1.0:
            raise ValueError("review_rate must be in [0, 1]")
        self._t_low += self._lr * (review_rate - self._budget)
        self._n += 1

    def init_from_scores(self, p_fraud: ArrayLike, t_high: float) -> None:
        """Set ``t_low`` so that ``ceil(budget * n)`` calibration scores fall in the band.

        The band is ``[t_low, t_high]``. If fewer scores lie at or below
        ``t_high`` than that, ``t_low`` is set to the smallest of them (the
        budget is then not attainable on this data). Resets ``n_updates``.

        Parameters
        ----------
        p_fraud : array_like
            Calibration fraud probabilities (all classes), at least one.
        t_high : float
            Current upper edge of the band.
        """
        p = np.asarray(p_fraud, dtype=float)
        if p.size == 0:
            raise ValueError("p_fraud must be non-empty")
        below = np.sort(p[p <= t_high])
        k = math.ceil(self._budget * p.size)
        if below.size == 0:
            self._t_low = t_high
        else:
            self._t_low = float(below[-min(k, below.size)])
        self._n = 0
