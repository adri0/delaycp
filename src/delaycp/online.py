"""Single-stream online quantile tracking with optional error integration."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["PIDQuantileTracker"]


def _tan_sat(x: float) -> float:
    """Tangent with ``tan(x) = sign(x) * inf`` for ``|x| >= pi / 2``."""
    if x >= math.pi / 2:
        return math.inf
    if x <= -math.pi / 2:
        return -math.inf
    return math.tan(x)


class PIDQuantileTracker:
    r"""Track one score threshold ``q`` targeting long-run miscoverage ``alpha``.

    Implements the P and I terms of conformal PID control (Angelopoulos,
    Candes & Tibshirani, 2023). After each update the threshold is::

        q = q_P + r_t(sum_i (err_i - alpha)),
        q_P <- q_P + lr * (err - alpha),
        r_t(x) = ki * tan(x * log(t) / (t * c_sat)),

    where ``err = 1`` iff ``score > q_used`` and ``t`` is the number of
    updates so far. ``tan(x)`` is taken as ``+/- inf`` once ``|x| >= pi/2``,
    so ``threshold`` may be infinite (the integrator's way of forcing
    coverage). The P state stays finite.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate, in ``(0, 1)``.
    lr : float
        Step size of the P (quantile tracking) term, ``> 0``.
    ki : float, default 0.0
        Integrator gain ``K_I``. ``0`` disables the integrator (P only).
    c_sat : float, default 1.0
        Saturation constant ``C_sat``, ``> 0``. Only used when ``ki > 0``.
    q_init : float, optional
        Initial threshold. Defaults to ``1 - alpha``. Not clipped.

    Notes
    -----
    The threshold is deliberately never clipped to ``[0, 1]``. Quantile
    tracking only guarantees coverage because ``q`` can leave the score range:
    ``q < 0`` forces an error and ``q > 1`` forbids one, and these are what
    drive the iterate back. Clipping would break the long-run bound.

    ``update`` takes ``q_used`` explicitly, so labels may arrive late: the
    error is judged against the threshold in effect at prediction time.

    Differences from the reference repo (``core/methods.py``): its integrator
    sums errors over ``[:t]`` (excluding the newest one) with ``t`` 0-indexed,
    so it lags the paper by one step; this class includes the newest error
    with ``t = n_updates``. The reference also rescales ``lr`` by a rolling
    score range by default (``proportional_lr``); this class uses a fixed
    ``lr``.
    """

    def __init__(
        self,
        alpha: float,
        lr: float,
        ki: float = 0.0,
        c_sat: float = 1.0,
        q_init: float | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        if ki < 0.0:
            raise ValueError("ki must be >= 0")
        if c_sat <= 0.0:
            raise ValueError("c_sat must be > 0")
        self._alpha = alpha
        self._lr = lr
        self._ki = ki
        self._c_sat = c_sat
        self._q_p = (1.0 - alpha) if q_init is None else float(q_init)
        self._err_sum = 0.0  # running sum of (err - alpha)
        self._n = 0

    @property
    def threshold(self) -> float:
        """Current threshold ``q`` (may lie outside ``[0, 1]`` or be infinite)."""
        if self._ki == 0.0 or self._n < 2:  # log(1) = 0, so r_1 = 0
            return self._q_p
        t = self._n
        arg = self._err_sum * math.log(t) / (t * self._c_sat)
        return self._q_p + self._ki * _tan_sat(arg)

    @property
    def n_updates(self) -> int:
        """Number of scores seen so far."""
        return self._n

    def update(self, score: float, q_used: float) -> None:
        """Incorporate a matured score.

        Parameters
        ----------
        score : float
            Observed nonconformity score.
        q_used : float
            Threshold that was in effect when the prediction was made.
        """
        err = 1.0 if score > q_used else 0.0
        self._q_p += self._lr * (err - self._alpha)
        self._err_sum += err - self._alpha
        self._n += 1

    def init_from_scores(self, scores: Sequence[float] | np.ndarray) -> None:
        """Set ``q`` to the split-conformal ``(1 - alpha)`` quantile.

        Uses the finite-sample correction: the ``ceil((n + 1) * (1 - alpha))``-th
        smallest score. If that rank exceeds ``n`` the exact answer is
        ``+inf``; the largest score is used instead so the tracker stays
        usable. The result is not clipped to ``[0, 1]``. Also resets the
        integrator sum and ``n_updates`` to 0.

        Parameters
        ----------
        scores : sequence of float
            Calibration scores, at least one.
        """
        arr = np.sort(np.asarray(scores, dtype=float))
        n = arr.size
        if n == 0:
            raise ValueError("scores must be non-empty")
        rank = min(math.ceil((n + 1) * (1.0 - self._alpha)), n)
        self._q_p = float(arr[rank - 1])
        self._err_sum = 0.0
        self._n = 0
