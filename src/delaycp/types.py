"""Shared types: review actions and per-class thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    """Decision taken for a transaction given its conformal prediction set."""

    APPROVE = 0
    REVIEW = 1
    DECLINE = 2


@dataclass(frozen=True)
class Thresholds:
    """Per-class nonconformity-score thresholds.

    Parameters
    ----------
    q_0 : float
        Threshold on the legit score ``s(x, 0) = p``.
    q_1 : float
        Threshold on the fraud score ``s(x, 1) = 1 - p``.
    """

    q_0: float
    q_1: float

    @property
    def t_low(self) -> float:
        """Smallest ``p_fraud`` for which "fraud" is in the set (``1 - q_1``)."""
        return 1.0 - self.q_1

    @property
    def t_high(self) -> float:
        """Largest ``p_fraud`` for which "legit" is in the set (``q_0``)."""
        return self.q_0
