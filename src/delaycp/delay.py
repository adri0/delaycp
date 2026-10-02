"""Delay buffer: hold predictions until their (delayed) labels arrive."""

from __future__ import annotations

import heapq
from collections.abc import Hashable
from dataclasses import dataclass

from delaycp.types import Thresholds


@dataclass(frozen=True)
class Released:
    """A prediction released by :class:`DelayBuffer` for updating.

    Parameters
    ----------
    tx_id : Hashable
        Transaction identifier given at registration.
    p_fraud : float
        Predicted fraud probability.
    y : int
        Observed label, or the buffer's ``unlabeled_at_maturity`` value if
        ``matured`` is true.
    thresholds_used : Thresholds
        Thresholds in effect when the prediction was made.
    pred_time : float
        Timestamp of the prediction.
    label_time : float
        Time the label arrived; for matured items, the time maturity was reached.
    matured : bool
        True if released because it reached maturity without a label.
    """

    tx_id: Hashable
    p_fraud: float
    y: int
    thresholds_used: Thresholds
    pred_time: float
    label_time: float
    matured: bool


class _Entry:
    __slots__ = ("p_fraud", "pred_time", "seq", "thresholds", "y", "label_time")

    def __init__(self, pred_time: float, p_fraud: float, thresholds: Thresholds, seq: int) -> None:
        self.pred_time = pred_time
        self.p_fraud = p_fraud
        self.thresholds = thresholds
        self.seq = seq
        self.y: int | None = None
        self.label_time = 0.0


class DelayBuffer:
    """Store predictions until their labels arrive; release in arrival order.

    Labels may arrive in any order and long after later predictions. Internally
    two heaps (label arrival, maturity deadline) plus a lazily-cleaned heap of
    prediction times keep every operation ``O(log n)`` amortised.

    A prediction made at time ``t`` *matures* at ``t + maturity``. If its label
    has not arrived by then it is released with label ``unlabeled_at_maturity``,
    or dropped when that is ``None``. A label whose ``label_time`` is strictly
    after the deadline is *late*: the item still matures, and the late label is
    accepted and ignored (so outcomes do not depend on when ``pop_ready`` is
    called). A label arriving exactly at the deadline counts as on time.

    Parameters
    ----------
    maturity : float or None
        Maximum wait, in the units of the timestamps (e.g. days). ``None``
        disables maturity: items wait for their label indefinitely.
    unlabeled_at_maturity : int or None, default 0
        Label assigned to items that mature unlabeled; ``None`` drops them.

    Notes
    -----
    Imputing ``0`` at maturity biases the update: frauds that are not
    discovered within the window (undiscovered fraud) are counted as legit.
    This inflates apparent coverage of the legit class, under-represents fraud
    in the fraud-class quantile, and the larger the label delay relative to
    ``maturity``, the stronger the effect. Dropping instead removes those items
    from both classes, which biases the other way when late labels are
    disproportionately fraud (chargebacks). Neither is unbiased; choose
    ``maturity`` beyond the bulk of the delay distribution.

    "Pending" means registered and not yet released by :meth:`pop_ready`.
    """

    def __init__(self, maturity: float | None, unlabeled_at_maturity: int | None = 0) -> None:
        if maturity is not None and maturity < 0:
            raise ValueError("maturity must be non-negative or None")
        if unlabeled_at_maturity not in (None, 0, 1):
            raise ValueError("unlabeled_at_maturity must be 0, 1 or None")
        self._maturity = maturity
        self._fill = unlabeled_at_maturity
        self._seq = 0
        self._known: set[Hashable] = set()
        self._labeled: set[Hashable] = set()
        self._entries: dict[Hashable, _Entry] = {}
        self._label_heap: list[tuple[float, int, Hashable]] = []
        self._mature_heap: list[tuple[float, int, Hashable]] = []
        self._age_heap: list[tuple[float, int, Hashable]] = []

    @property
    def pending_count(self) -> int:
        """Number of registered predictions not yet released."""
        return len(self._entries)

    def register(
        self,
        tx_id: Hashable,
        timestamp: float,
        p_fraud: float,
        thresholds_used: Thresholds,
    ) -> None:
        """Register a prediction made at ``timestamp``.

        Raises
        ------
        ValueError
            If ``tx_id`` was already registered.
        """
        if tx_id in self._known:
            raise ValueError(f"duplicate tx_id: {tx_id!r}")
        self._known.add(tx_id)
        seq = self._seq
        self._seq += 1
        self._entries[tx_id] = _Entry(timestamp, p_fraud, thresholds_used, seq)
        heapq.heappush(self._age_heap, (timestamp, seq, tx_id))
        if self._maturity is not None:
            heapq.heappush(self._mature_heap, (timestamp + self._maturity, seq, tx_id))

    def add_label(self, tx_id: Hashable, y: int, label_time: float) -> None:
        """Record that the label ``y`` for ``tx_id`` arrived at ``label_time``.

        Raises
        ------
        ValueError
            If ``tx_id`` is unknown, already labeled, or ``label_time``
            precedes the prediction time.
        """
        if tx_id not in self._known:
            raise ValueError(f"unknown tx_id: {tx_id!r}")
        if tx_id in self._labeled:
            raise ValueError(f"duplicate label for tx_id: {tx_id!r}")
        if y not in (0, 1):
            raise ValueError("y must be 0 or 1")
        self._labeled.add(tx_id)
        entry = self._entries.get(tx_id)
        if entry is None:
            return  # already released at maturity; late label ignored
        if label_time < entry.pred_time:
            self._labeled.discard(tx_id)
            raise ValueError("label_time precedes the prediction time")
        if self._maturity is not None and label_time > entry.pred_time + self._maturity:
            return  # late label: the item will mature unlabeled
        entry.y = y
        entry.label_time = label_time
        heapq.heappush(self._label_heap, (label_time, entry.seq, tx_id))

    def pop_ready(self, now: float) -> list[Released]:
        """Release every item labeled or matured at or before ``now``.

        Returns
        -------
        list of Released
            Sorted by release time (label arrival, or maturity deadline),
            ties broken by registration order. Matured items are omitted when
            ``unlabeled_at_maturity`` is ``None``.
        """
        ready: list[tuple[float, int, Released | None]] = []
        entries = self._entries
        lh = self._label_heap
        while lh and lh[0][0] <= now:
            t, seq, tx_id = heapq.heappop(lh)
            e = entries.pop(tx_id)
            assert e.y is not None
            ready.append(
                (t, seq, Released(tx_id, e.p_fraud, e.y, e.thresholds, e.pred_time, t, False))
            )
        mh = self._mature_heap
        while mh and mh[0][0] <= now:
            t, seq, tx_id = heapq.heappop(mh)
            e2 = entries.get(tx_id)
            if e2 is None or e2.y is not None:
                continue  # already released with its label (stale heap entry)
            del entries[tx_id]
            if self._fill is None:
                continue
            ready.append(
                (
                    t,
                    seq,
                    Released(tx_id, e2.p_fraud, self._fill, e2.thresholds, e2.pred_time, t, True),
                )
            )
        ready.sort(key=lambda r: (r[0], r[1]))
        return [r[2] for r in ready if r[2] is not None]

    def oldest_pending_age(self, now: float) -> float | None:
        """Age (``now`` minus prediction time) of the oldest pending item.

        Returns ``None`` if nothing is pending.
        """
        heap = self._age_heap
        while heap and heap[0][2] not in self._entries:
            heapq.heappop(heap)
        if not heap:
            return None
        return now - heap[0][0]
