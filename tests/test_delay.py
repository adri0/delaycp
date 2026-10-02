import time

import pytest

from delaycp.delay import DelayBuffer
from delaycp.types import Thresholds

TH = Thresholds(q_0=0.5, q_1=0.5)


def test_out_of_order_arrival_sorted_by_label_time() -> None:
    buf = DelayBuffer(maturity=None)
    for i in range(3):
        buf.register(i, float(i), 0.1 * i, TH)
    buf.add_label(0, 1, 50.0)
    buf.add_label(2, 0, 10.0)
    buf.add_label(1, 1, 30.0)
    assert buf.pop_ready(5.0) == []
    out = buf.pop_ready(40.0)
    assert [r.tx_id for r in out] == [2, 1]
    assert [r.label_time for r in out] == [10.0, 30.0]
    assert [r.y for r in out] == [0, 1]
    assert out[0].pred_time == 2.0 and out[0].thresholds_used is TH
    assert not any(r.matured for r in out)
    assert buf.pending_count == 1
    assert [r.tx_id for r in buf.pop_ready(50.0)] == [0]
    assert buf.pending_count == 0


def test_maturity_release_imputes_label() -> None:
    buf = DelayBuffer(maturity=10.0)
    buf.register("a", 0.0, 0.9, TH)
    buf.register("b", 1.0, 0.2, TH)
    buf.add_label("b", 1, 3.0)
    assert [r.tx_id for r in buf.pop_ready(9.9)] == ["b"]
    out = buf.pop_ready(10.0)
    assert len(out) == 1
    r = out[0]
    assert (r.tx_id, r.y, r.matured, r.label_time) == ("a", 0, True, 10.0)


def test_mixed_release_sorted_by_release_time() -> None:
    buf = DelayBuffer(maturity=10.0, unlabeled_at_maturity=1)
    buf.register("a", 0.0, 0.5, TH)  # matures at 10
    buf.register("b", 1.0, 0.5, TH)  # labeled at 5
    buf.register("c", 2.0, 0.5, TH)  # labeled at 12 == deadline: on time
    buf.add_label("b", 0, 5.0)
    buf.add_label("c", 0, 12.0)
    out = buf.pop_ready(100.0)
    assert [r.tx_id for r in out] == ["b", "a", "c"]
    assert [r.matured for r in out] == [False, True, False]
    assert out[1].y == 1


def test_late_label_is_ignored_regardless_of_pop_timing() -> None:
    for pop_first in (False, True):
        buf = DelayBuffer(maturity=10.0)
        buf.register("a", 0.0, 0.5, TH)
        if pop_first:
            assert [r.matured for r in buf.pop_ready(11.0)] == [True]
        buf.add_label("a", 1, 15.0)
        out = buf.pop_ready(20.0)
        assert [r.matured for r in out] == ([] if pop_first else [True])
        assert buf.pending_count == 0


def test_drop_mode() -> None:
    buf = DelayBuffer(maturity=5.0, unlabeled_at_maturity=None)
    buf.register("a", 0.0, 0.5, TH)
    buf.register("b", 0.0, 0.5, TH)
    buf.add_label("b", 1, 2.0)
    out = buf.pop_ready(100.0)
    assert [r.tx_id for r in out] == ["b"]
    assert buf.pending_count == 0


def test_duplicate_and_unknown_ids() -> None:
    buf = DelayBuffer(maturity=None)
    buf.register(1, 0.0, 0.5, TH)
    with pytest.raises(ValueError, match="duplicate tx_id"):
        buf.register(1, 1.0, 0.5, TH)
    with pytest.raises(ValueError, match="unknown"):
        buf.add_label(2, 0, 1.0)
    buf.add_label(1, 0, 1.0)
    with pytest.raises(ValueError, match="duplicate label"):
        buf.add_label(1, 1, 2.0)
    buf.pop_ready(5.0)
    with pytest.raises(ValueError, match="duplicate"):
        buf.register(1, 9.0, 0.5, TH)
    with pytest.raises(ValueError, match="duplicate label"):
        buf.add_label(1, 1, 9.0)


def test_label_before_prediction_rejected_and_retryable() -> None:
    buf = DelayBuffer(maturity=None)
    buf.register(1, 5.0, 0.5, TH)
    with pytest.raises(ValueError, match="precedes"):
        buf.add_label(1, 0, 4.0)
    buf.add_label(1, 0, 6.0)


def test_oldest_pending_age_and_count() -> None:
    buf = DelayBuffer(maturity=None)
    assert buf.oldest_pending_age(10.0) is None
    buf.register("late", 5.0, 0.5, TH)
    buf.register("early", 2.0, 0.5, TH)
    assert buf.pending_count == 2
    assert buf.oldest_pending_age(10.0) == 8.0
    buf.add_label("early", 0, 6.0)
    buf.pop_ready(6.0)
    assert buf.oldest_pending_age(10.0) == 5.0


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        DelayBuffer(maturity=-1.0)
    with pytest.raises(ValueError):
        DelayBuffer(maturity=1.0, unlabeled_at_maturity=2)


def test_performance_one_million() -> None:
    n = 1_000_000
    buf = DelayBuffer(maturity=500.0)
    start = time.perf_counter()
    for i in range(n):
        buf.register(i, float(i), 0.1, TH)
        if i % 3 == 0:
            buf.add_label(i, 0, float(i) + (i * 7919) % 400)
        if i % 1000 == 999:
            buf.pop_ready(float(i))
    buf.pop_ready(float(n) + 1000.0)
    elapsed = time.perf_counter() - start
    assert buf.pending_count == 0
    assert elapsed < 5.0, elapsed
