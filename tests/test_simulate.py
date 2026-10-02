import numpy as np
import pytest

from delaycp.metrics import class_coverage, fraud_auto_approved_rate
from delaycp.mondrian import actions_from_thresholds
from delaycp.simulate import (
    AdversarialScores,
    ReplayResult,
    inject_delays,
    make_stream,
    replay,
)
from delaycp.stream import OnlineConformalClassifier
from delaycp.types import Action, Thresholds

N = 40_000


def _clf(**kw: object) -> OnlineConformalClassifier:
    return OnlineConformalClassifier(0.1, 0.05, lr=0.01, **kw)  # type: ignore[arg-type]


# --- inject_delays -----------------------------------------------------------


def test_fixed_delay_is_exact() -> None:
    t = np.arange(5.0)
    lt = inject_delays(t, np.array([0, 1, 0, 1, 0]), "fixed", np.random.default_rng(0), delay=3.0)
    np.testing.assert_array_equal(lt, t + 3.0)


@pytest.mark.parametrize("median", [2.0, 30.0, 90.0])
def test_lognormal_median(median: float) -> None:
    rng = np.random.default_rng(1)
    t = np.zeros(N)
    lt = inject_delays(t, np.ones(N, dtype=int), "lognormal", rng, median=median, sigma=0.8)
    assert np.median(lt) == pytest.approx(median, rel=0.03)
    assert np.all(lt >= 0)


def test_mixture_component_medians() -> None:
    rng = np.random.default_rng(2)
    t = np.zeros(N)
    p_fast = 0.8
    lt = inject_delays(
        t, np.ones(N, dtype=int), "mixture", rng,
        p_fast=p_fast, fast_median=3.0, slow_median=90.0, sigma=0.4,
    )  # fmt: skip
    # components are well separated: fast ones fill the lower p_fast mass
    assert np.quantile(lt, p_fast / 2) == pytest.approx(3.0, rel=0.05)
    assert np.quantile(lt, p_fast + (1 - p_fast) / 2) == pytest.approx(90.0, rel=0.05)
    assert np.mean(lt < 20.0) == pytest.approx(p_fast, abs=0.01)


def test_never_reported_only_fraud() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(N) < 0.5).astype(int)
    lt = inject_delays(np.zeros(N), y, "fixed", rng, delay=1.0, p_never=0.3)
    assert not np.any(np.isinf(lt[y == 0]))
    assert np.mean(np.isinf(lt[y == 1])) == pytest.approx(0.3, abs=0.02)


def test_legit_delay_override() -> None:
    y = np.array([0, 1, 0])
    lt = inject_delays(
        np.zeros(3), y, "fixed", np.random.default_rng(0), delay=5.0, legit_delay=60
    )
    np.testing.assert_array_equal(lt, [60.0, 5.0, 60.0])


def test_inject_delays_rejects_bad_params() -> None:
    rng = np.random.default_rng(0)
    t, y = np.zeros(3), np.zeros(3, dtype=int)
    with pytest.raises(TypeError, match="requires"):
        inject_delays(t, y, "lognormal", rng, median=1.0)
    with pytest.raises(TypeError, match="unexpected"):
        inject_delays(t, y, "fixed", rng, delay=1.0, median=2.0)
    with pytest.raises(ValueError, match="unknown dist"):
        inject_delays(t, y, "weibull", rng)  # type: ignore[arg-type]


# --- make_stream ---------------------------------------------------------------


def test_stationary_rate_constant() -> None:
    s = make_stream(N, 0.05, np.random.default_rng(4))
    assert np.all(np.diff(s.timestamps) >= 0)
    half = N // 2
    assert np.mean(s.y[:half]) == pytest.approx(0.05, abs=0.006)
    assert np.mean(s.y[half:]) == pytest.approx(0.05, abs=0.006)
    assert isinstance(s.p_fraud, np.ndarray)
    assert s.p_fraud[s.y == 1].mean() > s.p_fraud[s.y == 0].mean() + 0.4


def test_abrupt_prevalence_shift() -> None:
    s = make_stream(N, 0.02, np.random.default_rng(5), "abrupt_prevalence_shift", shift_factor=5)
    after = s.timestamps >= 50.0
    assert np.mean(s.y[~after]) == pytest.approx(0.02, abs=0.004)
    assert np.mean(s.y[after]) == pytest.approx(0.10, abs=0.01)


def test_score_drift_lowers_new_fraud_scores() -> None:
    s = make_stream(N, 0.1, np.random.default_rng(6), "score_drift")
    assert isinstance(s.p_fraud, np.ndarray)
    after = s.timestamps >= 50.0
    f_before = s.p_fraud[(s.y == 1) & ~after].mean()
    f_after = s.p_fraud[(s.y == 1) & after].mean()
    assert f_after < f_before - 0.25
    # prevalence and legit scores are unchanged
    assert np.mean(s.y[after]) == pytest.approx(np.mean(s.y[~after]), abs=0.01)
    l_before = s.p_fraud[(s.y == 0) & ~after].mean()
    l_after = s.p_fraud[(s.y == 0) & after].mean()
    assert l_after == pytest.approx(l_before, abs=0.01)


def test_seasonal_spike() -> None:
    s = make_stream(N, 0.02, np.random.default_rng(7), "seasonal_spike", shift_factor=4)
    inside = (s.timestamps >= 60.0) & (s.timestamps < 70.0)
    assert np.mean(s.y[inside]) == pytest.approx(0.08, abs=0.015)
    assert np.mean(s.y[~inside]) == pytest.approx(0.02, abs=0.004)


def test_adversarial_needs_callback() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="thresholds_fn"):
        make_stream(10, 0.1, rng, "adversarial")
    with pytest.raises(ValueError, match="thresholds_fn"):
        make_stream(10, 0.1, rng, "stationary", thresholds_fn=lambda: Thresholds(0.5, 0.5))


def test_adversarial_scores_track_current_t_low() -> None:
    th = [Thresholds(q_0=0.9, q_1=0.7)]  # t_low = 0.3
    s = make_stream(
        2_000, 0.2, np.random.default_rng(8), "adversarial",
        thresholds_fn=lambda: th[0], adversarial_fraction=1.0, margin=0.02,
    )  # fmt: skip
    assert isinstance(s.p_fraud, AdversarialScores)
    p1 = s.p_fraud(0, 1_000)
    th[0] = Thresholds(q_0=0.9, q_1=0.4)  # t_low = 0.6
    p2 = s.p_fraud(1_000, 2_000)
    f1, f2 = p1[s.y[:1_000] == 1], p2[s.y[1_000:] == 1]
    assert np.all((f1 >= 0.28) & (f1 < 0.3))
    assert np.all((f2 >= 0.58) & (f2 < 0.6))
    assert not np.any(np.isnan(s.p_fraud.values))
    # every adversarial fraud is auto-approved under the thresholds it saw
    assert np.all(actions_from_thresholds(f1, Thresholds(0.9, 0.7)) == Action.APPROVE)


# --- replay --------------------------------------------------------------------


def _run(seed: int, batch_size: int = 50) -> ReplayResult:
    rng = np.random.default_rng(seed)
    s = make_stream(5_000, 0.05, rng, "abrupt_prevalence_shift")
    lt = inject_delays(s.timestamps, s.y, "lognormal", rng, median=3.0, sigma=0.8, p_never=0.1)
    assert isinstance(s.p_fraud, np.ndarray)
    return replay(_clf(maturity=20.0), s.timestamps, s.p_fraud, s.y, lt, batch_size)


def test_replay_deterministic() -> None:
    a, b = _run(9), _run(9)
    for field in ("y", "p_fraud", "sets", "actions", "t_low", "t_high", "label_times"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    c = _run(10)
    assert not np.array_equal(a.actions, c.actions)


def test_replay_outputs_consistent() -> None:
    r = _run(11)
    n = r.y.size
    assert r.sets.shape == (n, 2)
    # actions agree with the recorded thresholds in effect at prediction time
    for i in range(0, n, 50):
        th = Thresholds(q_0=r.t_high[i], q_1=1.0 - r.t_low[i])
        np.testing.assert_array_equal(
            r.actions[i : i + 50], actions_from_thresholds(r.p_fraud[i : i + 50], th)
        )
    # thresholds actually moved, and only between batches
    assert np.unique(r.t_low).size > 1
    assert np.all(r.t_low.reshape(-1, 50) == r.t_low.reshape(-1, 50)[:, :1])
    cov = class_coverage(r.y, r.sets)
    assert 0.0 <= cov[1].coverage <= 1.0


def test_replay_does_not_leak_future_labels() -> None:
    # labels only arrive after the stream ends: thresholds must never move
    rng = np.random.default_rng(12)
    s = make_stream(1_000, 0.1, rng)
    lt = inject_delays(s.timestamps, s.y, "fixed", rng, delay=1_000.0)
    assert isinstance(s.p_fraud, np.ndarray)
    r = replay(_clf(), s.timestamps, s.p_fraud, s.y, lt, batch_size=10)
    assert np.unique(r.t_low).size == 1
    assert np.unique(r.t_high).size == 1


def test_replay_adversarial_hides_below_live_t_low() -> None:
    def run(scenario: str) -> ReplayResult:
        rng = np.random.default_rng(13)
        clf = _clf()
        s = make_stream(
            10_000, 0.05, rng, scenario,  # type: ignore[arg-type]
            thresholds_fn=(lambda: clf.thresholds) if scenario == "adversarial" else None,
        )  # fmt: skip
        lt = inject_delays(s.timestamps, s.y, "fixed", rng, delay=1.0)
        return replay(clf, s.timestamps, s.p_fraud, s.y, lt, batch_size=20)

    adv, base = run("adversarial"), run("stationary")
    # where there is room below t_low, the adversarial half of frauds sit just under it
    room = (adv.y == 1) & (adv.t_low > 0.02)
    below = (adv.p_fraud >= adv.t_low - 0.02) & (adv.p_fraud < adv.t_low)
    assert np.mean(below[room]) == pytest.approx(0.5, abs=0.08)
    assert fraud_auto_approved_rate(adv.y, adv.actions) > (
        fraud_auto_approved_rate(base.y, base.actions) + 0.05
    )
    # the conformal controller reacts by lowering t_low
    assert adv.t_low[-1] < base.t_low[-1]


def test_replay_validates_inputs() -> None:
    t = np.arange(3.0)
    with pytest.raises(ValueError, match="precede"):
        replay(_clf(), t, np.zeros(3), np.zeros(3), t - 1.0)
    with pytest.raises(ValueError, match="batch_size"):
        replay(_clf(), t, np.zeros(3), np.zeros(3), t, batch_size=0)
