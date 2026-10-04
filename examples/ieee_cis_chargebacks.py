"""delaycp on the IEEE-CIS Fraud Detection data with simulated chargeback delays.

Download ``train_transaction.csv`` and ``train_identity.csv`` from
https://www.kaggle.com/c/ieee-fraud-detection/data, then from the repository root::

    uv sync --extra examples
    uv run python examples/ieee_cis_chargebacks.py path/to/ieee-cis

The ~182 days of labelled data are split in time: the first 121 days train a
LightGBM model, the next 30 calibrate, and the last ~31 are streamed one batch
at a time. Test-month labels arrive late, as chargebacks do (lognormal delay,
median 30 days, 10% of frauds never reported; legit is confirmed when the
60-day maturity window closes without a chargeback). Calibration uses the
calibration month's final labels, i.e. it assumes that month has matured.

Five ways of turning the model's ``p_fraud`` into APPROVE / REVIEW / DECLINE
are compared: a hand-picked score band, static split conformal (one threshold
for both classes), static Mondrian split conformal (one per class), and
delaycp's online classifier in coverage mode and in budget mode. Takes a few
minutes and ~3 GB of memory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from delaycp import metrics
from delaycp.mondrian import MondrianPID, actions_from_thresholds, sets_from_thresholds
from delaycp.simulate import inject_delays, replay
from delaycp.stream import OnlineConformalClassifier
from delaycp.types import Action, Thresholds

TRAIN_DAYS, CAL_DAYS = 121.0, 30.0  # test = everything after TRAIN_DAYS + CAL_DAYS
ALPHA_FRAUD, ALPHA_LEGIT = 0.10, 0.05  # target miscoverage per class
ALPHA_MARGINAL = 0.05  # single-threshold split conformal
FIXED_BAND = (0.10, 0.50)  # hand-picked: approve below, decline above, review between
CHARGEBACK_MEDIAN, CHARGEBACK_SIGMA, P_NEVER = 30.0, 0.6, 0.10
MATURITY = 60.0  # days without a chargeback after which a transaction counts as legit
# PID step size, in score units. Calibrated t_low is only ~0.01 here (many frauds
# score near 0), so a larger step lets a burst of missed frauds push t_low below 0,
# after which every transaction goes to REVIEW.
LR = 1e-3
REVIEW_BUDGET, BUDGET_LR = 0.005, 0.01
BATCH = 100  # transactions predicted with the same thresholds
FRAUD_WINDOW = 300  # frauds per rolling-coverage window

# Features: the raw transaction columns minus the 339 engineered V* columns,
# plus device type/info from the identity table. Categoricals are passed to
# LightGBM as pandas categories; missing values are left as NaN.
CATEGORICAL = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    *[f"M{i}" for i in range(1, 10)], "DeviceType", "DeviceInfo",
]  # fmt: skip
NUMERIC = [
    "TransactionAmt", *[f"card{i}" for i in (1, 2, 3, 5)], "addr1", "addr2", "dist1", "dist2",
    *[f"C{i}" for i in range(1, 15)], *[f"D{i}" for i in range(1, 16)],
]  # fmt: skip
FEATURES = NUMERIC + CATEGORICAL


@dataclass
class Data:
    """One time slice: features, labels, ``p_fraud`` (once scored) and time in days."""

    X: pd.DataFrame
    y: np.ndarray
    day: np.ndarray
    p: np.ndarray | None = None


def load(data_dir: Path) -> pd.DataFrame:
    """Join transactions with identity, sort by time and add ``day`` (days since start)."""
    tx_cols = ["TransactionID", "TransactionDT", "isFraud", *NUMERIC, *CATEGORICAL[:-2]]
    tx = pd.read_csv(data_dir / "train_transaction.csv", usecols=tx_cols)
    idt = pd.read_csv(
        data_dir / "train_identity.csv", usecols=["TransactionID", "DeviceType", "DeviceInfo"]
    )
    df = tx.merge(idt, on="TransactionID", how="left").sort_values("TransactionDT")
    df["day"] = (df["TransactionDT"] - df["TransactionDT"].min()) / 86_400.0
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame) -> tuple[Data, Data, Data]:
    """Temporal train / calibration / test split."""
    edges = [-np.inf, TRAIN_DAYS, TRAIN_DAYS + CAL_DAYS, np.inf]
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        d = df[(df["day"] >= lo) & (df["day"] < hi)]
        parts.append(Data(d[FEATURES], d["isFraud"].to_numpy(int), d["day"].to_numpy(float)))
    return parts[0], parts[1], parts[2]


def train(tr: Data, seed: int) -> lgb.Booster:
    """Plain LightGBM, no tuning or class reweighting."""
    params = {
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 64, "bagging_fraction": 0.8,
        "bagging_freq": 1, "feature_fraction": 0.8, "seed": seed, "verbose": -1,
    }  # fmt: skip
    return lgb.train(params, lgb.Dataset(tr.X, tr.y), num_boost_round=400)


@dataclass
class Result:
    """Per-transaction decisions of one method on the test stream."""

    sets: np.ndarray
    actions: np.ndarray
    t_low: np.ndarray
    t_high: np.ndarray


def static(p: np.ndarray, th: Thresholds) -> Result:
    """Decisions with thresholds frozen for the whole test period."""
    n = p.size
    return Result(
        sets_from_thresholds(p, th),
        actions_from_thresholds(p, th),
        np.full(n, th.t_low),
        np.full(n, th.t_high),
    )


def split_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile: the ``ceil((n + 1)(1 - alpha))``-th smallest score."""
    s = np.sort(scores)
    return float(s[min(int(np.ceil((s.size + 1) * (1 - alpha))), s.size) - 1])


def run_methods(cal: Data, test: Data, label_times: np.ndarray) -> dict[str, Result]:
    """Run the five methods on the test stream."""
    assert cal.p is not None and test.p is not None
    out: dict[str, Result] = {}
    lo, hi = FIXED_BAND
    out["Fixed band"] = static(test.p, Thresholds(q_0=hi, q_1=1 - lo))

    true_score = np.where(cal.y == 1, 1 - cal.p, cal.p)  # s(x, y) = 1 - p_hat(y | x)
    q = split_quantile(true_score, ALPHA_MARGINAL)
    out["Split conformal"] = static(test.p, Thresholds(q_0=q, q_1=q))

    mondrian = MondrianPID(ALPHA_FRAUD, ALPHA_LEGIT, lr=LR)
    mondrian.fit_calibration(cal.p, cal.y)
    out["Mondrian split"] = static(test.p, mondrian.thresholds)

    for name, kw in [
        ("delaycp coverage", {}),
        ("delaycp budget", {"mode": "budget", "review_budget": REVIEW_BUDGET,
                            "budget_lr": BUDGET_LR}),
    ]:  # fmt: skip
        clf = OnlineConformalClassifier(ALPHA_FRAUD, ALPHA_LEGIT, lr=LR, maturity=MATURITY, **kw)
        clf.fit_calibration(cal.p, cal.y)
        r = replay(clf, test.day, test.p, test.y, label_times, batch_size=BATCH)
        out[name] = Result(r.sets, r.actions, r.t_low, r.t_high)
    return out


def report(test: Data, results: dict[str, Result], label_times: np.ndarray) -> None:
    """Print the headline table."""
    end = float(test.day[-1])
    header = (
        f"{'Method':<18}{'Fraud cov':>10}{'(seen)':>8}{'Legit cov':>10}{'Review':>8}"
        f"{'Decline':>9}{'Fraud appr.':>12}{'Worst win':>10}"
    )
    print(f"\nTargets: fraud coverage {1 - ALPHA_FRAUD:.2f}, legit {1 - ALPHA_LEGIT:.2f}; "
          f"budget mode review rate {REVIEW_BUDGET:.1%}")  # fmt: skip
    print(f"(seen) = fraud coverage measurable from labels arrived by day {end:.0f}; "
          f"worst window = lowest coverage over {FRAUD_WINDOW} consecutive frauds\n")  # fmt: skip
    print(header + "\n" + "-" * len(header))
    for name, r in results.items():
        s = metrics.summary(test.y, r.sets, r.actions, fraud_window=FRAUD_WINDOW)
        seen = metrics.coverage_measured_at(end, test.y, r.sets, label_times, cls=1)
        cc, ar = s["class_coverage"], s["action_rates"]
        print(
            f"{name:<18}{cc[1].coverage:>10.3f}{seen.coverage:>8.3f}{cc[0].coverage:>10.3f}"
            f"{ar.review:>8.2%}{ar.decline:>9.2%}{s['fraud_auto_approved_rate']:>12.3f}"
            f"{s['worst_window_fraud_coverage']:>10.3f}"
        )


def plot(test: Data, results: dict[str, Result]) -> Figure:
    """Rolling fraud coverage, daily review rate and the band edges over time."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fraud_day = test.day[test.y == 1][FRAUD_WINDOW - 1 :]
    day_idx = np.floor(test.day).astype(int)
    days, n_per_day = np.unique(day_idx, return_counts=True)
    for i, (name, r) in enumerate(results.items()):
        color = f"C{i}"
        rc = metrics.rolling_coverage(test.y, r.sets, FRAUD_WINDOW, cls=1)
        axes[0].plot(fraud_day, rc, color=color, label=name)
        review = np.bincount(day_idx, r.actions == Action.REVIEW)[days] / n_per_day
        axes[1].plot(days, review, color=color, marker=".", label=name)
        axes[2].plot(test.day, np.clip(r.t_low, 0, 1), color=color, label=f"{name} t_low")
        axes[2].plot(test.day, np.clip(r.t_high, 0, 1), color=color, ls="--")
    axes[0].axhline(1 - ALPHA_FRAUD, color="k", ls=":", label="target")
    axes[0].set_ylabel(f"fraud coverage\n(last {FRAUD_WINDOW} frauds)")
    axes[1].axhline(REVIEW_BUDGET, color="k", ls=":", label="budget")
    axes[1].set(ylabel="daily review rate", yscale="log")
    axes[2].set(ylabel="t_low (solid), t_high (dashed)", xlabel="day", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir", type=Path, help="folder with train_transaction.csv etc.")
    ap.add_argument("--plot", type=Path, default=Path("ieee_cis_chargebacks.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr, cal, test = split(load(args.data_dir))
    print(f"train {len(tr.y):,} ({tr.y.mean():.2%} fraud), cal {len(cal.y):,} "
          f"({cal.y.mean():.2%}), test {len(test.y):,} ({test.y.mean():.2%})")  # fmt: skip
    model = train(tr, args.seed)
    cal.p, test.p = (np.asarray(model.predict(d.X)) for d in (cal, test))
    rng = np.random.default_rng(args.seed)
    label_times = inject_delays(
        test.day, test.y, "lognormal", rng, median=CHARGEBACK_MEDIAN, sigma=CHARGEBACK_SIGMA,
        p_never=P_NEVER, legit_delay=MATURITY,
    )  # fmt: skip
    results = run_methods(cal, test, label_times)
    report(test, results, label_times)
    plot(test, results).savefig(args.plot, dpi=120)
    print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
