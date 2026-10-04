"""Run every validation scenario and write a markdown report.

Usage, from the repository root::

    uv run python -m benchmarks.scenario_report              # print to stdout
    uv run python -m benchmarks.scenario_report -o report.md

The scenarios and the reasoning behind them are in
:mod:`benchmarks.scenarios`; ``tests/test_scenarios.py`` asserts on the same
numbers. Takes about half a minute.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from benchmarks import scenarios as sc

DRIFT_LABELS: dict[sc.DriftKind, str] = {
    "abrupt_prevalence_shift": "Prevalence ×5",
    "score_drift": "Fraud scores drift down",
}


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _f(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _pm(x: float) -> str:
    return f"±{x:.3f}"


def section_stationary() -> str:
    r = sc.scenario_stationary()
    rows = [
        ["delaycp (online)", _f(r["online_fraud_cov"]), _f(r["online_legit_cov"]),
         _pct(r["online_review"])],
        ["split conformal (static)", _f(r["split_fraud_cov"]), _f(r["split_legit_cov"]),
         _pct(r["split_review"])],
    ]  # fmt: skip
    return (
        "### 1. Stationary, no delay\n\n"
        f"200,000 transactions, {int(r['n_fraud'])} frauds; both methods calibrated on "
        f"the same {sc.N_CAL:,} examples ({int(r['n_cal_fraud'])} frauds).\n\n"
        + _table(["Method", "Fraud coverage", "Legit coverage", "Review rate"], rows)
    )


def section_fixed_delay() -> str:
    rows = []
    for tau in (1, 10, 100, 1000):
        r = sc.scenario_fixed_delay(tau)
        rows.append(
            [
                str(tau),
                _f(r["fraud_cov"]),
                _pm(r["bound_fraud"]),
                _f(r["fraud_rolling_mae"]),
                _f(r["fraud_worst_window"]),
                _f(r["legit_cov"]),
                _f(r["legit_rolling_mae"]),
                _f(r["legit_worst_window"]),
                str(int(r["in_flight_legit"])),
            ]
        )
    header = [
        "τ (steps)",
        "Fraud cov.",
        "Guarantee",
        "Fraud rolling error",
        "Worst fraud window",
        "Legit cov.",
        "Legit rolling error",
        "Worst legit window",
        "Legit labels in flight",
    ]
    return (
        "### 2. Fixed label delay\n\n"
        "One transaction per step, every label τ steps late. *Guarantee* is the "
        "a-priori bound on |fraud coverage − 0.90|. Rolling error is the mean "
        f"|coverage − target| over windows of {sc.FRAUD_WINDOW} frauds or "
        f"{sc.LEGIT_WINDOW:,} legits (independent outcomes would give about 0.017 and 0.004).\n\n"
        + _table(header, rows)
        + "\n\nLong-run coverage holds for every τ. Rolling error barely moves for fraud, "
        "but for legit it jumps at τ = 1000. Both trackers share `lr`, and legit "
        "labels outnumber fraud labels about 49 to 1, so roughly 1,000 legit "
        "updates are in flight at once. That makes the delayed tracker oscillate. "
        "Coverage still averages out, but individual windows can drop to about 0.5."
    )


def section_maturity() -> str:
    rows = []
    results = [sc.scenario_maturity(m) for m in (14.0, 30.0, 60.0, 90.0, 180.0)]
    shortest = results[0]
    for r in results:
        rows.append(
            [
                str(int(r["maturity"])),
                _pct(r["unreported_fraud_share"]),
                _pct(r["contamination"]),
                _f(r["fraud_cov"]),
                _f(r["legit_cov"]),
                _f(r["apparent_legit_cov"]),
                _f(r["predicted_legit_cov"]),
                _pct(r["review"]),
            ]
        )
    header = [
        "Maturity (days)",
        "Frauds unreported in window",
        "Legit class contamination",
        "True fraud cov.",
        "True legit cov.",
        "Measured legit cov.",
        "Predicted true legit cov.",
        "Review rate",
    ]
    return (
        "### 3. Chargeback delays and maturity window\n\n"
        f"Fraud labels arrive after a lognormal delay (median {sc.CHARGEBACK_MEDIAN:.0f} "
        f"days, σ = {sc.CHARGEBACK_SIGMA}; {sc.P_NEVER:.0%} never reported). Legit labels "
        "are confirmed when the maturity window closes. Frauds not reported by then "
        "are imputed as legit, both in calibration and online.\n\n"
        + _table(header, rows)
        + "\n\nFraud coverage is unbiased: late frauds leave the fraud class "
        "independently of their score. The overshoot at short windows is calibration "
        "noise, because only the frauds reported in time are available to calibrate "
        f"({int(shortest['n_cal_fraud_seen'])} at {int(shortest['maturity'])} days). "
        "Undiscovered frauds do pollute the legit class, and because nearly all "
        "of them score above `t_high`, the legit threshold rises until true legit coverage is "
        "≈ 1 − (α₀ − c·m_F)/(1 − c). Here c is the contamination and "
        "m_F = P(p > t_high | fraud). The measured legit coverage still reads "
        "≈ 0.95, so the bias is invisible in production. "
        f"This scenario uses `lr = {sc.LR_SLOW:g}`, because with tens of "
        "thousands of legit labels in flight the legit tracker is unstable at the "
        "default `lr` (see section 2)."
    )


def section_drift() -> str:
    rows = []
    for kind, label in DRIFT_LABELS.items():
        r = sc.scenario_drift(kind)
        rows.append(
            [
                label,
                f"{_f(r['online_fraud_cov_pre'])} / {_f(r['static_fraud_cov_pre'])}",
                f"{_f(r['online_fraud_cov_post'])} / {_f(r['static_fraud_cov_post'])}",
                f"{_f(r['online_legit_cov_post'])} / {_f(r['static_legit_cov_post'])}",
                f"{_pct(r['online_review_post'])} / {_pct(r['static_review_post'])}",
            ]
        )
    header = [
        "Change at day 50",
        "Fraud cov. before",
        "Fraud cov. after",
        "Legit cov. after",
        "Review after",
    ]
    return (
        "### 4. Prevalence shift and score drift\n\n"
        f"Lognormal label delays (median {sc.DELAY_MEDIAN:.0f} days, 100-day horizon). "
        "Cells are *delaycp / static split conformal*. *After* means from day 60 on.\n\n"
        + _table(header, rows)
        + "\n\nA prevalence shift does not change either class's score distribution, "
        "so per-class coverage holds for both methods. When fraud scores drift, the "
        "static thresholds lose most fraud coverage while delaycp recovers it. "
        "The legit columns show the delay effect from section 2: thousands of legit "
        "labels are in flight, so `t_high` oscillates. That costs some legit "
        "coverage after the change and inflates the review rate."
    )


def section_tiny() -> str:
    rows = []
    for k in (0, 10, 30):
        r = sc.scenario_tiny_fraud(k)
        rows.append(
            [
                str(k),
                "yes" if r["warned"] else "no",
                _f(r["split_fraud_cov"]),
                _f(r["fraud_cov_first_10pct"]),
                _f(r["fraud_cov"]),
                _pm(r["bound_fraud"]),
                _f(r["legit_cov"]),
                _pct(r["review"]),
            ]
        )
    header = [
        "Calibration frauds",
        "Warning",
        "Split fraud cov.",
        "Online fraud cov. (first 10%)",
        "Online fraud cov.",
        "Guarantee",
        "Legit cov.",
        "Review rate",
    ]
    return (
        "### 5. Tiny fraud class\n\n"
        "Calibration has fewer than 50 frauds (and 5,000 legits); 100,000-transaction "
        "stream.\n\n"
        + _table(header, rows)
        + "\n\nWith 10 calibration frauds the split-conformal threshold is far off. "
        "The online tracker pulls it back towards 0.90 as fraud labels arrive. With no "
        "frauds at all it starts from `q = 1 − α` (conservative)."
    )


def section_budget() -> str:
    rows = []
    for kind, label in DRIFT_LABELS.items():
        r = sc.scenario_budget(kind)
        rows.append(
            [
                label,
                _pct(r["review"]),
                _pct(r["review_post"]),
                _pct(r["max_rolling_dev"]),
                f"{_f(r['fraud_cov_pre'])} → {_f(r['fraud_cov_post'])}",
                _f(r["estimated_fraud_cov_last_1000"]),
            ]
        )
    header = [
        "Change at day 50",
        "Review rate",
        "Review after",
        "Max deviation (10k-tx windows)",
        "Fraud cov. before → after",
        "Estimated fraud cov. (last 1,000 labels)",
    ]
    return (
        "### 6. Budget mode under drift\n\n"
        f"Review budget {_pct(sc.REVIEW_BUDGET)}, batches of {sc.BUDGET_BATCH}.\n\n"
        + _table(header, rows)
        + "\n\nThe review rate stays on budget through both changes. Fraud coverage "
        "is an output in this mode: score drift lowers it, and the label-based "
        "estimate shows the drop one label delay later."
    )


def section_adversarial() -> str:
    rows = []
    for frac in (0.05, 0.5):
        r = sc.scenario_adversarial(frac)
        rows.append(
            [
                _pct(frac),
                _f(r["fraud_cov"]),
                _pm(r["bound_fraud"]),
                _f(r["fraud_worst_window"]),
                _f(r["mean_t_low_late"]),
                _pct(r["review"]),
            ]
        )
    header = [
        "Adversarial frauds",
        "Fraud cov.",
        "Guarantee",
        "Worst fraud window",
        "Mean t_low (2nd half)",
        "Review rate",
    ]
    return (
        "### 7. Adversarial\n\n"
        "A share of frauds is placed just below the live `t_low` (the adversary "
        "sees the current thresholds). Labels arrive with lognormal delays.\n\n"
        + _table(header, rows)
        + "\n\nThe long-run guarantee holds in both cases. When the adversarial share "
        "is below α the band stays reasonable. Above α the only way to cover those "
        "frauds is `t_low ≈ 0`, so most traffic goes to review."
    )


def report() -> str:
    """The full markdown report."""
    intro = (
        "## Validation scenarios\n\n"
        f"Targets: fraud coverage {sc.TARGET[1]:.2f} (α = {sc.ALPHA_FRAUD}), legit "
        f"coverage {sc.TARGET[0]:.2f} (α = {sc.ALPHA_LEGIT}), `lr = {sc.LR}` unless noted. "
        f"Synthetic streams with {_pct(sc.FRAUD_RATE)} fraud from `delaycp.simulate`, "
        "fixed seeds. Regenerate with `python -m benchmarks.scenario_report`."
    )
    sections = [
        section_stationary,
        section_fixed_delay,
        section_maturity,
        section_drift,
        section_tiny,
        section_budget,
        section_adversarial,
    ]
    return "\n\n".join([intro, *(s() for s in sections)]) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("-o", "--output", help="write the report here instead of stdout")
    args = parser.parse_args(argv)
    text = report()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
