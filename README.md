# delaycp

[![CI](https://github.com/adri0/delaycp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adri0/delaycp/actions/workflows/ci.yml)
[![Docs](https://github.com/adri0/delaycp/actions/workflows/docs.yml/badge.svg?branch=main)](https://adri0.github.io/delaycp/)
[![Tested on Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/tested%20on-3.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://github.com/adri0/delaycp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/adri0/delaycp)](https://github.com/adri0/delaycp/blob/main/LICENSE)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange)](#limitations)
[![Dependencies: numpy](https://img.shields.io/badge/dependencies-numpy-013243?logo=numpy)](https://github.com/adri0/delaycp/blob/main/pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy (strict)](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Status: pre-alpha.** APIs are unstable and may change without notice.

`delaycp` wraps any binary classifier that outputs `p = P(fraud)` and turns each score into one of three decisions: **approve**, **review** or **decline**. It keeps per-class error rates on target while labels arrive late, fraud is rare and the data drifts. It uses online conformal prediction, needs only NumPy, and does not touch your model.

Documentation: **<https://adri0.github.io/delaycp/>**

## The problem

A fraud model gives a score. Turning that score into decisions in production runs into four problems at once:

- **Delayed labels.** You learn that a transaction was fraud weeks later, when the chargeback arrives. You learn that it was legit only when enough time has passed without one. Any method that adjusts from feedback works with old feedback.
- **Class imbalance.** Fraud is often well under 1% of traffic. An overall error target is met almost entirely by the legit majority and says nothing about how much fraud gets through.
- **Drift.** Fraudsters adapt, so the model starts scoring new fraud lower. Thresholds tuned on last quarter's data quietly stop catching it.
- **Review budgets.** Uncertain cases go to people, and their time is limited. A method that holds coverage by sending everything to review is not usable.

delaycp sets two thresholds, `t_low` and `t_high`. Scores below `t_low` are approved, scores above `t_high` are declined, and scores in between go to review. Each threshold targets one class's error rate (Mondrian conformal prediction). The thresholds update online from each label as it arrives, judged against the thresholds that were in effect when the prediction was made ([conformal PID control](https://arxiv.org/abs/2307.16895) with delay-aware updates). A budget mode holds the review rate at a fixed level instead. See [Concepts](https://adri0.github.io/delaycp/concepts/) for a plain-language walkthrough.

## Installation

v0.1.0 is published to TestPyPI only. NumPy comes from PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ delaycp
```

From source, with [uv](https://docs.astral.sh/uv/) or pip:

```bash
git clone https://github.com/adri0/delaycp.git && cd delaycp
uv sync --extra dev          # or: pip install -e ".[dev]"
```

## Quickstart

```python
import numpy as np
from delaycp import Action, OnlineConformalClassifier

rng = np.random.default_rng(0)  # stand-in for your model's scores on labelled history
y_cal = (rng.random(20_000) < 0.02).astype(int)
p_cal = np.where(y_cal == 1, rng.beta(5, 2, 20_000), rng.beta(1, 12, 20_000))

clf = OnlineConformalClassifier(alpha_fraud=0.10, alpha_legit=0.05, lr=0.01, maturity=60)
clf.fit_calibration(p_cal, y_cal)  # warm-start both thresholds

# Day 0: score a batch. Each prediction is stored with the thresholds it used.
actions = clf.predict(["a", "b", "c"], timestamps=[0.0, 0.0, 0.0], p_fraud=[0.01, 0.4, 0.97])
print([Action(a).name for a in actions])  # ['APPROVE', 'REVIEW', 'DECLINE']

# Day 35: a chargeback arrives for "b". Updates apply when time advances.
clf.add_labels(["b"], ys=[1], label_times=[35.0])
clf.advance(now=35.0)
print(clf.thresholds.t_low, clf.thresholds.t_high)  # thresholds for the next batch
```

Timestamps are in whatever unit you choose (days here). With `maturity=60`, a transaction with no label after 60 days is treated as legit. [`delaycp.simulate`](https://adri0.github.io/delaycp/reference/delaycp/simulate/) has synthetic streams, delay injection and a `replay` helper. [`delaycp.metrics`](https://adri0.github.io/delaycp/reference/delaycp/metrics/) has per-class, rolling and "as measured at time t" coverage.

## Coverage mode and budget mode

`t_high` always targets legit coverage: at most `alpha_legit` of legit transactions are declined. The two modes differ in what `t_low` targets.

- **Coverage mode** (default). `t_low` is a conformal threshold driven by fraud labels, so at most `alpha_fraud` of frauds are approved in the long run. The review queue is whatever that takes. When fraud drifts toward lower scores, `t_low` falls and review volume grows, one label delay after the drift.
- **Budget mode** (`mode="budget", review_budget=0.005`). `t_low` is steered so that the share of REVIEW decisions tracks the budget. The review rate is known as soon as predictions are made, so the queue stays on budget from day one. Fraud coverage is then an *output*, not a guarantee. `estimated_fraud_coverage(window)` estimates it from the fraud labels as they arrive.

Use coverage mode when missing fraud is the binding constraint, and budget mode when review capacity is.

## Validation results

`benchmarks/scenario_report.py` runs seven simulated scenarios: 2% fraud, Beta-distributed scores, fixed seeds, targets of 0.90 fraud coverage and 0.95 legit coverage. `tests/test_scenarios.py` asserts on the same numbers. Headline results:

| Scenario | delaycp | Static split conformal |
|---|---|---|
| Stationary, no delay | fraud 0.900, legit 0.950, review 5.1% | fraud 0.918, legit 0.949, review 5.1% |
| Every label 1,000 steps late | fraud 0.901 (guarantee ±0.034), legit 0.950, worst 2,000-legit window 0.506 | – |
| Chargeback delays, 60-day maturity | true fraud 0.908, true legit 0.956 (measured 0.952) | – |
| Fraud prevalence ×5 at day 50 | fraud after 0.908, review after 34.6% | fraud after 0.915, review after 5.3% |
| Fraud scores drift down at day 50 | fraud after 0.919, review after 40.4% | fraud after 0.195, review after 5.8% |
| Only 10 calibration frauds | fraud 0.892 | fraud 0.682 |
| Budget mode (3%), score drift | review 3.0%, fraud 0.999 → 0.765 | – |
| Adversary at 5% / 50% of frauds | fraud 0.901 / 0.887, review 7.4% / 85.1% | – |

The full tables and the reasoning behind each scenario are in the [validation report](https://adri0.github.io/delaycp/validation/). Regenerate it with `uv run python -m benchmarks.scenario_report`.

On real data, [`examples/ieee_cis_chargebacks.py`](https://github.com/adri0/delaycp/blob/main/examples/ieee_cis_chargebacks.py) runs delaycp on the [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) data with simulated chargeback delays. It compares a fixed score band, static split conformal, static Mondrian split conformal, and delaycp in both modes. The [notebook version](https://adri0.github.io/delaycp/examples/ieee_cis_chargebacks/) walks through it with commentary. To run it, download `train_transaction.csv` and `train_identity.csv`, then:

```bash
uv sync --extra examples   # LightGBM, pandas, matplotlib, Jupyter: example-only dependencies
uv run python examples/ieee_cis_chargebacks.py path/to/ieee-cis
```

## Limitations

- **Binary only.** Labels are `0` (legit) and `1` (fraud).
- **The guarantees are long-run, not per transaction.** Coverage converges per class over many transactions. It says nothing about any single decision, and short windows can deviate a lot. This is especially true with long delays: both class trackers share one `lr`, and with about 1,000 legit labels in flight the legit threshold oscillates (scenario 2). In budget mode fraud coverage is not guaranteed at all.
- **Maturity-window bias.** Frauds reported after the maturity window are counted as legit. This pushes `t_high` up and lowers true legit coverage, while measured legit coverage still reads on target, so the bias is invisible in production (scenario 3). Choose a window that covers most of the chargeback delay.
- **The delay analysis rests on a preprint.** The delayed-feedback treatment follows El Halabi & Brandt (2026), which has not been peer reviewed.
- **Coverage is not cost.** Holding coverage under drift or attack can send a large share of traffic to review (scenarios 4 and 7).

## Not in v0.1 (roadmap)

These are deliberately left out of v0.1. See the [roadmap](https://adri0.github.io/delaycp/roadmap/) for details.

- [ ] Scorecaster (the D term of conformal PID control)
- [ ] Label-shift reweighting ([Podkopaev & Ramdas, 2021](https://proceedings.mlr.press/v161/podkopaev21a.html))
- [ ] Multiclass labels
- [ ] Segment Mondrian (per-segment thresholds, e.g. by merchant category or country)
- [ ] Graph methods (sets that use links between transactions)

## Development

```bash
uv run ruff check .
uv run mypy src
uv run pytest            # add -m "not slow" to skip the ~30 s scenario suite
```

Docs are built with [Zensical](https://zensical.org/). API pages are generated from the docstrings, through Zensical's built-in mkdocstrings support:

```bash
uv sync --extra docs
uv run python -m scripts.gen_docs_pages   # refresh the notebook and validation pages
uv run zensical serve
```

Releases: bump `version` in `pyproject.toml` and `__version__` in `src/delaycp/__init__.py`, then push a tag `vX.Y.Z`. The release workflow builds the package and publishes it to TestPyPI.

## Citation

If you use delaycp in your work, please cite it. GitHub's "Cite this repository" button uses [`CITATION.cff`](https://github.com/adri0/delaycp/blob/main/CITATION.cff).

## References

- I. Gibbs and E. Candès (2021). Adaptive conformal inference under distribution shift. *NeurIPS*. [arXiv:2106.00170](https://arxiv.org/abs/2106.00170)
- A. N. Angelopoulos, E. J. Candès and R. J. Tibshirani (2023). Conformal PID control for time series prediction. *NeurIPS*. [arXiv:2307.16895](https://arxiv.org/abs/2307.16895). Reference code: [aangelopoulos/conformal-time-series](https://github.com/aangelopoulos/conformal-time-series)
- L. El Halabi and A. Brandt (2026). Adaptive conformal inference under delayed feedback: coverage guarantees and a delay-to-memory diagnostic. Preprint, [arXiv:2609.07251](https://arxiv.org/abs/2609.07251)
- V. Vovk, A. Gammerman and G. Shafer (2005). *Algorithmic Learning in a Random World*. Springer. (Mondrian conformal prediction)
- A. Podkopaev and A. Ramdas (2021). Distribution-free uncertainty quantification for classification under label shift. *UAI*. [PMLR 161](https://proceedings.mlr.press/v161/podkopaev21a.html)
