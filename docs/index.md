# delaycp

**Online conformal prediction for streaming binary classification with delayed, class-imbalanced labels.**

`delaycp` wraps any binary classifier that outputs `p = P(fraud)` and turns each score into one of three decisions: **approve**, **review** or **decline**. It keeps per-class error rates on target while labels arrive late, fraud is rare and the data drifts. It needs only NumPy and does not touch your model.

!!! warning "Pre-alpha"
    APIs are unstable and may change without notice.

## Why

- **Delayed labels.** Fraud is confirmed weeks later by a chargeback, and legit only once enough time passes without one.
- **Class imbalance.** With well under 1% fraud, an overall error target says nothing about how much fraud gets through.
- **Drift.** Fraudsters adapt, and thresholds tuned on old data quietly stop catching new fraud.
- **Review budgets.** Uncertain cases go to people, and their time is limited.

delaycp keeps one threshold per class and updates both from each label as it arrives. Each label is judged against the thresholds that were in effect when its prediction was made. [Concepts](concepts.md) explains how this works in plain language.

## Install

v0.1.0 is published to TestPyPI only. NumPy comes from PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ delaycp
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

## Where next

- [Concepts](concepts.md): the three regions, coverage vs budget mode, and what the guarantee means.
- [Example: IEEE-CIS chargebacks](examples/ieee_cis_chargebacks.md): delaycp on real fraud data with simulated chargeback delays.
- [Validation scenarios](validation.md): the simulated scenarios the test suite checks.
- [API reference](reference/delaycp/index.md): generated from the docstrings.
- [Roadmap](roadmap.md): what is deliberately not in v0.1.
