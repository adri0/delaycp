# Concepts

This page explains what delaycp does in plain language. The [API reference](reference/delaycp/index.md) has the details.

## Three regions: approve, review, decline

Your model gives each transaction a score `p`, its estimate of the probability that the transaction is fraud. delaycp does not change the model. It picks two cut points on the score, `t_low` and `t_high`, and these split the score line into three regions:

<figure>
<svg viewBox="0 0 720 240" width="720" role="img" aria-labelledby="regions-title regions-desc" style="max-width:100%;height:auto;font-family:inherit">
  <title id="regions-title">The approve, review and decline regions</title>
  <desc id="regions-desc">A score line from 0 to 1 with two cut points. Below t_low only "legit" is plausible and the transaction is approved. Above t_high only "fraud" is plausible and it is declined. Between them both labels are plausible and it goes to manual review.</desc>
  <g font-size="13" fill="currentColor">
    <rect x="40" y="14" width="448" height="24" rx="4" fill="#2e9e5b" fill-opacity="0.18" stroke="#2e9e5b"/>
    <text x="52" y="31">"legit" is plausible: p ≤ t_high</text>
    <rect x="200" y="46" width="480" height="24" rx="4" fill="#d1493f" fill-opacity="0.18" stroke="#d1493f"/>
    <text x="668" y="63" text-anchor="end">"fraud" is plausible: p ≥ t_low</text>
  </g>
  <g font-size="14" fill="currentColor" text-anchor="middle">
    <rect x="40" y="88" width="160" height="66" fill="#2e9e5b" fill-opacity="0.30" stroke="#2e9e5b"/>
    <text x="120" y="116" font-weight="700">APPROVE</text>
    <text x="120" y="138" font-size="13">set {legit}</text>
    <rect x="200" y="88" width="288" height="66" fill="#d99a1e" fill-opacity="0.30" stroke="#d99a1e"/>
    <text x="344" y="116" font-weight="700">REVIEW</text>
    <text x="344" y="138" font-size="13">set {legit, fraud}</text>
    <rect x="488" y="88" width="192" height="66" fill="#d1493f" fill-opacity="0.30" stroke="#d1493f"/>
    <text x="584" y="116" font-weight="700">DECLINE</text>
    <text x="584" y="138" font-size="13">set {fraud}</text>
  </g>
  <g stroke="currentColor" fill="none">
    <line x1="40" y1="176" x2="680" y2="176"/>
    <line x1="40" y1="170" x2="40" y2="182"/>
    <line x1="680" y1="170" x2="680" y2="182"/>
    <line x1="200" y1="76" x2="200" y2="182" stroke-dasharray="4 3"/>
    <line x1="488" y1="76" x2="488" y2="182" stroke-dasharray="4 3"/>
  </g>
  <g font-size="13" fill="currentColor" text-anchor="middle">
    <text x="40" y="200">0</text>
    <text x="200" y="200" font-weight="700">t_low</text>
    <text x="488" y="200" font-weight="700">t_high</text>
    <text x="680" y="200">1</text>
    <text x="360" y="228">model score p = P(fraud)</text>
  </g>
</svg>
<figcaption>Each transaction lands in one region depending on its score.</figcaption>
</figure>

- **Approve** (`p < t_low`). The score is so low that "fraud" is not a plausible label. Let the transaction through.
- **Decline** (`p > t_high`). The score is so high that "legit" is not a plausible label. Block it.
- **Review** (`t_low ≤ p ≤ t_high`). Both labels are plausible, so the model alone cannot decide. Send the transaction to a person.

The set of plausible labels is the transaction's *prediction set*. Approve, review and decline are just the three non-empty sets: `{legit}`, `{legit, fraud}` and `{fraud}`.

If the model separates the classes very well, the cut points can cross (`t_low > t_high`). Then a score between them has an *empty* set: neither label is plausible. delaycp sends these to review too, because an empty set means the data no longer looks like what the thresholds were tuned on.

## Where the cut points come from

Each cut point protects one class, and you choose how much error to tolerate for each:

- `t_low` is set so that at most a fraction `alpha_fraud` of frauds score below it. These are the frauds that get approved. With `alpha_fraud = 0.10`, 90% of frauds are reviewed or declined.
- `t_high` is set so that at most a fraction `alpha_legit` of legit transactions score above it. These are the good customers who get declined. With `alpha_legit = 0.05`, 95% of legit transactions are approved or reviewed.

The fraction of a class that lands in a region where its label is plausible is that class's *coverage*. Targeting each class separately is called *Mondrian* (class-conditional) conformal prediction. It matters when fraud is rare. A single shared target would be met almost entirely by the legit majority and say nothing about fraud. The [IEEE-CIS example](examples/ieee_cis_chargebacks.md) shows this happening.

How many transactions end up in review depends on how good the model is: the more the two classes' scores overlap, the wider the middle region must be to meet both targets.

## Learning from labels as they arrive

The cut points start from a calibration set of labelled history (`fit_calibration`). After that they move online. Each time a label arrives, delaycp checks whether that transaction was *covered*, that is whether its true label was in its set. If a fraud was approved, `t_low` moves down a little. If a fraud was caught, it moves up a tiny bit. Legit labels move `t_high` the same way. The step size is `lr`, and the steps are sized so that, in the long run, the share of misses for each class matches its `alpha`. This is conformal PID control ([Angelopoulos, Candès & Tibshirani, 2023](https://arxiv.org/abs/2307.16895)), with the P term and an optional I term (`ki`).

Because the cut points keep adjusting, they follow drift. When fraudsters change tactics and the model starts scoring new fraud lower, misses pile up and `t_low` comes down until coverage recovers. A threshold frozen after calibration would not recover.

## Delayed labels

In fraud, the true label often arrives weeks later as a chargeback, and "legit" is usually only known once enough time has passed without one. delaycp handles this in two ways:

- **Judge each label against the thresholds it was scored with.** Every prediction is stored with the cut points in effect at that moment. When its label arrives, possibly much later, the miss or hit is judged against those stored cut points, not the current ones. This is what keeps the long-run guarantee valid under delay ([El Halabi & Brandt, 2026](https://arxiv.org/abs/2609.07251)).
- **Maturity window.** With `maturity=60`, a transaction with no chargeback after 60 days is treated as legit (`unlabeled_at_maturity=0`), or dropped (`None`). Frauds that are reported later than that are counted as legit. That biases the legit cut point, and the bias does not show up in the measured coverage. Pick a window that covers most of your chargeback delay. Scenario 3 of the [validation report](validation.md) measures this.

Delay slows feedback down. Corrections arrive one label delay late, so the cut points keep correcting for a while after a problem is gone. When many labels are in flight at once, a large `lr` makes them oscillate. Scenario 2 of the validation report shows how much.

## Coverage mode and budget mode

Review is done by people, and their time is limited. delaycp has two modes that differ in what `t_low` targets:

| | Coverage mode (default) | Budget mode |
|---|---|---|
| `t_low` targets | Fraud coverage `1 - alpha_fraud` | A review rate, e.g. 0.5% of traffic |
| Driven by | Fraud labels (arrive late) | The review rate itself (known immediately) |
| Fraud coverage | Guaranteed in the long run | An output; can drop under drift |
| Review volume | Whatever coverage needs; can grow | Held at the budget |
| `t_high` | Legit coverage target | Legit coverage target |

Use **coverage mode** when missing fraud is the constraint you cannot break. Use **budget mode** (`mode="budget"`, `review_budget=0.005`) when review capacity is. In budget mode the fraud labels are still used, but only to *estimate* fraud coverage (`estimated_fraud_coverage`). Watch that number: if fraud scores drift down, coverage falls, and the estimate shows the drop one label delay later.

## What the guarantee does and does not say

- It is **long-run and per class**. Over many transactions, the share of approved frauds tends to `alpha_fraud` and the share of declined legit transactions tends to `alpha_legit`. With `ki = 0` there is an explicit bound that holds for any sequence of scores and labels, adversarial ones included, once every label has arrived.
- It says **nothing about one transaction**. A single REVIEW does not mean "50% fraud", and an APPROVE is not "at most 10% fraud".
- Short windows can deviate a lot, especially with long delays or many labels in flight.
- It is about coverage, not cost. Holding coverage under drift can mean sending a lot of traffic to review. Scenario 7 of the validation report shows a strong adversary pushing review above 80%.
