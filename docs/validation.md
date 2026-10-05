## Validation scenarios

Targets: fraud coverage 0.90 (α = 0.1), legit coverage 0.95 (α = 0.05), `lr = 0.01` unless noted. Synthetic streams with 2.0% fraud from `delaycp.simulate`, fixed seeds. Regenerate with `python -m benchmarks.scenario_report`.

### 1. Stationary, no delay

200,000 transactions, 4073 frauds; both methods calibrated on the same 20,000 examples (424 frauds).

| Method | Fraud coverage | Legit coverage | Review rate |
|---|---|---|---|
| delaycp (online) | 0.900 | 0.950 | 5.1% |
| split conformal (static) | 0.918 | 0.949 | 5.1% |

### 2. Fixed label delay

One transaction per step, every label τ steps late. *Guarantee* is the a-priori bound on |fraud coverage − 0.90|. Rolling error is the mean |coverage − target| over windows of 200 frauds or 2,000 legits (independent outcomes would give about 0.017 and 0.004).

| τ (steps) | Fraud cov. | Guarantee | Fraud rolling error | Worst fraud window | Legit cov. | Legit rolling error | Worst legit window | Legit labels in flight |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.901 | ±0.025 | 0.011 | 0.855 | 0.950 | 0.001 | 0.945 | 1 |
| 10 | 0.901 | ±0.026 | 0.011 | 0.855 | 0.950 | 0.001 | 0.945 | 10 |
| 100 | 0.901 | ±0.027 | 0.011 | 0.855 | 0.950 | 0.002 | 0.942 | 100 |
| 1000 | 0.901 | ±0.034 | 0.013 | 0.850 | 0.950 | 0.084 | 0.506 | 994 |

Long-run coverage holds for every τ. Rolling error barely moves for fraud, but for legit it jumps at τ = 1000. Both trackers share `lr`, and legit labels outnumber fraud labels about 49 to 1, so roughly 1,000 legit updates are in flight at once. That makes the delayed tracker oscillate. Coverage still averages out, but individual windows can drop to about 0.5.

### 3. Chargeback delays and maturity window

Fraud labels arrive after a lognormal delay (median 30 days, σ = 0.8; 5% never reported). Legit labels are confirmed when the maturity window closes. Frauds not reported by then are imputed as legit, both in calibration and online.

| Maturity (days) | Frauds unreported in window | Legit class contamination | True fraud cov. | True legit cov. | Measured legit cov. | Predicted true legit cov. | Review rate |
|---|---|---|---|---|---|---|---|
| 14 | 84.7% | 1.7% | 0.945 | 0.969 | 0.952 | 0.966 | 3.1% |
| 30 | 51.9% | 1.0% | 0.933 | 0.962 | 0.952 | 0.960 | 3.8% |
| 60 | 22.3% | 0.5% | 0.908 | 0.956 | 0.952 | 0.954 | 4.4% |
| 90 | 12.1% | 0.2% | 0.910 | 0.954 | 0.952 | 0.952 | 4.6% |
| 180 | 5.8% | 0.1% | 0.910 | 0.953 | 0.952 | 0.951 | 4.7% |

Fraud coverage is unbiased: late frauds leave the fraud class independently of their score. The overshoot at short windows is calibration noise, because only the frauds reported in time are available to calibrate (65 at 14 days). Undiscovered frauds do pollute the legit class, and because nearly all of them score above `t_high`, the legit threshold rises until true legit coverage is ≈ 1 − (α₀ − c·m_F)/(1 − c). Here c is the contamination and m_F = P(p > t_high | fraud). The measured legit coverage still reads ≈ 0.95, so the bias is invisible in production. This scenario uses `lr = 5e-06`, because with tens of thousands of legit labels in flight the legit tracker is unstable at the default `lr` (see section 2).

### 4. Prevalence shift and score drift

Lognormal label delays (median 3 days, 100-day horizon). Cells are *delaycp / static split conformal*. *After* means from day 60 on.

| Change at day 50 | Fraud cov. before | Fraud cov. after | Legit cov. after | Review after |
|---|---|---|---|---|
| Prevalence ×5 | 0.904 / 0.917 | 0.908 / 0.915 | 0.932 / 0.950 | 34.6% / 5.3% |
| Fraud scores drift down | 0.899 / 0.917 | 0.919 / 0.195 | 0.932 / 0.950 | 40.4% / 5.8% |

A prevalence shift does not change either class's score distribution, so per-class coverage holds for both methods. When fraud scores drift, the static thresholds lose most fraud coverage while delaycp recovers it. The legit columns show the delay effect from section 2: thousands of legit labels are in flight, so `t_high` oscillates. That costs some legit coverage after the change and inflates the review rate.

### 5. Tiny fraud class

Calibration has fewer than 50 frauds (and 5,000 legits); 100,000-transaction stream.

| Calibration frauds | Warning | Split fraud cov. | Online fraud cov. (first 10%) | Online fraud cov. | Guarantee | Legit cov. | Review rate |
|---|---|---|---|---|---|---|---|
| 0 | yes | 1.000 | 0.995 | 0.920 | ±0.050 | 0.950 | 5.1% |
| 10 | yes | 0.682 | 0.842 | 0.892 | ±0.050 | 0.950 | 5.1% |
| 30 | yes | 0.897 | 0.911 | 0.900 | ±0.050 | 0.950 | 5.1% |

With 10 calibration frauds the split-conformal threshold is far off. The online tracker pulls it back towards 0.90 as fraud labels arrive. With no frauds at all it starts from `q = 1 − α` (conservative).

### 6. Budget mode under drift

Review budget 3.0%, batches of 100.

| Change at day 50 | Review rate | Review after | Max deviation (10k-tx windows) | Fraud cov. before → after | Estimated fraud cov. (last 1,000 labels) |
|---|---|---|---|---|---|
| Prevalence ×5 | 3.0% | 3.0% | 0.3% | 0.999 → 0.999 | 1.000 |
| Fraud scores drift down | 3.0% | 3.0% | 0.3% | 0.999 → 0.765 | 0.761 |

The review rate stays on budget through both changes. Fraud coverage is an output in this mode: score drift lowers it, and the label-based estimate shows the drop one label delay later.

### 7. Adversarial

A share of frauds is placed just below the live `t_low` (the adversary sees the current thresholds). Labels arrive with lognormal delays.

| Adversarial frauds | Fraud cov. | Guarantee | Worst fraud window | Mean t_low (2nd half) | Review rate |
|---|---|---|---|---|---|
| 5.0% | 0.901 | ±0.073 | 0.835 | 0.414 | 7.4% |
| 50.0% | 0.887 | ±0.073 | 0.465 | -0.082 | 85.1% |

The long-run guarantee holds in both cases. When the adversarial share is below α the band stays reasonable. Above α the only way to cover those frauds is `t_low ≈ 0`, so most traffic goes to review.
