# Roadmap

v0.1 is deliberately small: binary labels, per-class PID thresholds, delay handling and a review budget. The items below are **not** in v0.1. They are planned, roughly in order, but nothing here is promised.

- [ ] **Scorecaster.** The D term of conformal PID control: a model that forecasts the next score quantile, so thresholds can move ahead of a predictable change (weekly cycles, paydays) instead of after it.
- [ ] **Label-shift reweighting.** Weight calibration and online updates by an estimate of the current class prior ([Podkopaev & Ramdas, 2021](https://proceedings.mlr.press/v161/podkopaev21a.html)), for when the fraud rate moves and labels lag.
- [ ] **Multiclass.** More than two labels (e.g. fraud types), with sets beyond `{legit}`, `{fraud}` and both.
- [ ] **Segment Mondrian.** Separate thresholds per segment (merchant category, country, channel) in addition to per class, with pooling for small segments.
- [ ] **Graph methods.** Conformal sets that use links between transactions (shared cards, devices, accounts), where exchangeability fails along the graph.

Have a use case for one of these, or something else? [Open an issue](https://github.com/adri0/delaycp/issues).
