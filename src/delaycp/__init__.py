"""delaycp: online conformal prediction for streaming, delayed-label binary classification.

The main entry point, [`OnlineConformalClassifier`][delaycp.stream.OnlineConformalClassifier],
is re-exported here together with [`Action`][delaycp.types.Action] and
[`Thresholds`][delaycp.types.Thresholds]. The modules are:

- [`delaycp.stream`][]: the streaming classifier (predict, buffer, update).
- [`delaycp.mondrian`][]: per-class thresholds and the set-to-action mapping.
- [`delaycp.online`][]: the single-threshold PID quantile tracker.
- [`delaycp.delay`][]: the buffer that holds predictions until labels arrive.
- [`delaycp.budget`][]: budget mode, holding the REVIEW rate near a target.
- [`delaycp.metrics`][]: coverage, action-rate and cost metrics.
- [`delaycp.simulate`][]: synthetic streams, delay injection and replay.
- [`delaycp.types`][]: shared types.
"""

from delaycp.stream import OnlineConformalClassifier
from delaycp.types import Action, Thresholds

__version__ = "0.1.0"

__all__ = ["Action", "OnlineConformalClassifier", "Thresholds", "__version__"]
