"""Bidirectional credibility weight tracking.

Implements Eq. 8 from Section 4.2:
  Invalid probe: w_i^(t+1) = w_i^(t) * (1 - delta_t)
  Valid probe:   w_i^(t+1) = w_i^(t) + delta_t * (1 - w_i^(t))

where delta_t = fraction of invalid probes at round t.
Steady-state for agent with invalid rate p: w* = 1 - p.
"""


class CredibilityTracker:
    """Track per-auditor credibility weights."""

    def __init__(self, initial_weight: float = 1.0):
        self._weights: dict[str, float] = {}
        self._initial = initial_weight
        self._round_invalid_count = 0
        self._round_total_count = 0

    def get_weight(self, auditor_id: str) -> float:
        return self._weights.get(auditor_id, self._initial)

    def update(self, auditor_id: str, valid: bool) -> None:
        """Update credibility after observing a probe.

        Bidirectional: invalid probes decay, valid probes recover.
        """
        w = self.get_weight(auditor_id)
        delta = self._current_delta()

        if valid:
            # Recovery toward 1.0
            w = w + delta * (1.0 - w)
        else:
            # Decay toward 0.0
            w = w * (1.0 - delta)

        self._weights[auditor_id] = max(0.0, min(1.0, w))

        # Track round statistics for delta estimation
        self._round_total_count += 1
        if not valid:
            self._round_invalid_count += 1

    def _current_delta(self) -> float:
        """Adaptive update rate = fraction of invalid probes."""
        if self._round_total_count == 0:
            return 0.1  # cold-start
        return self._round_invalid_count / self._round_total_count

    def reset_round(self) -> None:
        """Reset per-round counters (call at end of each round)."""
        self._round_invalid_count = 0
        self._round_total_count = 0

    def get_all_weights(self) -> dict[str, float]:
        return dict(self._weights)
