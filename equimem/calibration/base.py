"""Base calibration interface for EquiMem."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CalibrationResult:
    """Result of calibrating a proposed delta."""
    rho: float                  # Composite calibration score in [0, 1]
    rho_detect: float           # Detection signal
    rho_align: float            # Alignment signal
    committed: bool             # Whether rho >= rho*
    details: dict[str, Any] = None


class CalibratorBase(ABC):
    """Abstract base class for memory calibration.

    All calibration logic is LLM-free. The only inputs
    from agents are their existing retrieval queries
    (embedding) or traversal paths (graph), intercepted
    from normal debate activity.
    """

    def __init__(self):
        self._score_history: list[float] = []
        self._rho_star: float = 0.5  # initial cold-start

    @property
    def threshold(self) -> float:
        """Adaptive commit threshold (running median)."""
        if len(self._score_history) < 5:
            return self._rho_star
        sorted_scores = sorted(self._score_history[-100:])
        return sorted_scores[len(sorted_scores) // 2]

    @abstractmethod
    def score(self, delta, memory_state, auditor_signals, **kwargs) -> CalibrationResult:
        """Score a proposed delta against the memory state.

        Args:
            delta: Proposed memory update (entries or edges).
            memory_state: Current memory state M_t.
            auditor_signals: Intercepted probes from auditors.

        Returns:
            CalibrationResult with composite score and commit decision.
        """
        ...

    def _update_history(self, rho: float) -> None:
        """Record a committed score for threshold adaptation."""
        self._score_history.append(rho)

    @staticmethod
    def geometric_mean(a: float, b: float) -> float:
        """Compute sqrt(a * b), the geometric mean used for composition."""
        return (max(0.0, a) * max(0.0, b)) ** 0.5
