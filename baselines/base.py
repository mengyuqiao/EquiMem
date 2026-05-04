"""Base interface for all safeguard methods."""

from abc import ABC, abstractmethod
from equimem.calibration.base import CalibrationResult


class SafeguardBase(ABC):
    """All safeguards (including EquiMem) implement this interface."""

    @abstractmethod
    def score(self, delta, memory_state, auditor_signals, **kwargs) -> CalibrationResult:
        """Score a proposed delta.

        Args:
            delta: Proposed memory update.
            memory_state: Current memory state M_t.
            auditor_signals: list of (auditor_id, probe) tuples.

        Returns:
            CalibrationResult with rho and commit decision.
        """
        ...
