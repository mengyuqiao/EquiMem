"""Vanilla safeguard: no defence.

Always commits every delta with full trust. Used as the
lower-bound baseline in all experiments.
"""

from .base import SafeguardBase
from equimem.calibration.base import CalibrationResult


class VanillaSafeguard(SafeguardBase):

    def score(self, delta, memory_state, auditor_signals, **kwargs):
        return CalibrationResult(
            rho=1.0,
            rho_detect=1.0,
            rho_align=1.0,
            committed=True,
        )
