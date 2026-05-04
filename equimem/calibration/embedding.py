"""Embedding-based memory calibration.

Implements the two-pass scoring from Section 4.2:
  Pass 1 (detection): rho_detect via neighborhood density
  Pass 2 (alignment): rho_align via retrieval disruption (RBO)

Composite: rho = sqrt(rho_detect * rho_align)
"""

import numpy as np
from .base import CalibratorBase, CalibrationResult
from .credibility import CredibilityTracker
from ..utils.rbo import rbo_distance


class EmbeddingCalibrator(CalibratorBase):
    """Calibrator for embedding (vector) memory."""

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        self.credibility = CredibilityTracker()
        self._neighborhood_sizes: list[float] = []

    def score(self, delta, memory_state, auditor_signals, **kwargs) -> CalibrationResult:
        """Score a proposed embedding delta.

        Args:
            delta: list of (vector, text) tuples to be added.
            memory_state: Current memory with .search() and .vectors.
            auditor_signals: list of (auditor_id, query_str) tuples,
                intercepted from agents' existing retrieval queues.
        """
        rho_detect = self._detection_pass(delta, memory_state)
        rho_align = self._alignment_pass(delta, memory_state, auditor_signals)
        rho = self.geometric_mean(rho_detect, rho_align)
        committed = rho >= self.threshold

        if committed:
            self._update_history(rho)

        return CalibrationResult(
            rho=rho,
            rho_detect=rho_detect,
            rho_align=rho_align,
            committed=committed,
            details={"threshold": self.threshold},
        )

    def _detection_pass(self, delta, memory_state) -> float:
        """Compute rho_detect: neighborhood density of new vectors.

        High density = new entry dominates retrieval over a wide
        region = high-influence manipulation that u_c rewards.
        """
        if not delta or not memory_state.vectors:
            return 1.0  # vacuously pass if memory is empty

        # Adaptive radius: running median of pairwise similarities
        radius = self._adaptive_radius(memory_state)
        r_bar = np.mean(self._neighborhood_sizes) if self._neighborhood_sizes else 1.0

        ratios = []
        for vec, _ in delta:
            # Count neighbors within adaptive radius
            similarities = memory_state.similarities(vec)
            r_k = np.sum(similarities >= radius)
            self._neighborhood_sizes.append(r_k)
            ratios.append(r_k / (r_bar + 1e-8))

        return max(0.0, 1.0 - np.mean(ratios))

    def _alignment_pass(self, delta, memory_state, auditor_signals) -> float:
        """Compute rho_align: retrieval disruption via RBO.

        For each auditor's intercepted query, measure how much
        delta changes the top-k retrieval ranking.
        """
        if not auditor_signals:
            return 1.0

        disruptions = []
        for auditor_id, query in auditor_signals:
            w_i = self.credibility.get_weight(auditor_id)

            # Validate probe (structural check, not semantic)
            if not self._is_valid_probe(query):
                self.credibility.update(auditor_id, valid=False)
                continue
            self.credibility.update(auditor_id, valid=True)

            # Top-k before delta
            topk_before = memory_state.search(query, k=self.top_k)
            # Top-k after delta (simulated merge)
            topk_after = memory_state.search_with_delta(query, delta, k=self.top_k)

            d_rbo = rbo_distance(topk_before, topk_after)
            disruptions.append(w_i * d_rbo)

        if not disruptions:
            return 1.0

        return max(0.0, 1.0 - np.mean(disruptions))

    def _adaptive_radius(self, memory_state) -> float:
        """Running median of pairwise similarities."""
        if hasattr(memory_state, '_similarity_cache'):
            return np.median(memory_state._similarity_cache)
        return 0.5  # fallback

    @staticmethod
    def _is_valid_probe(query: str) -> bool:
        """Structural validity check (constant-time, no semantics)."""
        if not query or not query.strip():
            return False
        if len(query) < 3 or len(query) > 2000:
            return False
        return True
