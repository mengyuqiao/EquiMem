"""Graph-based memory calibration.

Implements the two-pass scoring from Section 4.2:
  Pass 1 (detection): s_local (neighborhood) + s_path (multi-hop BFS)
    combined as rho_detect = alpha * s_local + (1-alpha) * s_path
  Pass 2 (alignment): rho_align via reachability probes

Composite: rho = sqrt(rho_detect * rho_align)
"""

import math
import numpy as np
from .base import CalibratorBase, CalibrationResult
from .credibility import CredibilityTracker
from ..utils.graph_bfs import bidirectional_bfs
from ..utils.compat import relation_compatibility


class GraphCalibrator(CalibratorBase):
    """Calibrator for graph (structural) memory."""

    def __init__(self):
        super().__init__()
        self.credibility = CredibilityTracker()
        self._compat_history: list[float] = []

    def score(self, delta_edges, graph, auditor_signals, **kwargs) -> CalibrationResult:
        """Score proposed graph edges.

        Args:
            delta_edges: list of (u, r, v) triples to be added.
            graph: Current graph memory with adjacency and edge types.
            auditor_signals: list of (auditor_id, path) tuples,
                where path is the auditor's current traversal path.
        """
        rho_detect = self._detection_pass(delta_edges, graph)
        rho_align = self._alignment_pass(delta_edges, graph, auditor_signals)
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

    def _detection_pass(self, delta_edges, graph) -> float:
        """Compute rho_detect per edge, then average.

        For each edge: rho_detect(e) = alpha * s_local(e) + (1-alpha) * s_path(e)
        alpha = fraction of edges with at least one supporting path.
        """
        if not delta_edges:
            return 1.0

        L_max = max(2, math.ceil(math.log(max(1, graph.num_nodes()))))
        eta_t = self._adaptive_eta()

        edge_scores = []
        has_path_count = 0

        for u, r, v in delta_edges:
            s_loc = self._s_local(u, r, graph)
            s_pat = self._s_path(u, r, v, graph, L_max, eta_t)
            if s_pat > 0:
                has_path_count += 1
            edge_scores.append((s_loc, s_pat))

        # Alpha: fraction with supporting paths
        alpha = has_path_count / len(delta_edges) if delta_edges else 0.5

        rho_detect_values = [
            alpha * s_loc + (1 - alpha) * s_pat
            for s_loc, s_pat in edge_scores
        ]
        return np.mean(rho_detect_values)

    def _alignment_pass(self, delta_edges, graph, auditor_signals) -> float:
        """Compute rho_align via reachability probes.

        Each auditor's current traversal path is checked for
        reachability to a randomly sampled delta edge.
        """
        if not auditor_signals or not delta_edges:
            return 1.0

        rng = np.random.default_rng()
        reachable_scores = []

        for auditor_id, auditor_path in auditor_signals:
            w_i = self.credibility.get_weight(auditor_id)

            # Sample one edge uniformly at random
            e_star = delta_edges[rng.integers(len(delta_edges))]

            # Check reachability from auditor's path endpoints
            is_reachable = self._check_reachability(
                e_star, auditor_path, graph
            )

            if not is_reachable:
                self.credibility.update(auditor_id, valid=False)
            else:
                self.credibility.update(auditor_id, valid=True)

            reachable_scores.append(w_i * float(is_reachable))

        return np.mean(reachable_scores) if reachable_scores else 1.0

    def _s_local(self, u: str, r: str, graph) -> float:
        """Local consistency: fraction of u's neighbors with compatible relations."""
        neighbors = graph.get_neighbors(u)
        if not neighbors:
            return 0.0

        compat_set = graph.get_compatible_relations(r)
        compatible_count = sum(
            1 for _, r_prime, _ in neighbors
            if r_prime in compat_set
        )
        return compatible_count / (len(neighbors) + 1e-8)

    def _s_path(self, u: str, r: str, v: str, graph, L_max: int, eta: float) -> float:
        """Path consistency: fraction of u-v paths supporting relation r."""
        paths = bidirectional_bfs(graph, u, v, max_depth=L_max)
        if not paths:
            return 0.0

        supporting = sum(
            1 for path in paths
            if relation_compatibility(r, path.relation_type, graph) >= eta
        )
        return supporting / len(paths)

    def _check_reachability(self, edge, auditor_path, graph) -> bool:
        """O(log|V|) reachability check from auditor path endpoints."""
        u, r, v = edge
        if not auditor_path:
            return False

        endpoints = [auditor_path[0], auditor_path[-1]]
        compat_set = graph.get_compatible_relations(r)

        for endpoint in endpoints:
            # Check 1-hop reachability via compatible relations
            neighbors = graph.get_neighbors(endpoint)
            for _, r_prime, target in neighbors:
                if target in (u, v) and r_prime in compat_set:
                    return True
        return False

    def _adaptive_eta(self) -> float:
        """Running median of historical compatibility scores."""
        if len(self._compat_history) < 5:
            return 0.3  # cold-start
        return float(np.median(self._compat_history[-100:]))
