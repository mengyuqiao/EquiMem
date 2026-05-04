"""Audit Collusion attack.

Adversarial auditors coordinate their probe signals to inflate
rho_align, making poisoned entries appear consistent with the
memory state.

For embedding memory:
  Craft probe queries designed to minimize d_RBO when the
  poisoned delta is merged — the query is chosen so that the
  top-k results don't change, hiding the disruption.

For graph memory:
  Submit attestation paths that validate incorrect edges by
  constructing paths through existing nodes that appear to
  reach the poisoned edge's endpoints.

This attack targets the auditor side of the zero-trust game —
the alignment signal and credibility weights w_i.
"""

import numpy as np
from .base import AttackBase


class AuditCollusionAttack(AttackBase):
    """Adversarial auditors coordinate to inflate rho_align."""

    def __init__(self, n_adversaries: int):
        super().__init__(n_adversaries)

    def modify_delta(self, delta, memory, agent_idx, rng):
        """Memory Poisoning is not the focus of this attack,
        but colluding auditors need something to collude about.
        We inject mild poison (less aggressive than MemoryPoisoning).
        """
        if isinstance(delta, str):
            return f"[COLLUDED] {delta}"
        return delta

    def modify_probe(self, probe, delta, memory, rng):
        """Craft a probe that makes the poisoned delta look consistent.

        The colluding auditor replaces their real probe with one
        designed to produce low d_RBO (embedding) or validate
        the poisoned edge (graph).
        """
        if memory is None:
            return probe

        state = memory.get_state()

        if hasattr(state, "vectors") and state.vectors is not None:
            return self._craft_embedding_probe(delta, state, rng)
        elif hasattr(state, "get_neighbors"):
            return self._craft_graph_probe(delta, state, rng)
        else:
            return probe

    def _craft_embedding_probe(self, delta, state, rng):
        """Craft a query whose top-k results are stable with/without delta.

        Strategy: choose a query that retrieves entries far from
        the poisoned vectors, so adding the poison doesn't change
        the top-k list (d_RBO ≈ 0, inflating rho_align).
        """
        if not hasattr(state, "vectors") or len(state.vectors) == 0:
            return "neutral query"

        vectors = np.array(state.vectors)

        # Find the region of memory farthest from the poisoned entries
        if isinstance(delta, list) and delta:
            if isinstance(delta[0], tuple) and len(delta[0]) == 2:
                poison_vecs = np.array([v for v, _ in delta])
            else:
                return "neutral query"
        else:
            return "neutral query"

        poison_centroid = np.mean(poison_vecs, axis=0)

        # Find memory entries most distant from poison
        similarities = vectors @ poison_centroid / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(poison_centroid) + 1e-8
        )
        farthest_idx = np.argmin(similarities)

        # Use the text of the farthest entry as the probe query
        # This ensures top-k results won't include the poison
        if hasattr(state, "entries"):
            return state.entries[farthest_idx]

        return "neutral query far from poison region"

    def _craft_graph_probe(self, delta, state, rng):
        """Craft an attestation path that reaches the poisoned edge.

        Strategy: find an existing path from a legitimate node
        to one of the poisoned edge's endpoints, using only
        compatible relation types. This makes the reachability
        check return True, inflating rho_align.
        """
        if not isinstance(delta, list) or not delta:
            return []

        # Pick one poisoned edge
        target_edge = delta[0] if isinstance(delta[0], tuple) else delta
        if not isinstance(target_edge, tuple) or len(target_edge) != 3:
            return []

        u, r, v = target_edge

        # Build a path from a well-connected node to u
        # using legitimate edges that exist in the graph
        path = self._find_legitimate_path_to(u, state, rng)
        return path

    def _find_legitimate_path_to(self, target, state, rng, max_depth=3):
        """BFS from high-degree nodes toward the target.

        Returns a list of nodes forming a path, or empty list if
        no path is found.
        """
        if not hasattr(state, "get_neighbors"):
            return []

        # Start from a random node with edges
        all_neighbors = state.get_neighbors(target)
        if not all_neighbors:
            return [target]

        # Build a short path ending at target
        path = [target]
        current = target
        for _ in range(max_depth):
            neighbors = state.get_neighbors(current)
            if not neighbors:
                break
            # Pick a random neighbor
            u, r, v = neighbors[rng.integers(len(neighbors))]
            prev_node = u if v == current else v
            if prev_node not in path:
                path.insert(0, prev_node)
                current = prev_node
            else:
                break

        return path
