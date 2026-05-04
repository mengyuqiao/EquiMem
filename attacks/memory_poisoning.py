"""Memory Poisoning attack.

Adversarial contributors inject fabricated entries designed to
pass detection checks while corrupting downstream reasoning.

For embedding memory:
  Craft vectors with high cosine similarity to many existing
  entries, creating "retrieval magnets" that dominate top-k.

For graph memory:
  Add edges that are locally plausible (pass s_local) but
  globally inconsistent (fail s_path at depth >= 2).

This attack targets the contributor side of the zero-trust
game — the detection signal D = 1 - rho_detect.
"""

import numpy as np
from .base import AttackBase


class MemoryPoisoningAttack(AttackBase):
    """Inject fabricated memory entries."""

    def __init__(self, n_adversaries: int, poison_strength: float = 0.8):
        super().__init__(n_adversaries)
        self.poison_strength = poison_strength

    def modify_delta(self, delta, memory, agent_idx, rng):
        """Replace delta with poisoned content."""
        if memory is None:
            return delta

        state = memory.get_state()

        # Detect memory type and craft appropriate poison
        if hasattr(state, "vectors") and state.vectors is not None:
            return self._poison_embedding(delta, state, rng)
        elif hasattr(state, "get_neighbors"):
            return self._poison_graph(delta, state, rng)
        else:
            return self._poison_text(delta, rng)

    def _poison_embedding(self, delta, state, rng):
        """Craft embedding vectors near the centroid of existing memory.

        Strategy: place poisoned vectors close to the centroid so they
        have high similarity to many entries, creating retrieval magnets
        that dominate top-k results for a wide range of queries.
        """
        if not hasattr(state, "vectors") or len(state.vectors) == 0:
            return delta

        vectors = np.array(state.vectors)
        centroid = np.mean(vectors, axis=0)

        poisoned = []
        for item in delta if isinstance(delta, list) else [delta]:
            # Perturb centroid slightly to avoid exact detection
            noise = rng.normal(0, 0.02, centroid.shape)
            poison_vec = centroid + noise
            poison_vec = poison_vec / (np.linalg.norm(poison_vec) + 1e-8)

            # Generate plausible-sounding but incorrect text
            poison_text = self._generate_plausible_falsehood(item, rng)
            poisoned.append((poison_vec, poison_text))

        return poisoned

    def _poison_graph(self, delta, state, rng):
        """Add edges that are locally plausible but globally inconsistent.

        Strategy: pick a high-degree node u, find a distant node v
        (>= 3 hops away), and connect them with u's most common
        relation type. This passes s_local (the relation is common
        for u) but fails s_path (no existing multi-hop path supports
        this connection).
        """
        if not hasattr(state, "get_neighbors"):
            return delta

        poisoned = []
        for item in delta if isinstance(delta, list) else [delta]:
            if isinstance(item, tuple) and len(item) == 3:
                u, r, v = item
                # Keep u and r (locally plausible) but redirect v
                # to a distant node
                distant_v = self._find_distant_node(u, state, rng)
                if distant_v is not None:
                    poisoned.append((u, r, distant_v))
                else:
                    poisoned.append(item)
            else:
                poisoned.append(item)

        return poisoned

    def _poison_text(self, delta, rng):
        """Fallback: inject plausible-sounding incorrect text."""
        if isinstance(delta, str):
            return self._generate_plausible_falsehood(delta, rng)
        return delta

    def _find_distant_node(self, source, state, rng, min_hops=3):
        """Find a node at least min_hops away from source via BFS."""
        visited = {source}
        frontier = [source]

        for depth in range(min_hops):
            next_frontier = []
            for node in frontier:
                neighbors = state.get_neighbors(node)
                for _, _, v in neighbors:
                    if v not in visited:
                        visited.add(v)
                        next_frontier.append(v)
            frontier = next_frontier

        # Pick from nodes beyond min_hops
        if frontier:
            return rng.choice(frontier)

        # Fallback: any unvisited node
        all_nodes = set(state.get_all_nodes()) if hasattr(state, "get_all_nodes") else set()
        candidates = list(all_nodes - visited)
        if candidates:
            return rng.choice(candidates)
        return None

    @staticmethod
    def _generate_plausible_falsehood(original, rng):
        """Craft text that reads naturally but contains incorrect facts.

        In a real attack, this would use an LLM to generate fluent
        but false text. Here we simulate by corrupting key entities.
        """
        if isinstance(original, str):
            return f"[POISONED] {original}"
        if isinstance(original, tuple) and len(original) == 2:
            _, text = original
            return f"[POISONED] {text}"
        return str(original)
