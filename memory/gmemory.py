"""G-Memory adapter (Zhang et al., 2025).

Wraps G-Memory's own hierarchical three-tier graph memory
(insight → query → interaction graphs) into EquiMem's
GraphMemory interface.

G-Memory's internal structure is preserved; this adapter
exposes graph-level access for calibration while delegating
all memory logic to the vendor implementation.
"""

import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx

from .base import GraphMemory

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "GMemory"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))


class GMemoryAdapter(GraphMemory):
    """Wrap G-Memory's vendor implementation into GraphMemory interface."""

    def __init__(self, llms_type: list[str]):
        self._vendor = None
        self._graph = nx.DiGraph()
        self._trust_weights: dict[tuple, float] = {}
        self._relation_counts: dict[str, int] = defaultdict(int)
        self._cooccurrence: dict[tuple, int] = defaultdict(int)

        try:
            from mas.memory_gmemory import MemoryGMemory
            self._vendor = MemoryGMemory(llms_type)
        except ImportError:
            pass

    # ---- MemoryBase interface ----

    def save(self, content, trust_weight: float = 1.0) -> None:
        if self._vendor is not None:
            # Delegate to vendor's save logic
            if isinstance(content, str):
                self._vendor(content)
            self._sync_graph_from_vendor()
        else:
            # Standalone mode: directly add to local graph
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, tuple) and len(item) == 3:
                        u, r, v = item
                        self._graph.add_edge(u, v, relation=r)
                        self._trust_weights[(u, r, v)] = trust_weight
                        self._relation_counts[r] += 1
                        self._update_cooccurrence(u)
            elif isinstance(content, str):
                # Store as a node with the text
                node_id = f"entry_{self._graph.number_of_nodes()}"
                self._graph.add_node(node_id, text=content)

    def search(self, query: str, top_k: int = 5):
        if self._vendor is not None:
            result = self._vendor(query)
            if isinstance(result, str):
                return [result] if result else []
            return list(result)[:top_k]

        # Standalone: entity matching
        matched = self._match_entities(query)
        results = []
        for node in matched[:top_k]:
            text = self._get_neighborhood_text(node)
            results.append(text)
        return results

    def get_state(self):
        if self._vendor is not None:
            self._sync_graph_from_vendor()
        return self

    def reset(self) -> None:
        self._graph.clear()
        self._trust_weights.clear()
        self._relation_counts.clear()
        self._cooccurrence.clear()
        if self._vendor is not None:
            # Reset vendor state if possible
            if hasattr(self._vendor, "reset"):
                self._vendor.reset()

    # ---- GraphMemory interface ----

    def get_neighbors(self, node: str) -> list[tuple[str, str, str]]:
        edges = []
        for u, v, data in self._graph.edges(node, data=True):
            edges.append((u, data.get("relation", "related_to"), v))
        for u, v, data in self._graph.in_edges(node, data=True):
            edges.append((u, data.get("relation", "related_to"), v))
        return edges

    def get_compatible_relations(self, r: str) -> set[str]:
        compat = set()
        for (r1, r2), count in self._cooccurrence.items():
            if r1 == r and count > 0:
                compat.add(r2)
            elif r2 == r and count > 0:
                compat.add(r1)
        compat.add(r)
        return compat

    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def get_all_nodes(self) -> list[str]:
        return list(self._graph.nodes())

    def get_cooccurrence(self, r1: str, r2: str) -> int:
        return self._cooccurrence.get((r1, r2), 0) + self._cooccurrence.get((r2, r1), 0)

    def get_relation_count(self, r: str) -> int:
        return self._relation_counts.get(r, 0)

    # ---- Internal ----

    def _sync_graph_from_vendor(self):
        """Sync internal nx graph from vendor's internal state.

        G-Memory maintains three tiers: insight, query, interaction.
        We expose the interaction graph for calibration.
        """
        if self._vendor is None:
            return

        # Access vendor's internal graph
        # The exact attribute name depends on G-Memory's implementation
        for attr in ["_interaction_graph", "interaction_graph", "_graph", "graph"]:
            vendor_graph = getattr(self._vendor, attr, None)
            if vendor_graph is not None and isinstance(vendor_graph, nx.DiGraph):
                self._graph = vendor_graph
                self._rebuild_statistics()
                return

    def _rebuild_statistics(self):
        """Rebuild relation counts and co-occurrence from graph."""
        self._relation_counts.clear()
        self._cooccurrence.clear()
        for u, v, data in self._graph.edges(data=True):
            r = data.get("relation", "related_to")
            self._relation_counts[r] += 1

        for node in self._graph.nodes():
            self._update_cooccurrence(node)

    def _update_cooccurrence(self, node: str):
        rels = [data.get("relation", "") for _, _, data in self._graph.edges(node, data=True)]
        for i, r1 in enumerate(rels):
            for r2 in rels[i + 1:]:
                self._cooccurrence[(r1, r2)] += 1

    def _match_entities(self, query: str) -> list[str]:
        query_lower = query.lower()
        matched = [n for n in self._graph.nodes() if n.lower() in query_lower]
        matched.sort(key=lambda n: self._graph.degree(n), reverse=True)
        return matched

    def _get_neighborhood_text(self, node: str) -> str:
        edges = []
        for u, v, data in self._graph.edges(node, data=True):
            r = data.get("relation", "related_to")
            edges.append(f"({u}, {r}, {v})")
        return "; ".join(edges[:15])
