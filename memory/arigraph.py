"""AriGraph memory adapter (Anokhin et al., 2024).

AriGraph maintains a semantic knowledge graph (entity-relation
triples) plus episodic memory. Both are incrementally updated
as agents produce reasoning traces during debate.

This is a NEW adapter not present in G-Memory's codebase.
It implements the GraphMemory interface for EquiMem's
graph calibration (s_local, s_path, rho_align).
"""

import math
from collections import defaultdict

import networkx as nx
import ollama

from .base import GraphMemory


EXTRACT_PROMPT = """Extract all entity-relation triples from the text below.
Format each triple as: subject | relation | object
One triple per line. No other text.

Text: {text}

Triples:"""


class AriGraphAdapter(GraphMemory):
    """Semantic KG + episodic memory with incremental construction."""

    def __init__(self, llms_type: list[str]):
        self._model = llms_type[0] if llms_type else "qwen3:8b"
        self._graph = nx.DiGraph()
        self._episodes: list[dict] = []
        self._trust_weights: dict[tuple, float] = {}
        self._relation_counts: dict[str, int] = defaultdict(int)
        self._cooccurrence: dict[tuple, int] = defaultdict(int)

    # ---- MemoryBase interface ----

    def save(self, content, trust_weight: float = 1.0) -> None:
        if isinstance(content, list):
            # Graph delta: list of (u, r, v) triples
            for item in content:
                if isinstance(item, tuple) and len(item) == 3:
                    self._add_edge(item, trust_weight)
                else:
                    self._extract_and_add(str(item), trust_weight)
        elif isinstance(content, str):
            self._extract_and_add(content, trust_weight)

    def search(self, query: str, top_k: int = 5):
        # Entity matching + 2-hop neighborhood retrieval
        matched = self._match_entities(query)
        results = []
        for node in matched[:top_k]:
            neighborhood = self._get_neighborhood_text(node, max_hops=2)
            results.append(neighborhood)

        # Also include recent episodes
        recent = self._episodes[-top_k:]
        for ep in recent:
            results.append(ep["content"][:300])

        return results[:top_k]

    def get_state(self):
        return self

    def reset(self) -> None:
        self._graph.clear()
        self._episodes.clear()
        self._trust_weights.clear()
        self._relation_counts.clear()
        self._cooccurrence.clear()

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

    # ---- Internal methods ----

    def _add_edge(self, triple: tuple, trust_weight: float):
        u, r, v = triple
        self._graph.add_edge(u, v, relation=r)
        self._trust_weights[(u, r, v)] = trust_weight
        self._relation_counts[r] += 1
        self._update_cooccurrence(u)

    def _extract_and_add(self, text: str, trust_weight: float):
        """Extract triples via LLM and add to graph."""
        self._episodes.append({
            "content": text,
            "timestamp": len(self._episodes),
        })

        triples = self._extract_triples(text)
        for triple in triples:
            self._add_edge(triple, trust_weight)

    def _extract_triples(self, text: str) -> list[tuple]:
        prompt = EXTRACT_PROMPT.format(text=text)
        try:
            response = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 256},
            )
            raw = response.get("message", {}).get("content", "")
        except Exception:
            return []

        triples = []
        for line in raw.strip().split("\n"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3 and all(parts):
                triples.append(tuple(parts))
        return triples

    def _match_entities(self, query: str) -> list[str]:
        """Find graph nodes mentioned in the query."""
        query_lower = query.lower()
        matched = []
        for node in self._graph.nodes():
            if node.lower() in query_lower or query_lower in node.lower():
                matched.append(node)
        # Sort by degree (most connected first)
        matched.sort(key=lambda n: self._graph.degree(n), reverse=True)
        return matched

    def _get_neighborhood_text(self, node: str, max_hops: int = 2) -> str:
        """Serialize the local neighborhood as text."""
        edges = set()
        visited = {node}
        frontier = [node]
        for _ in range(max_hops):
            next_frontier = []
            for n in frontier:
                for u, v, data in self._graph.edges(n, data=True):
                    r = data.get("relation", "related_to")
                    edges.add(f"({u}, {r}, {v})")
                    if v not in visited:
                        visited.add(v)
                        next_frontier.append(v)
            frontier = next_frontier
        return "; ".join(list(edges)[:20])

    def _update_cooccurrence(self, node: str):
        """Update relation co-occurrence stats for edges sharing this node."""
        rels = [data.get("relation", "") for _, _, data in self._graph.edges(node, data=True)]
        for i, r1 in enumerate(rels):
            for r2 in rels[i + 1:]:
                self._cooccurrence[(r1, r2)] += 1
