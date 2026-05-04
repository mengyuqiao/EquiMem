"""Wrap G-Memory's memory implementations into EquiMem's interface."""

import sys
sys.path.insert(0, "vendor/GMemory")

# Import G-Memory's memory classes directly
from mas.memory_voyager import MemoryVoyager
from mas.memory_memorybank import MemoryMemoryBank
from mas.memory_gmemory import MemoryGMemory

from memory.base import MemoryBase, EmbeddingMemory, GraphMemory


class VoyagerAdapter(EmbeddingMemory):
    """Wraps G-Memory's Voyager into EquiMem's EmbeddingMemory interface."""

    def __init__(self, llms_type: list[str]):
        self._inner = MemoryVoyager(llms_type)
        self._entries = []       # track stored entries
        self._vectors = []       # track embeddings
        self._encoder = None     # lazy-init MiniLM

    def save(self, content: str, trust_weight: float = 1.0) -> None:
        # Use G-Memory's internal save
        self._inner(content)  # G-Memory's __call__ does save+retrieve
        self._entries.append((content, trust_weight))

    def search(self, query: str, top_k: int = 5) -> list[str]:
        # Use G-Memory's internal retrieval
        result = self._inner(query)
        # Parse the returned string into a list
        return self._parse_retrieval(result, top_k)

    def search_with_delta(self, query, delta, top_k):
        # Simulate merge: temporarily add delta, search, remove
        for vec, text in delta:
            self._inner._store.add(text)
        results = self.search(query, top_k)
        # Roll back (or use a copy)
        return results

    def similarities(self, vector):
        # Compute cosine similarities to all stored vectors
        import numpy as np
        if not self._vectors:
            return np.array([])
        vecs = np.array(self._vectors)
        sims = vecs @ vector / (
            np.linalg.norm(vecs, axis=1) * np.linalg.norm(vector) + 1e-8
        )
        return sims

    @property
    def vectors(self):
        return self._vectors


class MemoryBankAdapter(EmbeddingMemory):
    """Same pattern as VoyagerAdapter, wrapping MemoryMemoryBank."""

    def __init__(self, llms_type):
        self._inner = MemoryMemoryBank(llms_type)
        # ... same interface methods ...


class GMemoryAdapter(GraphMemory):
    """Wraps G-Memory's own G-Memory into EquiMem's GraphMemory interface."""

    def __init__(self, llms_type):
        self._inner = MemoryGMemory(llms_type)
        # G-Memory internally maintains insight/query/interaction graphs
        # We expose the interaction graph for calibration

    def save(self, content, trust_weight=1.0):
        self._inner(content)

    def search(self, query, top_k=5):
        return self._parse_retrieval(self._inner(query), top_k)

    def get_neighbors(self, node):
        # Access G-Memory's internal graph
        return list(self._inner._interaction_graph.edges(node, data=True))

    def get_compatible_relations(self, r):
        # Compute from co-occurrence in the interaction graph
        return self._compute_compat(r)

    def num_nodes(self):
        return self._inner._interaction_graph.number_of_nodes()