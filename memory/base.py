"""Base interfaces for memory adapters.

All memory architectures implement MemoryBase. Embedding and
graph memories extend it with architecture-specific methods
needed by EquiMem's calibration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MemoryState:
    """Snapshot of memory state passed to calibration."""
    entries: list = field(default_factory=list)
    vectors: np.ndarray = None
    metadata: dict = None


class MemoryBase(ABC):
    """Interface that all memory adapters must implement."""

    @abstractmethod
    def save(self, content, trust_weight: float = 1.0) -> None:
        """Write an entry to memory with a trust weight."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k relevant entries."""
        ...

    @abstractmethod
    def search_with_delta(self, query: str, delta, top_k: int = 5) -> list[str]:
        """Simulate retrieval after merging delta (for rho_align)."""
        ...

    @abstractmethod
    def get_state(self) -> Any:
        """Return current state for calibration scoring."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all entries (called between tasks)."""
        ...


class EmbeddingMemory(MemoryBase):
    """Extended interface for embedding (vector) memories.

    Adds methods needed by EmbeddingCalibrator:
    - similarities(): cosine similarity to all stored vectors
    - vectors: access to raw embedding matrix
    """

    @abstractmethod
    def similarities(self, vector: np.ndarray) -> np.ndarray:
        """Return cosine similarities between vector and all stored entries."""
        ...

    @property
    @abstractmethod
    def vectors(self) -> np.ndarray:
        """All stored embedding vectors as (N, d) array."""
        ...


class GraphMemory(MemoryBase):
    """Extended interface for graph (structural) memories.

    Adds methods needed by GraphCalibrator:
    - get_neighbors(): edges incident to a node
    - get_compatible_relations(): R_compat from co-occurrence
    - num_nodes(): graph size for L_max computation
    """

    @abstractmethod
    def get_neighbors(self, node: str) -> list[tuple[str, str, str]]:
        """Return edges incident to node as (u, r, v) triples."""
        ...

    @abstractmethod
    def get_compatible_relations(self, r: str) -> set[str]:
        """Return R_compat(r) from co-occurrence statistics."""
        ...

    @abstractmethod
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        ...

    @abstractmethod
    def get_cooccurrence(self, r1: str, r2: str) -> int:
        """Count of nodes where r1 and r2 co-occur as edge types."""
        ...

    @abstractmethod
    def get_relation_count(self, r: str) -> int:
        """Total number of edges with relation type r."""
        ...

    @abstractmethod
    def get_all_nodes(self) -> list[str]:
        """Return all node IDs."""
        ...
