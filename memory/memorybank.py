"""MemoryBank memory adapter.

Wraps G-Memory's MemoryBank implementation into EquiMem's
EmbeddingMemory interface.

MemoryBank (Zhong et al., 2024) stores memory entries with
timestamps and supports forgetting via an Ebbinghaus-inspired
decay mechanism. For our purposes, the decay is secondary to
the trust-discounted retrieval from EquiMem.
"""

import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import EmbeddingMemory

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "GMemory"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))


class MemoryBankAdapter(EmbeddingMemory):
    """Wrap G-Memory's MemoryBank into EmbeddingMemory interface."""

    def __init__(self, llms_type: list[str], embed_model: str = "all-MiniLM-L6-v2"):
        self._entries_text: list[str] = []
        self._entries_vec: list[np.ndarray] = []
        self._trust_weights: list[float] = []
        self._timestamps: list[int] = []
        self._step = 0
        self._encoder = SentenceTransformer(embed_model)
        self._dim = self._encoder.get_sentence_embedding_dimension()

        self._vendor = None
        try:
            from mas.memory_memorybank import MemoryMemoryBank
            self._vendor = MemoryMemoryBank(llms_type)
        except ImportError:
            pass

    def save(self, content, trust_weight: float = 1.0) -> None:
        self._step += 1
        if isinstance(content, list):
            for item in content:
                if isinstance(item, tuple) and len(item) == 2:
                    vec, text = item
                    self._entries_vec.append(np.array(vec))
                    self._entries_text.append(text)
                else:
                    text = str(item)
                    vec = self._encoder.encode(text)
                    self._entries_vec.append(vec)
                    self._entries_text.append(text)
                self._trust_weights.append(trust_weight)
                self._timestamps.append(self._step)
        elif isinstance(content, str):
            vec = self._encoder.encode(content)
            self._entries_vec.append(vec)
            self._entries_text.append(content)
            self._trust_weights.append(trust_weight)
            self._timestamps.append(self._step)

    def search(self, query: str, top_k: int = 5):
        if not self._entries_vec:
            return []
        q_vec = self._encoder.encode(query)
        sims = self.similarities(q_vec)
        weighted_sims = sims * np.sqrt(np.array(self._trust_weights))
        top_idx = np.argsort(weighted_sims)[::-1][:top_k]
        return [self._entries_text[i] for i in top_idx]

    def search_with_delta(self, query: str, delta: list, top_k: int) -> list[str]:
        orig_len = len(self._entries_vec)
        for vec, text in delta:
            self._entries_vec.append(np.array(vec))
            self._entries_text.append(text)
            self._trust_weights.append(1.0)
            self._timestamps.append(self._step)

        results = self.search(query, top_k)

        self._entries_vec = self._entries_vec[:orig_len]
        self._entries_text = self._entries_text[:orig_len]
        self._trust_weights = self._trust_weights[:orig_len]
        self._timestamps = self._timestamps[:orig_len]
        return results

    def similarities(self, vector: np.ndarray) -> np.ndarray:
        if not self._entries_vec:
            return np.array([])
        vecs = np.array(self._entries_vec)
        norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(vector)
        return vecs @ vector / (norms + 1e-8)

    @property
    def vectors(self) -> np.ndarray:
        if not self._entries_vec:
            return np.zeros((0, self._dim))
        return np.array(self._entries_vec)

    @property
    def entries(self) -> list[str]:
        return list(self._entries_text)

    def get_state(self):
        return self

    def reset(self) -> None:
        self._entries_text.clear()
        self._entries_vec.clear()
        self._trust_weights.clear()
        self._timestamps.clear()
        self._step = 0
